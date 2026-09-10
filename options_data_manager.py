"""
OptionsDataManager — fetch & compute volatility/options-derived features
from free data sources (VIX term structure, SKEW, OHLCV vol estimators).

Architecture:
    OptionsDataManager
    ├── __init__()                        → setup
    ├── fetch_market_vol_data()           → download VIX term structure + SKEW + SPY
    │   └── _download_with_retry()        → yfinance download with error handling
    ├── build_market_features()           → market-level features (same for all stocks)
    │   ├── _vix_term_structure()         → slope, ratio, contango, curvature (8)
    │   ├── _vix_regime()                 → level, regime bucket, momentum, percentile (8)
    │   ├── _skew_features()              → SKEW level, momentum, VIX-SKEW interaction (4)
    │   └── _volatility_risk_premium()    → VRP at multiple horizons (6)
    ├── build_stock_features()            → per-stock vol estimators from OHLCV
    │   ├── _parkinson_vol()              → high-low range estimator
    │   ├── _garman_klass_vol()           → OHLC estimator
    │   ├── _yang_zhang_vol()             → overnight + intraday estimator
    │   └── _vol_spreads()                → spreads between estimators
    ├── save(path) / load(path)           → feather persistence
    └── merge_to_stock_data()             → broadcast market features to (ticker, time) index

Usage:
    odm = OptionsDataManager()
    odm.fetch_market_vol_data('2015-01-01', '2026-03-11')
    market_features = odm.build_market_features()          # DatetimeIndex
    stock_features  = odm.build_stock_features(prices_df)  # (ticker, time) MultiIndex
    combined = odm.merge_to_stock_data(market_features, stock_features)
"""

import numpy as np
import pandas as pd
import yfinance as yf


class OptionsDataManager:

    MONTH = 21
    YEAR  = 252
    QUARTER = 63

    def __init__(self):
        self._raw = {}          # raw downloaded data
        self._market_feat = {}  # market-level features
        self._stock_feat = {}   # stock-level features

    # ══════════════════════════════════════════════════════════════════════════
    # FETCHING
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_market_vol_data(
        self, start: str, end: str
    ) -> dict[str, pd.DataFrame]:
        """
        Download VIX term structure, SKEW, and SPY from yfinance.

        Parameters
        ----------
        start, end : str (YYYY-MM-DD)

        Returns
        -------
        dict of DataFrames keyed by name
        """
        print("Fetching market volatility data...")

        # VIX term structure (4 tenors)
        vix_tickers = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M"]
        vix_df = self._download_with_retry(vix_tickers, start, end)
        if vix_df is not None:
            # yfinance sorts alphabetically: ^VIX, ^VIX3M, ^VIX6M, ^VIX9D
            vix_df.columns = ['VIX_30D', 'VIX_3M', 'VIX_6M', 'VIX_9D']
            self._raw['vix_term'] = vix_df
            print(f"  VIX term structure: {len(vix_df)} days")

        # SKEW index
        skew_df = self._download_with_retry(['^SKEW'], start, end)
        if skew_df is not None:
            skew_df.columns = ['SKEW']
            self._raw['skew'] = skew_df
            print(f"  SKEW: {len(skew_df)} days")

        # SPY for realised vol computation
        spy_df = self._download_with_retry(['SPY'], start, end)
        if spy_df is not None:
            spy_df.columns = ['SPY']
            self._raw['spy'] = spy_df
            print(f"  SPY: {len(spy_df)} days")

        print(f"✅ Market data fetched "
              f"({self._raw['vix_term'].index[0].date()} → "
              f"{self._raw['vix_term'].index[-1].date()})")

        return self._raw

    def _download_with_retry(
        self, tickers: list[str], start: str, end: str, retries: int = 2
    ) -> pd.DataFrame | None:
        """Download close prices from yfinance with retry logic."""
        for attempt in range(retries + 1):
            try:
                df = yf.download(
                    tickers, start=start, end=end,
                    auto_adjust=True, progress=False
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df = df['Close']
                elif 'Close' in df.columns:
                    df = df[['Close']]

                if isinstance(df, pd.Series):
                    df = df.to_frame()

                return df
            except Exception as e:
                if attempt < retries:
                    print(f"    Retry {attempt + 1}...")
                else:
                    print(f"    Failed: {e}")
                    return None

    # ══════════════════════════════════════════════════════════════════════════
    # MARKET-LEVEL FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    def build_market_features(self) -> pd.DataFrame:
        """
        Build all market-level features from downloaded data.

        Returns
        -------
        pd.DataFrame with DatetimeIndex
        """
        print("Building market-level features...")
        features = {}

        features.update(self._vix_term_structure())
        features.update(self._vix_regime())
        features.update(self._skew_features())
        features.update(self._volatility_risk_premium())
        features.update(self._cross_market_interactions())
        self._market_feat = features

        result = pd.DataFrame(features)
        result = result.replace([np.inf, -np.inf], np.nan)

        print(f"✅ Built {result.shape[1]} market-level features")
        print(f"   NaN rate: {result.isna().mean().mean():.1%}")

        return result

    def _vix_term_structure(self) -> dict[str, pd.Series]:
        """
        VIX term structure features — slope, ratio, contango, curvature.

        The term structure tells the model HOW LONG the market expects
        current conditions to last:
          - Contango (normal): short VIX < long VIX → calm, uncertainty grows with time
          - Backwardation (crisis): short VIX > long VIX → crisis NOW, will normalise
          - Flat: market unsure about persistence
        """
        print("  VIX term structure (13 features)...")
        vix = self._raw['vix_term']
        f = {}

        # Spreads between tenors (in VIX points)
        f['mkt_vix_slope_9d_30d'] = vix['VIX_30D'] - vix['VIX_9D']
        f['mkt_vix_slope_30d_3m'] = vix['VIX_3M']  - vix['VIX_30D']
        f['mkt_vix_slope_3m_6m']  = vix['VIX_6M']  - vix['VIX_3M']
        f['mkt_vix_slope_full']   = vix['VIX_6M']  - vix['VIX_9D']

        # Ratios (more stable than spreads for XGBoost)
        f['mkt_vix_ratio_9d_30d'] = vix['VIX_9D'] / (vix['VIX_30D'] + 1e-9)
        f['mkt_vix_ratio_30d_3m'] = vix['VIX_30D'] / (vix['VIX_3M'] + 1e-9)

        # Contango / backwardation flag (binary)
        f['mkt_vix_contango'] = (f['mkt_vix_slope_30d_3m'] > 0).astype(float)

        # Curvature: is the middle of the curve humped or dipped?
        # Positive = belly is rich (convex), negative = belly is cheap (concave)
        f['mkt_vix_curvature'] = (
            (vix['VIX_9D'] + vix['VIX_6M']) / 2
            - (vix['VIX_30D'] + vix['VIX_3M']) / 2
        )

        # Additional shape descriptors
        f['mkt_vix_ratio_9d_6m'] = vix['VIX_9D'] / (vix['VIX_6M'] + 1e-9)
        f['mkt_vix_curve_twist'] = f['mkt_vix_slope_9d_30d'] - f['mkt_vix_slope_30d_3m']
        f['mkt_vix_inversion_9d_30d'] = (vix['VIX_9D'] > vix['VIX_30D']).astype(float)
        f['mkt_vix_inversion_30d_3m'] = (vix['VIX_30D'] > vix['VIX_3M']).astype(float)

        slope_ma = f['mkt_vix_slope_9d_30d'].rolling(self.QUARTER).mean()
        slope_std = f['mkt_vix_slope_9d_30d'].rolling(self.QUARTER).std()
        f['mkt_vix_slope_9d_30d_z63'] = (f['mkt_vix_slope_9d_30d'] - slope_ma) / (slope_std + 1e-9)

        return f

    def _vix_regime(self) -> dict[str, pd.Series]:
        """
        VIX level and regime features.

        These tell the model the CURRENT volatility environment — the model
        can learn to use different strategies in calm vs fearful markets.
        """
        print("  VIX regime (11 features)...")
        vix_30d = self._raw['vix_term']['VIX_30D']
        f = {}

        # Level
        f['mkt_vix_level']  = vix_30d

        # Regime bucket (0=calm → 4=panic)
        f['mkt_vix_regime'] = pd.cut(
            vix_30d,
            bins=[0, 15, 20, 25, 30, 200],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        # Momentum (is fear rising or falling?)
        f['mkt_vix_ret_1d']  = vix_30d.pct_change(1)
        f['mkt_vix_ret_5d']  = vix_30d.pct_change(5)
        f['mkt_vix_ret_21d'] = vix_30d.pct_change(self.MONTH)

        # Mean reversion
        vix_ma = vix_30d.rolling(self.MONTH).mean()
        f['mkt_vix_vs_ma21'] = vix_30d / (vix_ma + 1e-9) - 1

        # Percentile within trailing 1-year (where is VIX in its own range?)
        f['mkt_vix_pctile_1y'] = vix_30d.rolling(self.YEAR).rank(pct=True)

        # Vol-of-VIX (VVIX proxy — uncertainty about uncertainty)
        f['mkt_vvix_proxy'] = vix_30d.pct_change(1).rolling(self.MONTH).std()

        # Additional level-state descriptors
        vix_std_63d = vix_30d.rolling(self.QUARTER).std()
        vix_ma_63d = vix_30d.rolling(self.QUARTER).mean()
        f['mkt_vix_level_z63'] = (vix_30d - vix_ma_63d) / (vix_std_63d + 1e-9)
        f['mkt_vix_regime_change_5d'] = f['mkt_vix_regime'].diff(5)
        f['mkt_vix_drawup_63d'] = vix_30d / (vix_30d.rolling(self.QUARTER).min() + 1e-9) - 1

        return f

    def _skew_features(self) -> dict[str, pd.Series]:
        """
        CBOE SKEW index features.

        SKEW measures how asymmetric the options market's expectations are
        (left tail vs right tail). Useful as a regime descriptor, not a
        crash predictor — behaves counterintuitively during crises.
        """
        print("  SKEW (7 features)...")
        f = {}

        if 'skew' not in self._raw:
            print("    SKEW data not available, skipping...")
            return f

        skew = self._raw['skew']['SKEW']

        f['mkt_skew_level']    = skew
        f['mkt_skew_ret_5d']   = skew.pct_change(5)
        f['mkt_skew_pctile_1y'] = skew.rolling(self.YEAR).rank(pct=True)

        # VIX-SKEW interaction: high VIX + high SKEW = different from high VIX + low SKEW
        vix_30d = self._raw['vix_term']['VIX_30D']
        f['mkt_vix_skew_ratio'] = vix_30d / (skew + 1e-9)

        skew_ma = skew.rolling(self.MONTH).mean()
        skew_std = skew.rolling(self.QUARTER).std()
        f['mkt_skew_vs_ma21'] = skew / (skew_ma + 1e-9) - 1
        f['mkt_skew_z63'] = (skew - skew.rolling(self.QUARTER).mean()) / (skew_std + 1e-9)
        f['mkt_skew_momentum_21d'] = skew.pct_change(self.MONTH)

        return f

    def _volatility_risk_premium(self) -> dict[str, pd.Series]:
        """
        Volatility Risk Premium = VIX (implied) − Realised Vol.

        Positive VRP means options are 'expensive' relative to what actually
        happens. Historically positive ~84% of days — the market charges a
        fear premium. Negative VRP → realised vol exceeds expectations → crisis.
        """
        print("  Volatility Risk Premium (9 features)...")
        f = {}

        if 'spy' not in self._raw:
            print("    SPY data not available, skipping VRP...")
            return f

        spy_ret = self._raw['spy']['SPY'].pct_change()
        vix_30d = self._raw['vix_term']['VIX_30D']

        # Realised vol (annualised, in %)
        rv_21d = spy_ret.rolling(self.MONTH).std() * np.sqrt(self.YEAR) * 100
        rv_63d = spy_ret.rolling(3 * self.MONTH).std() * np.sqrt(self.YEAR) * 100

        f['mkt_rv_21d'] = rv_21d
        f['mkt_rv_63d'] = rv_63d

        # VRP = implied − realised (in VIX points)
        f['mkt_vrp_21d'] = vix_30d - rv_21d
        f['mkt_vrp_63d'] = vix_30d - rv_63d

        # VRP regime (binary)
        f['mkt_vrp_positive'] = (f['mkt_vrp_21d'] > 0).astype(float)

        # VRP z-score (normalised — how extreme is current VRP?)
        vrp_ma  = pd.Series(f['mkt_vrp_21d']).rolling(3 * self.MONTH).mean()
        vrp_std = pd.Series(f['mkt_vrp_21d']).rolling(3 * self.MONTH).std()
        f['mkt_vrp_zscore'] = (f['mkt_vrp_21d'] - vrp_ma) / (vrp_std + 1e-9)
        f['mkt_vrp_ratio_21d'] = vix_30d / (rv_21d + 1e-9)
        f['mkt_vrp_ratio_63d'] = vix_30d / (rv_63d + 1e-9)
        f['mkt_vrp_change_5d'] = f['mkt_vrp_21d'].diff(5)

        return f

    def _cross_market_interactions(self) -> dict[str, pd.Series]:
        """
        Cross-market interaction features from VIX term, SKEW, and SPY.

        These try to capture market states where the level of fear, the shape
        of the volatility curve, and realized market behavior disagree.
        """
        print("  Cross-market interactions (7 features)...")
        f = {}

        if 'spy' not in self._raw:
            return f

        spy = self._raw['spy']['SPY']
        spy_ret = spy.pct_change()
        vix_30d = self._raw['vix_term']['VIX_30D']
        slope = self._raw['vix_term']['VIX_3M'] - self._raw['vix_term']['VIX_30D']

        # Market drawdown state
        rolling_peak = spy.rolling(self.QUARTER).max()
        f['mkt_spy_drawdown_63d'] = spy / (rolling_peak + 1e-9) - 1
        f['mkt_spy_ret_5d'] = spy.pct_change(5)

        # Downside / upside realized vol split
        downside = spy_ret.where(spy_ret < 0, 0.0)
        upside = spy_ret.where(spy_ret > 0, 0.0)
        down_vol = downside.rolling(self.MONTH).std() * np.sqrt(self.YEAR) * 100
        up_vol = upside.rolling(self.MONTH).std() * np.sqrt(self.YEAR) * 100
        f['mkt_downside_rv_21d'] = down_vol
        f['mkt_upside_rv_21d'] = up_vol
        f['mkt_down_up_rv_ratio'] = down_vol / (up_vol + 1e-9)

        # When VIX is high but curve is calm, or vice versa
        f['mkt_vix_x_term'] = vix_30d * slope

        if 'skew' in self._raw:
            skew = self._raw['skew']['SKEW']
            f['mkt_vix_x_skew'] = vix_30d * skew

        return f

    # ══════════════════════════════════════════════════════════════════════════
    # STOCK-LEVEL FEATURES (from OHLCV)
    # ══════════════════════════════════════════════════════════════════════════

    def build_stock_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-stock volatility estimators from OHLCV data.

        Parameters
        ----------
        prices : pd.DataFrame with MultiIndex (ticker, time)
                 Must have columns: close, open, high, low

        Returns
        -------
        pd.DataFrame with (ticker, time) MultiIndex
        """
        print("Building stock-level vol features...")
        f = {}

        close = prices['close']
        open_ = prices['open']
        high  = prices['high']
        low   = prices['low']

        by = lambda s, func: s.groupby(level='ticker').transform(func)
        daily_ret = by(close, lambda x: x.pct_change())

        # ── Close-to-close vol (baseline) ─────────────────────────────────
        cc_vol = by(daily_ret, lambda x: x.rolling(self.MONTH).std()) * np.sqrt(self.YEAR)
        f['stk_cc_vol'] = cc_vol

        # ── Parkinson volatility (high-low range estimator) ───────────────
        log_hl_sq = (np.log(high / (low + 1e-9))) ** 2
        f['stk_parkinson_vol'] = by(
            log_hl_sq,
            lambda x: np.sqrt(x.rolling(self.MONTH).mean() / (4 * np.log(2)))
        ) * np.sqrt(self.YEAR)

        # ── Garman-Klass volatility (OHLC) ───────────────────────────────
        log_hl2 = np.log(high / (low + 1e-9)) ** 2
        log_co2 = np.log(close / (open_ + 1e-9)) ** 2
        gk_var  = 0.5 * log_hl2 - (2 * np.log(2) - 1) * log_co2
        f['stk_garman_klass_vol'] = by(
            gk_var,
            lambda x: np.sqrt(x.clip(lower=0).rolling(self.MONTH).mean())
        ) * np.sqrt(self.YEAR)

        # ── Yang-Zhang volatility (overnight + intraday, most efficient) ──
        log_oc = np.log(open_ / (close.groupby(level='ticker').shift(1) + 1e-9))
        log_co = np.log(close / (open_ + 1e-9))
        n = self.MONTH
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        ov = by(log_oc, lambda x: x.rolling(n).var())         # overnight var
        cv = by(log_co, lambda x: x.rolling(n).var())         # intraday var
        rs = by(log_hl_sq, lambda x: x.rolling(n).mean()) / (4 * np.log(2))  # Rogers-Satchell
        yz_var = ov + k * cv + (1 - k) * rs
        f['stk_yang_zhang_vol'] = np.sqrt(yz_var.clip(lower=0)) * np.sqrt(self.YEAR)

        # ── Vol spreads (capture different risk characteristics) ──────────
        # Parkinson > CC → large intraday swings that reverse by close
        f['stk_vol_spread_pk_cc'] = f['stk_parkinson_vol'] - cc_vol

        # Ratio (normalised spread)
        f['stk_vol_ratio_pk_cc'] = f['stk_parkinson_vol'] / (cc_vol + 1e-9)

        # Yang-Zhang vs CC — captures overnight gap contribution
        f['stk_vol_spread_yz_cc'] = f['stk_yang_zhang_vol'] - cc_vol

        # ── Realised vol term structure (stock-level) ─────────────────────
        cc_vol_63d = by(daily_ret, lambda x: x.rolling(3*self.MONTH).std()) * np.sqrt(self.YEAR)
        f['stk_vol_term_slope'] = cc_vol - cc_vol_63d   # positive = vol expanding

        # ── Vol regime change speed (how fast is vol moving?) ─────────────
        f['stk_vol_momentum'] = by(cc_vol, lambda x: x.pct_change(5))

        # ── Additional stock-level volatility structure features ──────────
        cc_vol_63d = by(daily_ret, lambda x: x.rolling(3*self.MONTH).std()) * np.sqrt(self.YEAR)
        cc_vol_126d = by(daily_ret, lambda x: x.rolling(6*self.MONTH).std()) * np.sqrt(self.YEAR)

        f['stk_cc_vol_63d'] = cc_vol_63d
        f['stk_cc_vol_126d'] = cc_vol_126d
        f['stk_vol_term_ratio_21_63'] = cc_vol / (cc_vol_63d + 1e-9)
        f['stk_vol_term_ratio_21_126'] = cc_vol / (cc_vol_126d + 1e-9)
        f['stk_vol_ratio_yz_cc'] = f['stk_yang_zhang_vol'] / (cc_vol + 1e-9)
        f['stk_vol_ratio_gk_cc'] = f['stk_garman_klass_vol'] / (cc_vol + 1e-9)
        f['stk_vol_spread_yz_pk'] = f['stk_yang_zhang_vol'] - f['stk_parkinson_vol']

        # Overnight vs intraday contribution
        overnight_vol = np.sqrt(ov.clip(lower=0)) * np.sqrt(self.YEAR)
        intraday_vol = np.sqrt(cv.clip(lower=0)) * np.sqrt(self.YEAR)
        f['stk_overnight_vol'] = overnight_vol
        f['stk_intraday_vol'] = intraday_vol
        f['stk_overnight_intraday_ratio'] = overnight_vol / (intraday_vol + 1e-9)

        # Upside / downside realized volatility
        downside = daily_ret.where(daily_ret < 0, 0.0)
        upside = daily_ret.where(daily_ret > 0, 0.0)
        downside_vol = by(downside, lambda x: x.rolling(self.MONTH).std()) * np.sqrt(self.YEAR)
        upside_vol = by(upside, lambda x: x.rolling(self.MONTH).std()) * np.sqrt(self.YEAR)
        f['stk_downside_vol_21d'] = downside_vol
        f['stk_upside_vol_21d'] = upside_vol
        f['stk_down_up_vol_ratio'] = downside_vol / (upside_vol + 1e-9)

        # Volatility-of-volatility and range persistence
        f['stk_vol_of_vol_63d'] = by(cc_vol, lambda x: x.rolling(3*self.MONTH).std())
        daily_range_pct = (high - low) / (close + 1e-9)
        f['stk_range_pct_ma21'] = by(daily_range_pct, lambda x: x.rolling(self.MONTH).mean())
        f['stk_range_pct_z63'] = by(
            daily_range_pct,
            lambda x: (x - x.rolling(3*self.MONTH).mean()) / (x.rolling(3*self.MONTH).std() + 1e-9)
        )
        self._stock_feat = f

        result = pd.DataFrame(f, index=prices.index)
        result = result.replace([np.inf, -np.inf], np.nan)

        print(f"✅ Built {result.shape[1]} stock-level vol features")
        return result

    def feature_names(self) -> dict[str, list[str]]:
        """Return the last-built market and stock feature names."""
        return {
            'market': sorted(self._market_feat.keys()) if self._market_feat else [],
            'stock': sorted(self._stock_feat.keys()) if self._stock_feat else [],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # MERGING & PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def merge_to_stock_data(
        self,
        market_features: pd.DataFrame,
        stock_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Broadcast market-level features (DatetimeIndex) to match the
        (ticker, time) MultiIndex of stock features, then concat.

        Parameters
        ----------
        market_features : pd.DataFrame with DatetimeIndex
        stock_features  : pd.DataFrame with (ticker, time) MultiIndex

        Returns
        -------
        pd.DataFrame with (ticker, time) MultiIndex, all columns combined
        """
        print("Merging market + stock features...")

        # Get the time index from stock features
        time_values = stock_features.index.get_level_values('time')

        # Map market features to each (ticker, time) row via its time component
        market_aligned = market_features.reindex(time_values)
        market_aligned.index = stock_features.index  # overwrite with MultiIndex

        combined = pd.concat([stock_features, market_aligned], axis=1)
        print(f"✅ Combined: {combined.shape[1]} features "
              f"({stock_features.shape[1]} stock + {market_features.shape[1]} market)")

        return combined

    def save(self, df: pd.DataFrame, path: str):
        """Save DataFrame to feather format."""
        df.reset_index().to_feather(path)
        print(f"💾 Saved to {path}")

    def load(self, path: str) -> pd.DataFrame:
        """Load DataFrame from feather format."""
        df = pd.read_feather(path)
        if 'ticker' in df.columns and 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index(['ticker', 'time']).sort_index()
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
        return df


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import os

    odm = OptionsDataManager()

    # 1. Fetch market data
    odm.fetch_market_vol_data('2015-01-01', '2026-03-11')

    # 2. Build market features
    mkt = odm.build_market_features()
    print(f"\nMarket features: {mkt.shape}")
    print(mkt.tail(5).to_string())

    # 3. Build stock features (using a small test set)
    print("\n" + "=" * 80)
    print("STOCK-LEVEL DEMO")
    print("=" * 80)

    # Load existing price data if available
    price_file = 'result/sp500_OHLCV.feather'
    if os.path.exists(price_file):
        prices = pd.read_feather(price_file)
        if 'ticker' in prices.columns:
            time_col = 'time' if 'time' in prices.columns else 'date'
            prices[time_col] = pd.to_datetime(prices[time_col])
            prices = prices.set_index(['ticker', time_col]).sort_index()
            if time_col == 'date':
                prices.index.names = ['ticker', 'time']

        # Use only 3 tickers for demo
        demo_tickers = prices.index.get_level_values('ticker').unique()[:3]
        demo_prices = prices.loc[demo_tickers]
        print(f"Using {len(demo_tickers)} tickers from {price_file}")

        stk = odm.build_stock_features(demo_prices)
        print(f"\nStock features: {stk.shape}")
        print(stk.tail(5).to_string())

        # 4. Merge
        combined = odm.merge_to_stock_data(mkt, stk)
        print(f"\nCombined: {combined.shape}")

        # Save
        os.makedirs('result', exist_ok=True)
        odm.save(mkt, 'result/options_market_features.feather')
        odm.save(combined, 'result/options_features_test.feather')
    else:
        print(f"Price file not found: {price_file}")
        print("Saving market features only.")
        os.makedirs('result', exist_ok=True)
        odm.save(mkt, 'result/options_market_features.feather')

    # 5. Summary
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    print(f"\nMarket-level ({mkt.shape[1]} features):")
    for col in mkt.columns:
        print(f"  {col}")
