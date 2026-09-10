import argparse
from io import StringIO
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.interpolate import RectBivariateSpline

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent / "code"
if CODE_DIR.exists():
    sys.path.insert(0, str(CODE_DIR))

from options_data_manager import OptionsDataManager


TENOR_COLS = ["VIX_9D", "VIX_30D", "VIX_3M", "VIX_6M"]
TENOR_DAYS = np.array([9, 30, 91, 182], dtype=float)
TENOR_LABELS = ["9D", "30D", "3M", "6M"]
PLOTLY_CONFIG = {"displayModeBar": True, "scrollZoom": True, "responsive": True}
CBOE_DAILY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
CBOE_SERIES = {
    "VIX9D": ("VIX_9D", "CLOSE"),
    "VIX": ("VIX_30D", "CLOSE"),
    "VIX3M": ("VIX_3M", "CLOSE"),
    "VIX6M": ("VIX_6M", "CLOSE"),
    "SKEW": ("SKEW", "SKEW"),
}


def parse_cboe_csv(text: str, symbol: str, value_column: str) -> pd.Series:
    frame = pd.read_csv(StringIO(text), skipinitialspace=True)
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    required = {"DATE", value_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{symbol} CSV missing columns {sorted(required - set(frame.columns))}")

    dates = pd.to_datetime(frame["DATE"], format="%m/%d/%Y", errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name=symbol).dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if len(series) < 30 or (series <= 0).any():
        raise ValueError(f"{symbol} CSV failed row-count or positive-value validation")
    return series


def fetch_cboe_series(
    session: requests.Session,
    symbol: str,
    value_column: str,
    cache_dir: Path,
    retries: int = 2,
) -> pd.Series:
    cache_path = cache_dir / f"{symbol}_History.csv"
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(CBOE_DAILY_URL.format(symbol=symbol), timeout=30)
            response.raise_for_status()
            series = parse_cboe_csv(response.text, symbol, value_column)
            cache_dir.mkdir(parents=True, exist_ok=True)
            response_frame = pd.DataFrame({"DATE": series.index.strftime("%m/%d/%Y"), value_column: series.values})
            response_frame.to_csv(cache_path, index=False)
            return series
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(attempt + 1)

    if cache_path.exists():
        print(f"  {symbol}: fresh download failed; using cached history")
        return parse_cboe_csv(cache_path.read_text(encoding="utf-8"), symbol, value_column)
    raise RuntimeError(f"Unable to fetch {symbol} from Cboe and no cache is available") from error


def fetch_spy_close(
    start: str,
    end: str,
    cache_dir: Path,
) -> pd.DataFrame:
    cache_path = cache_dir / "SPY_yfinance_daily.csv"
    legacy_cache_path = cache_dir / "SPY_api_daily.csv"

    try:
        downloaded = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if downloaded.empty:
            raise RuntimeError("yfinance returned no SPY bars")
        close = downloaded["Close"]
        if isinstance(close, pd.DataFrame):
            close = close["SPY"]
        spy = close.rename("SPY").to_frame()
        spy.index = pd.to_datetime(spy.index).normalize()
        spy = spy.dropna().sort_index()
        cache_dir.mkdir(parents=True, exist_ok=True)
        spy.rename_axis("DATE").reset_index().to_csv(cache_path, index=False)
        return spy
    except Exception as error:
        available_cache = cache_path if cache_path.exists() else legacy_cache_path
        if not available_cache.exists():
            raise RuntimeError("Unable to fetch SPY bars from yfinance and no cache is available") from error
        print("  SPY: fresh yfinance download failed; using cached history")
        spy = pd.read_csv(available_cache, parse_dates=["DATE"]).set_index("DATE")
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        return spy.loc[(spy.index >= start_ts) & (spy.index < end_ts)]


def fetch_cboe_market_data(
    odm: OptionsDataManager,
    start: str,
    end: str,
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    print("Fetching official Cboe volatility histories...")
    session = requests.Session()
    session.trust_env = False

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    fetched: dict[str, pd.Series] = {}
    for symbol, (output_column, value_column) in CBOE_SERIES.items():
        series = fetch_cboe_series(session, symbol, value_column, cache_dir)
        fetched[output_column] = series.loc[(series.index >= start_ts) & (series.index < end_ts)]
        print(f"  {symbol}: {len(fetched[output_column]):,} sessions")

    vix_term = pd.concat([fetched[column] for column in TENOR_COLS], axis=1, join="inner")
    vix_term.columns = TENOR_COLS
    skew = fetched["SKEW"].rename("SKEW").to_frame()
    spy = fetch_spy_close(start, end, cache_dir)

    common_dates = vix_term.index.intersection(skew.index).intersection(spy.index)
    if common_dates.empty:
        raise RuntimeError("Cboe term indices, SKEW, and SPY have no common report date")
    latest_common = common_dates.max()

    odm._raw = {
        "vix_term": vix_term.loc[:latest_common],
        "skew": skew.loc[:latest_common],
        "spy": spy.loc[:latest_common],
    }
    print(f"  Latest common completed session: {latest_common:%Y-%m-%d}")
    return odm._raw


def parse_report_date(date_text: str | None) -> datetime:
    if date_text is None:
        return datetime.now()
    try:
        return datetime.strptime(date_text, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD format, for example 20260609") from exc


def exclusive_end_date(report_date: datetime) -> str:
    # All source filters treat end as exclusive. For non-trading dates, the
    # latest prior completed common session is used.
    return (report_date + timedelta(days=1)).strftime("%Y-%m-%d")


def pct_rank(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    if clean.empty or pd.isna(value):
        return np.nan
    return float((clean <= value).mean())


def fmt(value, digits=1, suffix=""):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def fmt_pct(value, digits=0):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}%}"


def regime_label(vix_level: float) -> tuple[str, str]:
    if pd.isna(vix_level):
        return "Unknown", "neutral"
    if vix_level < 15:
        return "Calm", "good"
    if vix_level < 20:
        return "Normal", "good"
    if vix_level < 25:
        return "Elevated", "warn"
    if vix_level < 30:
        return "High", "bad"
    return "Panic", "bad"


def classify_surface(latest, term_values):
    vix_9d, vix_30d, vix_3m, vix_6m = term_values
    slope_30_3m = latest["mkt_vix_slope_30d_3m"]
    full_slope = latest["mkt_vix_slope_full"]
    curvature = latest["mkt_vix_curvature"]

    if vix_30d > vix_3m:
        shape = "Backwardation"
        shape_class = "bad"
        shape_text = "Front volatility is priced above 3M volatility. This is the cleanest stress signal in the surface."
    elif slope_30_3m < 1.0:
        shape = "Compressed Contango"
        shape_class = "warn"
        shape_text = "The curve is still upward sloping, but the front-to-mid premium is thin. Stress is not gone."
    else:
        shape = "Contango"
        shape_class = "good"
        shape_text = "The curve is upward sloping. Markets are pricing current uncertainty as likely to decay."

    if vix_9d > vix_30d:
        front_text = "9D is above 30D, so the market is paying up for immediate event risk."
    elif vix_30d - vix_9d < 0.75:
        front_text = "The 9D to 30D spread is narrow, so near-term event risk remains meaningful."
    else:
        front_text = "The front end is relaxed relative to 30D implied volatility."

    if curvature > 0.5:
        belly_text = "Positive curvature: the wings are high relative to the 30D/3M belly."
    elif curvature < -0.5:
        belly_text = "Negative curvature: the 30D/3M belly is rich relative to the wings."
    else:
        belly_text = "Low curvature: the surface is mostly moving as a parallel curve."

    return {
        "shape": shape,
        "shape_class": shape_class,
        "shape_text": shape_text,
        "front_text": front_text,
        "belly_text": belly_text,
        "slope_30_3m": slope_30_3m,
        "full_slope": full_slope,
        "curvature": curvature,
    }


def build_surface(vix_term_df: pd.DataFrame, lookback: int):
    surface_df = vix_term_df[TENOR_COLS].dropna().tail(lookback)
    z_raw = surface_df.values.astype(float)
    y_raw = np.arange(len(surface_df), dtype=float)
    x_fine = np.linspace(TENOR_DAYS[0], TENOR_DAYS[-1], 80)
    y_fine = np.linspace(0, len(surface_df) - 1, min(len(surface_df), 220))

    try:
        spline = RectBivariateSpline(y_raw, TENOR_DAYS, z_raw, kx=min(3, len(y_raw) - 1), ky=3)
        z_fine = spline(y_fine, x_fine)
    except Exception:
        z_fine = np.vstack([np.interp(x_fine, TENOR_DAYS, row) for row in z_raw])
        y_fine = y_raw

    return surface_df, x_fine, y_fine, z_fine


def surface_figure(surface_df, x_fine, y_fine, z_fine):
    dates = surface_df.index
    z_raw = surface_df.values.astype(float)
    current_curve = np.interp(x_fine, TENOR_DAYS, z_raw[-1])

    date_hover = []
    for yi in y_fine:
        idx = int(np.clip(round(yi), 0, len(dates) - 1))
        date_hover.append(dates[idx].strftime("%Y-%m-%d"))

    hover_text = [
        [
            f"Date: {date_hover[i]}<br>Tenor: {int(round(x))}D<br>IV: {z_fine[i, j]:.2f}%"
            for j, x in enumerate(x_fine)
        ]
        for i in range(len(y_fine))
    ]

    colorscale = [
        [0.00, "#07111f"],
        [0.18, "#123c69"],
        [0.34, "#00b7c7"],
        [0.52, "#2dd4bf"],
        [0.68, "#f7c948"],
        [0.84, "#ff7a45"],
        [1.00, "#ff2e63"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_fine,
            y=y_fine,
            z=z_fine,
            colorscale=colorscale,
            opacity=0.94,
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title=dict(text="IV %", font=dict(color="#d7e3f4")),
                tickfont=dict(color="#93a4b8"),
                bgcolor="rgba(5,10,18,0.75)",
                bordercolor="rgba(148,163,184,0.25)",
                len=0.62,
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, project=dict(z=True), highlightwidth=2)
            ),
            lighting=dict(ambient=0.72, diffuse=0.48, specular=0.20, roughness=0.58),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_fine,
            y=np.full_like(x_fine, y_fine[-1]),
            z=current_curve + 0.25,
            mode="lines",
            line=dict(color="#f7c948", width=8),
            name="Current curve",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=TENOR_DAYS,
            y=np.full(len(TENOR_DAYS), y_fine[-1]),
            z=z_raw[-1] + 0.35,
            mode="markers+text",
            marker=dict(size=7, color="#f7c948", line=dict(color="#ffffff", width=1.4)),
            text=[f"{v:.1f}" for v in z_raw[-1]],
            textposition="top center",
            textfont=dict(color="#ffffff", size=12, family="SFMono-Regular, Menlo, monospace"),
            name="Current tenors",
            hovertemplate="Tenor: %{x:.0f}D<br>IV: %{z:.2f}%<extra></extra>",
        )
    )

    tick_idx = np.linspace(0, len(dates) - 1, min(8, len(dates)), dtype=int)
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(
                title="Tenor",
                tickvals=TENOR_DAYS,
                ticktext=TENOR_LABELS,
                backgroundcolor="#07111f",
                gridcolor="rgba(255,255,255,0.07)",
                titlefont=dict(color="#d7e3f4"),
                tickfont=dict(color="#93a4b8"),
            ),
            yaxis=dict(
                title="Date",
                tickvals=tick_idx,
                ticktext=[dates[i].strftime("%b %d") for i in tick_idx],
                backgroundcolor="#07111f",
                gridcolor="rgba(255,255,255,0.07)",
                titlefont=dict(color="#d7e3f4"),
                tickfont=dict(color="#93a4b8", size=10),
            ),
            zaxis=dict(
                title="Implied Volatility",
                backgroundcolor="#07111f",
                gridcolor="rgba(255,255,255,0.08)",
                titlefont=dict(color="#d7e3f4"),
                tickfont=dict(color="#93a4b8"),
            ),
            camera=dict(eye=dict(x=1.65, y=-1.95, z=0.85)),
            aspectratio=dict(x=1.0, y=1.65, z=0.72),
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(7,17,31,0.72)",
            bordercolor="rgba(148,163,184,0.22)",
            font=dict(color="#d7e3f4"),
        ),
        font=dict(color="#d7e3f4"),
    )
    return fig


def term_structure_figure(vix_term_df: pd.DataFrame):
    df = vix_term_df[TENOR_COLS].dropna()
    latest = df.iloc[-1].values.astype(float)
    traces = [("Current", latest, "#f7c948", 4, "solid")]
    if len(df) > 5:
        traces.append(("1W ago", df.iloc[-6].values.astype(float), "#38bdf8", 2, "dash"))
    if len(df) > 21:
        traces.append(("1M ago", df.iloc[-22].values.astype(float), "#a78bfa", 2, "dot"))

    fig = go.Figure()
    for name, values, color, width, dash in traces:
        fig.add_trace(
            go.Scatter(
                x=TENOR_LABELS,
                y=values,
                mode="lines+markers+text" if name == "Current" else "lines+markers",
                text=[f"{v:.1f}" for v in values] if name == "Current" else None,
                textposition="top center",
                line=dict(color=color, width=width, dash=dash),
                marker=dict(size=9, color=color, line=dict(color="#07111f", width=1)),
                name=name,
            )
        )

    fig.update_layout(
        height=355,
        margin=dict(l=38, r=20, t=12, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.72)",
        yaxis=dict(title="IV %", gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#93a4b8")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#d7e3f4", size=12)),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#d7e3f4")),
        font=dict(color="#d7e3f4"),
    )
    return fig


def time_series_figure(mkt: pd.DataFrame):
    df = mkt.tail(252 * 2)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("VIX level", "Volatility risk premium", "30D to 3M slope"),
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["mkt_vix_level"], name="VIX", line=dict(color="#f7c948", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["mkt_rv_21d"], name="21D realized vol", line=dict(color="#38bdf8", width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["mkt_vrp_21d"], name="VRP 21D", line=dict(color="#2dd4bf", width=2)), 2, 1)
    fig.add_hline(y=0, row=2, col=1, line_color="rgba(255,255,255,0.28)", line_width=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["mkt_vix_slope_30d_3m"], name="30D to 3M", line=dict(color="#a78bfa", width=2)), 3, 1)
    fig.add_hline(y=0, row=3, col=1, line_color="rgba(255,255,255,0.28)", line_width=1)

    fig.update_layout(
        height=560,
        margin=dict(l=42, r=24, t=48, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.72)",
        showlegend=True,
        legend=dict(orientation="h", y=1.05, x=0, font=dict(color="#d7e3f4")),
        font=dict(color="#d7e3f4"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#93a4b8"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#93a4b8"))
    for annotation in fig.layout.annotations:
        annotation.font.color = "#d7e3f4"
        annotation.font.size = 12
    return fig


def find_analog_days(mkt: pd.DataFrame, latest: pd.Series, max_rows=6):
    cols = ["mkt_vix_level", "mkt_vix_slope_30d_3m", "mkt_vix_curvature", "mkt_vrp_21d"]
    hist = mkt[cols].dropna().iloc[:-1]
    if hist.empty:
        return pd.DataFrame()

    scaled = (hist - hist.mean()) / (hist.std() + 1e-9)
    latest_scaled = (latest[cols] - hist.mean()) / (hist.std() + 1e-9)
    distance = np.sqrt(((scaled - latest_scaled) ** 2).sum(axis=1))
    analog = hist.assign(distance=distance).sort_values("distance").head(max_rows)
    return analog


def table_rows(df: pd.DataFrame):
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{idx.strftime('%Y-%m-%d')}</td>"
            f"<td>{fmt(row['mkt_vix_level'])}</td>"
            f"<td>{fmt(row['mkt_vix_slope_30d_3m'], 2)}</td>"
            f"<td>{fmt(row['mkt_vix_curvature'], 2)}</td>"
            f"<td>{fmt(row['mkt_vrp_21d'], 1, '%')}</td>"
            f"<td>{fmt(row['distance'], 2)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def forward_volatility(short_iv: float, short_days: float, long_iv: float, long_days: float) -> float:
    variance = (long_days * long_iv**2 - short_days * short_iv**2) / (long_days - short_days)
    return float(np.sqrt(variance)) if variance > 0 else np.nan


def forward_curve(term_values: np.ndarray) -> np.ndarray:
    vix_9d, vix_30d, vix_3m, vix_6m = term_values
    return np.array(
        [
            vix_9d,
            forward_volatility(vix_9d, 9, vix_30d, 30),
            forward_volatility(vix_30d, 30, vix_3m, 91),
            forward_volatility(vix_3m, 91, vix_6m, 182),
        ]
    )


def forward_curve_figure(term_values: np.ndarray):
    labels = ["0-9D", "9-30D", "30D-3M", "3M-6M"]
    forwards = forward_curve(term_values)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=term_values,
            mode="lines+markers",
            name="Constant-maturity IV",
            line=dict(color="#7aa2f7", width=3),
            marker=dict(size=8),
            hovertemplate="%{x}<br>IV: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=forwards,
            mode="lines+markers",
            name="Forward volatility",
            line=dict(color="#bb9af7", width=3, dash="dash"),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Forward vol: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=38, r=16, t=42, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Volatility %", gridcolor="rgba(255,255,255,.08)", tickfont=dict(color="#8f9bad")),
        xaxis=dict(gridcolor="rgba(255,255,255,.03)", tickfont=dict(color="#8f9bad")),
        legend=dict(orientation="h", y=1.18, x=0, font=dict(color="#8f9bad", size=11)),
        font=dict(color="#e8edf5"),
        hovermode="x unified",
    )
    return fig


def rolling_rank(series: pd.Series, window: int = 252) -> pd.Series:
    return series.rolling(window, min_periods=63).rank(pct=True)


def status_for_score(score: float, low: str, middle: str, high: str) -> tuple[str, str]:
    if pd.isna(score):
        return "N/A", ""
    if score >= 0.80:
        return high, "bad"
    if score >= 0.60:
        return middle, "elev"
    return low, "good"


def build_state_context(mkt: pd.DataFrame, vix_term_df: pd.DataFrame) -> dict:
    aligned = mkt.join(vix_term_df[TENOR_COLS], how="inner")
    front_spread = aligned["VIX_9D"] - aligned["VIX_30D"]
    scores = pd.DataFrame(index=aligned.index)
    scores["level"] = rolling_rank(aligned["mkt_vix_level"])
    scores["momentum"] = rolling_rank(aligned["mkt_vix_ret_5d"])
    scores["front"] = rolling_rank(front_spread)
    scores["tail"] = rolling_rank(aligned["mkt_skew_level"])
    scores["vrp"] = rolling_rank(aligned["mkt_vrp_21d"])
    scores["composite"] = (
        scores["level"] * 0.30
        + scores["momentum"] * 0.20
        + scores["front"] * 0.20
        + scores["tail"] * 0.15
        + scores["vrp"] * 0.15
    )

    latest = aligned.iloc[-1]
    previous = aligned.iloc[-6]
    latest_scores = scores.iloc[-1]
    previous_scores = scores.iloc[-6]
    composite = float(latest_scores["composite"] * 100)
    prior_composite = float(previous_scores["composite"] * 100)
    delta = composite - prior_composite

    if delta >= 5:
        direction = "Deteriorating ↘"
    elif delta <= -5:
        direction = "Improving ↗"
    else:
        direction = "Stable →"

    regime, _ = regime_label(latest["mkt_vix_level"])
    surface = classify_surface(latest, latest[TENOR_COLS].values.astype(float))
    if latest["VIX_9D"] > latest["VIX_30D"] or surface["shape"] == "Backwardation":
        qualifier = "FRAGILE"
    elif latest_scores["tail"] >= 0.80:
        qualifier = "TAIL-RICH"
    elif latest_scores["vrp"] >= 0.80:
        qualifier = "PREMIUM-RICH"
    elif surface["shape"] == "Contango" and latest["mkt_vix_level"] < 20:
        qualifier = "BALANCED"
    else:
        qualifier = "TRANSITION"

    momentum_status = "RISING" if latest["mkt_vix_ret_5d"] > 0.10 else "FALLING" if latest["mkt_vix_ret_5d"] < -0.10 else "STABLE"
    momentum_class = "bad" if momentum_status == "RISING" else "good" if momentum_status == "FALLING" else "elev"
    front_status = "STRESSED" if front_spread.iloc[-1] > 0 else "WATCH" if front_spread.iloc[-1] > -1 else "RELAXED"
    front_class = "bad" if front_status == "STRESSED" else "elev" if front_status == "WATCH" else "good"
    level_status, level_class = status_for_score(latest_scores["level"], "NORMAL", "ELEVATED", "HIGH")
    tail_status, tail_class = status_for_score(latest_scores["tail"], "NORMAL", "ELEVATED", "EXPENSIVE")
    vrp_status, vrp_class = status_for_score(latest_scores["vrp"], "NORMAL", "RICH", "VERY RICH")

    rows = [
        ("Level", latest_scores["level"], f"{latest_scores['level']:.0%}", level_status, level_class),
        ("Momentum", latest_scores["momentum"], f"{latest['mkt_vix_ret_5d']:+.1%}", momentum_status, momentum_class),
        ("Front end", latest_scores["front"], f"{front_spread.iloc[-1]:+.2f}", front_status, front_class),
        ("Tail risk", latest_scores["tail"], f"{latest_scores['tail']:.0%}", tail_status, tail_class),
        ("IV-RV", latest_scores["vrp"], f"{latest['mkt_vrp_21d']:+.1f}", vrp_status, vrp_class),
    ]

    description = (
        f"VIX is {regime.lower()} at {latest['mkt_vix_level']:.2f}. The curve is {surface['shape'].lower()}, "
        f"the 9D-to-30D spread is {front_spread.iloc[-1]:+.2f} points, and SKEW ranks at "
        f"{latest_scores['tail']:.0%} of its trailing-year range. The composite stress score is {composite:.0f}/100."
    )
    interpretation = " ".join([surface["shape_text"], surface["front_text"], surface["belly_text"]])
    transition_note = (
        f"Front-end spread {front_spread.iloc[-1] - front_spread.iloc[-6]:+.2f} pts • "
        f"SKEW {latest['mkt_skew_level'] - previous['mkt_skew_level']:+.1f} • "
        f"IV-RV {latest['mkt_vrp_21d'] - previous['mkt_vrp_21d']:+.1f}"
    )
    return {
        "scores": latest_scores,
        "rows": rows,
        "regime": regime.upper(),
        "qualifier": qualifier,
        "description": description,
        "interpretation": interpretation,
        "direction": direction,
        "composite": composite,
        "previous_composite": prior_composite,
        "delta": delta,
        "previous_vix": float(previous["mkt_vix_level"]),
        "transition_note": transition_note,
    }


def find_analog_outcomes(
    mkt: pd.DataFrame,
    latest: pd.Series,
    spy: pd.DataFrame,
    max_rows: int = 20,
    min_separation: int = 21,
) -> pd.DataFrame:
    cols = ["mkt_vix_level", "mkt_vix_slope_30d_3m", "mkt_vix_curvature", "mkt_vrp_21d", "mkt_skew_level"]
    close = spy.iloc[:, 0].rename("spy_close").dropna()
    joined = mkt[cols].join(close, how="inner").dropna()
    if len(joined) < 315:
        return pd.DataFrame()

    candidates = joined.iloc[:-63].copy()
    means = candidates[cols].mean()
    stds = candidates[cols].std().replace(0, np.nan)
    target = latest[cols]
    candidates["distance"] = np.sqrt((((candidates[cols] - means) / stds - (target - means) / stds) ** 2).sum(axis=1))

    selected = []
    positions = {date: pos for pos, date in enumerate(joined.index)}
    for date, row in candidates.sort_values("distance").iterrows():
        pos = positions[date]
        if any(abs(pos - chosen) < min_separation for chosen in selected):
            continue
        selected.append(pos)
        if len(selected) >= max_rows:
            break

    records = []
    for pos in selected:
        date = joined.index[pos]
        row = joined.iloc[pos]
        record = row.to_dict()
        record["date"] = date
        record["distance"] = float(candidates.at[date, "distance"])
        record["curve"] = "Backwardation" if row["mkt_vix_slope_30d_3m"] < 0 else "Compressed" if row["mkt_vix_slope_30d_3m"] < 1 else "Contango"
        for horizon in (5, 21, 63):
            start_close = joined.iloc[pos]["spy_close"]
            path = joined.iloc[pos : pos + horizon + 1]["spy_close"]
            record[f"ret_{horizon}"] = (path.iloc[-1] / start_close - 1) * 100
            record[f"dd_{horizon}"] = float((path / path.cummax() - 1).min() * 100)
        records.append(record)
    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


def outcome_configs(analogs: pd.DataFrame) -> dict:
    configs = {}
    bins = [-np.inf, -10, -5, 0, 5, 10, np.inf]
    for horizon in (5, 21, 63):
        returns = analogs[f"ret_{horizon}"].dropna()
        drawdowns = analogs[f"dd_{horizon}"].dropna()
        histogram = pd.cut(returns, bins=bins, right=False).value_counts(sort=False)
        percentages = (histogram / max(len(returns), 1) * 100).round().astype(int).tolist()
        maximum = max(percentages) if percentages else 1
        median_return = float(returns.median())
        median_drawdown = float(drawdowns.median())
        if median_return > 1:
            bias = "constructive"
        elif median_return < -1:
            bias = "defensive"
        else:
            bias = "mixed"
        risk = "material path risk" if median_drawdown <= -5 else "contained median drawdown"
        configs[f"{horizon}D"] = {
            "ret": f"{median_return:+.1f}%",
            "retValue": round(median_return, 3),
            "hit": f"{(returns > 0).mean():.0%}",
            "dd": f"{median_drawdown:.1f}%",
            "hist": percentages,
            "histWidths": [round(value / maximum * 100) if maximum else 0 for value in percentages],
            "text": f"Historical {horizon}-session bias is {bias}, with {risk} across {len(returns)} independent analogs.",
        }
    return configs


def analog_table_rows(analogs: pd.DataFrame, rows: int = 6) -> str:
    html_rows = []
    for date, row in analogs.head(rows).iterrows():
        data = " ".join(
            f'data-ret{horizon}d="{row[f"ret_{horizon}"]:.4f}" data-dd{horizon}d="{row[f"dd_{horizon}"]:.4f}"'
            for horizon in (5, 21, 63)
        )
        ret = row["ret_21"]
        html_rows.append(
            f'<tr class="analog-row" {data}>'
            f"<td>{date:%Y-%m-%d}</td><td>{row['distance']:.2f}</td><td>{row['mkt_vix_level']:.1f}</td>"
            f"<td>{row['curve']}</td><td>{row['mkt_skew_level']:.0f}</td>"
            f'<td class="analog-return {"pos" if ret >= 0 else "neg"}">{ret:+.1f}%</td>'
            f'<td class="analog-drawdown">{row["dd_21"]:.1f}%</td></tr>'
        )
    return "\n".join(html_rows)


def _build_legacy_html(context):
    surface_html = context["surface_fig"].to_html(
        include_plotlyjs=True, full_html=False, div_id="vix-surface", config=PLOTLY_CONFIG
    )
    curve_html = context["curve_fig"].to_html(
        include_plotlyjs=False, full_html=False, div_id="term-curve", config=PLOTLY_CONFIG
    )
    ts_html = context["timeseries_fig"].to_html(
        include_plotlyjs=False, full_html=False, div_id="vix-timeseries", config=PLOTLY_CONFIG
    )

    latest = context["latest"]
    surface = context["surface"]
    regime, regime_class = regime_label(latest["mkt_vix_level"])
    term_values = context["term_values"]
    changes = context["term_changes"]
    date_str = context["date_str"]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    analog_rows = table_rows(context["analogs"])

    if latest["mkt_vix_ret_5d"] > 0.15:
        momentum_view = "Fear is building quickly. Treat the surface as unstable until the front end stops steepening."
    elif latest["mkt_vix_ret_5d"] < -0.15:
        momentum_view = "Fear is fading. The key confirmation is whether the curve rebuilds contango without VRP collapsing."
    else:
        momentum_view = "VIX momentum is muted. The surface shape matters more than the latest level change."

    if latest["mkt_vrp_21d"] > 7:
        vrp_view = "VRP is rich: implied volatility is carrying a large premium over realized volatility."
    elif latest["mkt_vrp_21d"] < 0:
        vrp_view = "VRP is negative: realized volatility is outrunning implied volatility, a stress condition."
    else:
        vrp_view = "VRP is close to normal: implied volatility is not obviously mispriced versus realized volatility."

    skew_level = latest.get("mkt_skew_level", np.nan)
    skew_pctile_1y = latest.get("mkt_skew_pctile_1y", np.nan)
    if pd.isna(skew_level):
        skew_view = "SKEW is unavailable for the latest date."
    elif not pd.isna(skew_pctile_1y) and skew_pctile_1y < 0.25:
        skew_view = (
            f"SKEW is {skew_level:.1f}, but its 1-year rank is only {skew_pctile_1y:.0%}. "
            "Tail premium is not stretched relative to the past year."
        )
    elif not pd.isna(skew_pctile_1y) and skew_pctile_1y > 0.80:
        skew_view = (
            f"SKEW is {skew_level:.1f} and ranks in the {skew_pctile_1y:.0%} percentile over 1 year. "
            "Left-tail hedging demand is elevated in both absolute and relative terms."
        )
    elif skew_level > 150 and latest["mkt_vix_level"] < 20:
        skew_view = "Low-to-normal VIX with very high SKEW is a divergence: headline vol is calm, but crash protection is expensive."
    elif skew_level > 135:
        skew_view = "SKEW is elevated, so left-tail hedging demand is above normal."
    else:
        skew_view = "SKEW is not sending a strong tail-hedging warning."

    cards = [
        ("VIX", fmt(latest["mkt_vix_level"], 2), regime, regime_class),
        ("1Y VIX Rank", fmt_pct(latest["mkt_vix_pctile_1y"]), "relative level", "neutral"),
        ("30D to 3M", fmt(latest["mkt_vix_slope_30d_3m"], 2), surface["shape"], surface["shape_class"]),
        ("VRP 21D", fmt(latest["mkt_vrp_21d"], 1, "%"), "implied - realized", "warn" if latest["mkt_vrp_21d"] > 7 else "neutral"),
        ("SKEW", fmt(skew_level, 1), "tail premium", "warn" if not pd.isna(skew_level) and skew_level > 135 else "neutral"),
        ("1Y SKEW Rank", fmt_pct(skew_pctile_1y), "relative tail premium", "warn" if not pd.isna(skew_pctile_1y) and skew_pctile_1y > 0.80 else "neutral"),
        ("Surface Range", f"{context['surface_min']:.1f}-{context['surface_max']:.1f}", f"{context['lookback']} sessions", "neutral"),
    ]
    card_html = "\n".join(
        f"""
        <div class="metric {klass}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """
        for label, value, note, klass in cards
    )

    tenor_cards = "\n".join(
        f"""
        <div class="tenor">
          <span>{label}</span>
          <strong>{value:.2f}</strong>
          <em class="{ 'up' if change > 0 else 'down' if change < 0 else '' }">{change:+.2f}% 1D</em>
        </div>
        """
        for label, value, change in zip(TENOR_LABELS, term_values, changes)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VIX Surface Intelligence | {date_str}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #050a12;
  --panel: rgba(9, 18, 32, 0.82);
  --panel-2: rgba(13, 28, 48, 0.78);
  --line: rgba(148, 163, 184, 0.22);
  --text: #d7e3f4;
  --muted: #8fa3ba;
  --cyan: #2dd4bf;
  --blue: #38bdf8;
  --gold: #f7c948;
  --red: #ff477e;
  --green: #34d399;
  --violet: #a78bfa;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background:
    radial-gradient(circle at 20% 0%, rgba(56, 189, 248, 0.16), transparent 28%),
    radial-gradient(circle at 90% 12%, rgba(247, 201, 72, 0.12), transparent 26%),
    linear-gradient(135deg, #04070d 0%, #07111f 52%, #050a12 100%);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}}
.shell {{ width: min(1480px, calc(100vw - 32px)); margin: 0 auto; padding: 26px 0 42px; }}
.topbar {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin-bottom: 18px; }}
.eyebrow {{ color: var(--cyan); text-transform: uppercase; font-size: 12px; font-weight: 800; letter-spacing: .14em; }}
h1 {{ margin: 4px 0 0; font-size: clamp(34px, 5vw, 72px); line-height: .92; letter-spacing: 0; }}
.date {{ color: var(--muted); font-size: 13px; white-space: nowrap; }}
.hero {{ display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr); gap: 18px; align-items: stretch; }}
.surface-panel, .panel, .metric, .tenor, .callout {{
  background: linear-gradient(180deg, var(--panel), rgba(6, 13, 24, .86));
  border: 1px solid var(--line);
  box-shadow: 0 20px 70px rgba(0,0,0,.28);
}}
.surface-panel {{ min-height: 680px; padding: 14px; border-radius: 8px; }}
.surface-title {{ display:flex; justify-content:space-between; align-items:center; padding: 4px 6px 8px; gap: 16px; }}
.surface-title h2, .panel h2 {{ margin:0; font-size: 16px; letter-spacing: 0; }}
.surface-title span {{ color: var(--muted); font-size: 12px; }}
#vix-surface {{ height: 620px; }}
.side {{ display: grid; gap: 12px; align-content: start; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
.metric {{ border-radius: 8px; padding: 14px; min-height: 105px; }}
.metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 800; }}
.metric-value {{ font-size: 28px; font-weight: 900; margin-top: 8px; }}
.metric-note {{ color: var(--muted); font-size: 12px; margin-top: 4px; line-height: 1.35; }}
.metric.good {{ border-color: rgba(52, 211, 153, .42); }}
.metric.warn {{ border-color: rgba(247, 201, 72, .50); }}
.metric.bad {{ border-color: rgba(255, 71, 126, .52); }}
.callout {{ border-radius: 8px; padding: 16px; }}
.callout strong {{ color: var(--gold); }}
.callout p {{ margin: 8px 0 0; color: var(--muted); line-height: 1.48; font-size: 14px; }}
.tenor-strip {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; }}
.tenor {{ border-radius: 8px; padding: 12px; }}
.tenor span, .tenor em {{ display:block; color: var(--muted); font-size: 12px; font-style: normal; }}
.tenor strong {{ display:block; font-size: 24px; margin: 4px 0; }}
.tenor .up {{ color: var(--red); }}
.tenor .down {{ color: var(--green); }}
.grid {{ display: grid; grid-template-columns: minmax(0, .9fr) minmax(0, 1.1fr); gap: 18px; margin-top: 18px; }}
.panel {{ border-radius: 8px; padding: 16px; overflow: hidden; }}
.panel p {{ color: var(--muted); line-height: 1.55; font-size: 14px; }}
.analysis-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
.analysis-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: rgba(7, 17, 31, .56); }}
.analysis-card b {{ display:block; color: #fff; margin-bottom: 7px; }}
.analysis-card span {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }}
th, td {{ text-align: right; padding: 10px 8px; border-bottom: 1px solid rgba(148, 163, 184, .16); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
.footer {{ color: var(--muted); font-size: 12px; margin-top: 18px; text-align: right; }}
@media (max-width: 1050px) {{
  .hero, .grid {{ grid-template-columns: 1fr; }}
  .surface-panel {{ min-height: 580px; }}
  #vix-surface {{ height: 520px; }}
}}
@media (max-width: 720px) {{
  .shell {{ width: min(100vw - 20px, 1480px); padding-top: 16px; }}
  .topbar {{ display:block; }}
  .date {{ margin-top: 8px; }}
  .metric-grid, .tenor-strip, .analysis-grid {{ grid-template-columns: 1fr; }}
  .surface-panel {{ padding: 8px; }}
}}
</style>
</head>
<body>
<main class="shell">
  <div class="topbar">
    <div>
      <div class="eyebrow">Volatility Surface Intelligence · {context['source_label']}</div>
      <h1>VIX Surface</h1>
    </div>
    <div class="date">As of {date_str} · generated {generated_at}</div>
  </div>

  <section class="hero">
    <div class="surface-panel">
      <div class="surface-title">
        <h2>Interactive 3D Term Structure Surface</h2>
        <span>Drag to rotate · scroll to zoom · current curve highlighted</span>
      </div>
      {surface_html}
    </div>
    <aside class="side">
      <div class="metric-grid">{card_html}</div>
      <div class="callout">
        <strong>{surface["shape"]}: {fmt(surface["slope_30_3m"], 2)} point 30D to 3M slope</strong>
        <p>{surface["shape_text"]}</p>
        <p>{surface["front_text"]}</p>
        <p>{surface["belly_text"]}</p>
      </div>
      <div class="tenor-strip">{tenor_cards}</div>
    </aside>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Current Curve</h2>
      {curve_html}
    </div>
    <div class="panel">
      <h2>VIX Context</h2>
      {ts_html}
    </div>
  </section>

  <section class="panel" style="margin-top:18px;">
    <h2>Deep VIX Read</h2>
    <div class="analysis-grid">
      <div class="analysis-card"><b>Momentum</b><span>{momentum_view}</span></div>
      <div class="analysis-card"><b>VRP</b><span>{vrp_view}</span></div>
      <div class="analysis-card"><b>SKEW</b><span>{skew_view}</span></div>
    </div>
    <p>
      The surface should get the most weight. VIX level tells how expensive 30D volatility is,
      but the surface shows where stress is concentrated. Front-end inversion points to immediate
      event risk, weak contango suggests fragile normalization, and a clean upward curve supports
      a calmer risk backdrop. VRP and SKEW are secondary checks: VRP asks whether implied volatility
      is expensive versus realized movement; SKEW asks whether left-tail protection is unusually
      expensive even when headline VIX is not.
    </p>
  </section>

  <section class="panel" style="margin-top:18px;">
    <h2>Nearest Historical Surface Analogs</h2>
    <table>
      <thead>
        <tr>
          <th>Date</th><th>VIX</th><th>30D-3M</th><th>Curvature</th><th>VRP</th><th>Distance</th>
        </tr>
      </thead>
      <tbody>{analog_rows}</tbody>
    </table>
  </section>

  <div class="footer">{context['source_note']} Single-file HTML with embedded Plotly.</div>
</main>
</body>
</html>"""


def build_html(context: dict) -> str:
    latest = context["latest"]
    mkt = context["mkt"]
    vix_term_df = context["vix_term_df"]
    term_values = context["term_values"]
    forwards = context["forward_values"]
    surface = context["surface"]
    state = context["state"]
    analogs = context["analogs"]
    configs = outcome_configs(analogs)
    default = configs["21D"]

    forward_html = context["forward_fig"].to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id="forward-curve",
        config=PLOTLY_CONFIG,
    )
    surface_html = context["surface_fig"].to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id="vix-surface",
        config=PLOTLY_CONFIG,
    )

    state_rows = []
    for label, score, value, status, klass in state["rows"]:
        width = 0 if pd.isna(score) else int(np.clip(round(score * 100), 0, 100))
        state_rows.append(
            f'<div class="row"><span>{label}</span><div class="bar"><div class="fill" style="width:{width}%"></div></div>'
            f'<span class="tag">{value}</span><span class="status {klass}">{status}</span></div>'
        )

    histogram_labels = ["< -10%", "-10 to -5%", "-5 to 0%", "0 to 5%", "5 to 10%", "> 10%"]
    histogram_rows = "\n".join(
        f'<div class="histrow"><span>{label}</span><div class="histbar"><span style="width:{width}%"></span></div><b>{value}%</b></div>'
        for label, width, value in zip(histogram_labels, default["histWidths"], default["hist"])
    )

    vix_5d_change = latest["mkt_vix_level"] - mkt["mkt_vix_level"].iloc[-6]
    vix9d_series = vix_term_df["VIX_9D"]
    vix9d_5d_change = vix9d_series.iloc[-1] - vix9d_series.iloc[-6]
    slope_5d_change = latest["mkt_vix_slope_30d_3m"] - mkt["mkt_vix_slope_30d_3m"].iloc[-6]
    front_class = "stress" if term_values[0] > term_values[1] else ""
    forward_note = (
        "near-term event premium is front-loaded"
        if term_values[0] > forwards[1] + 1
        else "risk builds beyond the first nine days"
        if forwards[1] > term_values[0] + 1
        else "near-term risk is evenly distributed"
    )
    forward_readthrough = (
        f"The 0-9D segment is {forwards[0]:.1f}% and the 9-30D forward segment is {forwards[1]:.1f}%. "
        f"The 30D-3M and 3M-6M forward segments are {forwards[2]:.1f}% and {forwards[3]:.1f}%, respectively. "
        f"This is consistent with {surface['shape'].lower()} and {forward_note}."
    )
    ridge = "front-loaded" if term_values[0] > term_values[-1] + 0.5 else "back-loaded" if term_values[-1] > term_values[0] + 0.5 else "balanced"
    current_score = int(np.clip(round(state["composite"]), 2, 98))
    previous_score = int(np.clip(round(state["previous_composite"]), 2, 98))
    vix_rank = state["scores"]["level"]
    vix9d_rank = rolling_rank(vix9d_series).iloc[-1]
    vrp_rank = state["scores"]["vrp"]
    skew_rank = state["scores"]["tail"]
    vrp_status = status_for_score(vrp_rank, "normal", "rich", "very rich")[0].lower()
    skew_status = status_for_score(skew_rank, "normal", "elevated", "tail expensive")[0].lower()

    replacements = {
        "DATE": context["date_str"],
        "SOURCE_LABEL": context["source_label"],
        "SOURCE_NOTE": context["source_note"],
        "GENERATED_AT": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "REGIME": state["regime"],
        "STATE_QUALIFIER": state["qualifier"],
        "STATE_DESCRIPTION": state["description"],
        "STATE_DIRECTION": state["direction"],
        "STATE_DELTA": f"{state['delta']:+.0f} points / 5D",
        "VIX": f"{latest['mkt_vix_level']:.2f}",
        "VIX_RANK": f"{vix_rank:.0%}",
        "VIX_CHANGE": f"{vix_5d_change:+.2f}",
        "VIX_CHANGE_CLASS": "stress" if vix_5d_change > 0 else "ok",
        "VIX9D": f"{term_values[0]:.2f}",
        "VIX9D_RANK": f"{vix9d_rank:.0%}",
        "VIX9D_CHANGE": f"{vix9d_5d_change:+.2f}",
        "FRONT_CLASS": front_class,
        "SLOPE": f"{latest['mkt_vix_slope_30d_3m']:+.2f}",
        "SURFACE_SHAPE": surface["shape"].lower(),
        "SLOPE_CHANGE": f"{slope_5d_change:+.2f}",
        "FORWARD_9_30": fmt(forwards[1], 1),
        "FORWARD_NOTE": forward_note,
        "VRP": f"{latest['mkt_vrp_21d']:+.1f}",
        "VRP_RANK": f"{vrp_rank:.0%}",
        "VRP_STATUS": vrp_status,
        "SKEW": f"{latest['mkt_skew_level']:.1f}",
        "SKEW_RANK": f"{skew_rank:.0%}",
        "SKEW_STATUS": skew_status,
        "STATE_VECTOR_ROWS": "\n".join(state_rows),
        "STATE_INTERPRETATION": state["interpretation"],
        "FORWARD_FIGURE": forward_html,
        "FORWARD_READTHROUGH": forward_readthrough,
        "PREVIOUS_SCORE": previous_score,
        "CURRENT_SCORE": current_score,
        "PREVIOUS_VIX": f"{state['previous_vix']:.2f}",
        "TRANSITION_NOTE": state["transition_note"],
        "LOOKBACK": context["lookback"],
        "RIDGE": ridge,
        "SURFACE_PEAK": f"{max(term_values):.1f}",
        "VIX6M": f"{term_values[-1]:.1f}",
        "SURFACE_FIGURE": surface_html,
        "ANALOG_ROWS": analog_table_rows(analogs),
        "MEDIAN_RETURN": default["ret"],
        "HIT_RATE": default["hit"],
        "MEDIAN_DRAWDOWN": default["dd"],
        "ANALOG_COUNT": len(analogs),
        "HISTOGRAM_ROWS": histogram_rows,
        "OUTCOME_INTERPRETATION": default["text"],
        "OUTCOME_CONFIGS": json.dumps(configs, separators=(",", ":")),
    }
    html = (SCRIPT_DIR / "report_template.html").read_text(encoding="utf-8")
    for key, value in replacements.items():
        html = html.replace(f"@@{key}@@", str(value))
    remaining = sorted(set(re.findall(r"@@([A-Z0-9_]+)@@", html)))
    if remaining:
        raise ValueError(f"Unresolved report template placeholders: {remaining}")
    return html


def generate_report(
    start: str,
    end: str,
    lookback: int,
    output: str | None,
    source: str = "cboe",
    cache_dir: str = "data/cboe_vix_cache",
):
    odm = OptionsDataManager()
    if source == "cboe":
        fetch_cboe_market_data(odm, start, end, Path(cache_dir))
        source_note = "Data source: official Cboe daily index closes; SPY closes from yfinance for realized-volatility calculations."
        source_label = "Official Cboe EOD"
    elif source == "yfinance":
        odm.fetch_market_vol_data(start, end)
        source_note = "Data source: yfinance/Cboe index proxies via OptionsDataManager."
        source_label = "yfinance"
    else:
        raise ValueError(f"Unsupported data source: {source}")

    mkt = odm.build_market_features().dropna(subset=["mkt_vix_level"])
    vix_term_df = odm._raw["vix_term"].dropna(subset=TENOR_COLS)

    if len(mkt) < 252 or len(vix_term_df) < 30:
        raise ValueError("Not enough VIX history to build a useful report.")

    latest = mkt.iloc[-1]
    prev_term = vix_term_df.iloc[-2][TENOR_COLS].values.astype(float)
    term_values = vix_term_df.iloc[-1][TENOR_COLS].values.astype(float)
    term_changes = (term_values - prev_term) / prev_term * 100
    date_str = mkt.index[-1].strftime("%Y-%m-%d")

    surface_df, x_fine, y_fine, z_fine = build_surface(vix_term_df, lookback)
    surface = classify_surface(latest, term_values)
    state = build_state_context(mkt, vix_term_df)
    analogs = find_analog_outcomes(mkt, latest, odm._raw["spy"])
    if len(analogs) < 6:
        raise ValueError("Not enough independent historical analogs with forward outcomes.")
    forward_values = forward_curve(term_values)
    context = {
        "date_str": date_str,
        "lookback": len(surface_df),
        "latest": latest,
        "mkt": mkt,
        "vix_term_df": vix_term_df,
        "term_values": term_values,
        "term_changes": term_changes,
        "forward_values": forward_values,
        "surface": surface,
        "state": state,
        "surface_fig": surface_figure(surface_df, x_fine, y_fine, z_fine),
        "forward_fig": forward_curve_figure(term_values),
        "analogs": analogs,
        "surface_min": float(np.nanmin(z_fine)),
        "surface_max": float(np.nanmax(z_fine)),
        "source_note": source_note,
        "source_label": source_label,
    }

    html = build_html(context)
    output_path = Path(output or f"market_vix_{source}_report_{date_str}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)

    return str(output_path), date_str


def main():
    parser = argparse.ArgumentParser(description="Build a single-file VIX surface market report.")
    parser.add_argument(
        "date",
        nargs="?",
        type=parse_report_date,
        help="Optional report date in YYYYMMDD format. Example: python market_condition_report.py 20260609",
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date for volatility history.")
    parser.add_argument(
        "--end",
        default=None,
        help="Override the exclusive report end date. Normally derived from the optional YYYYMMDD date.",
    )
    parser.add_argument("--lookback", type=int, default=126, help="Trading sessions shown in the VIX surface.")
    parser.add_argument("--output", default=None, help="Output HTML path.")
    parser.add_argument(
        "--source",
        choices=("cboe", "yfinance"),
        default="cboe",
        help="Volatility-index source. Cboe official daily files are the default.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cboe_vix_cache",
        help="Last-known-good cache for Cboe histories and SPY API bars.",
    )
    args = parser.parse_args()

    report_date = args.date or datetime.now()
    end = args.end or exclusive_end_date(report_date)

    output_path, date_str = generate_report(
        args.start,
        end,
        args.lookback,
        args.output,
        args.source,
        args.cache_dir,
    )
    print(f"[+] Wrote {args.source} VIX surface report for {date_str}: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
