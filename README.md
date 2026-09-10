# VIX Market Condition Report

An interactive, static VIX term-structure report generated from official Cboe
end-of-day index histories. SPY adjusted closes from yfinance are used only for
realized-volatility calculations.

The report combines a current volatility-state vector, forward-variance curve,
five-session state transition, interactive 3D term surface, and independent
historical analog outcomes at 5, 21, and 63 trading sessions.

The published page is rebuilt after each U.S. trading day by GitHub Actions.
The generated HTML is deployed as a GitHub Pages artifact and is not committed
to Git history on each refresh.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python market_condition_report.py \
  --source cboe \
  --output site/index.html \
  --cache-dir data/cboe_vix_cache
```

The builder uses the latest completed session shared by the Cboe VIX term
indices, Cboe SKEW, and SPY. It falls back to its last-known-good local cache
when a fresh source request fails.

## Deployment

`.github/workflows/deploy-pages.yml` supports:

- automatic weekday refreshes at 23:30 UTC;
- manual runs from the GitHub Actions page; and
- redeployment when the default branch changes.

This report is for personal research and is not investment advice.
