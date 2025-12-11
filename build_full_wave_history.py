"""
build_full_wave_history.py

Builds Full_Wave_History.csv for all WAVES Intelligence™ Waves using
the internal WAVE_WEIGHTS from waves_engine.py and historical prices
from yfinance.

Output schema (per *position* per *date*):
    Date, Wave, Position, Weight, Ticker, Price, MarketValue

waves_engine.py then groups by (Wave, Date) and converts MarketValue
into NAV per Wave.

Run once from the repo root:
    python build_full_wave_history.py
"""

import datetime as dt
import os

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit(
        "yfinance is required. Add `yfinance` to requirements.txt and redeploy, "
        "then run this script again."
    )

from waves_engine import WAVE_WEIGHTS  # uses the same internal weights as the console


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

# How far back to build history (trading days)
YEARS_BACK = 5
TRADING_DAYS_PER_YEAR = 252
LOOKBACK_DAYS = YEARS_BACK * TRADING_DAYS_PER_YEAR

OUTPUT_FILE = "Full_Wave_History.csv"


def _fetch_prices_for_universe(tickers, start_date, end_date):
    """
    Fetch adjusted close prices for the full universe of tickers using yfinance.
    Returns a DataFrame indexed by date, columns = tickers.
    """
    tickers = sorted(set(t.strip().upper() for t in tickers if t.strip()))
    if not tickers:
        return pd.DataFrame()

    print(f"[build_full_wave_history] Fetching prices for {len(tickers)} tickers...")
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date + dt.timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if data.empty:
        raise RuntimeError("Price download failed; no data returned from yfinance.")

    # Normalize multi-index vs single-index shapes
    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if (t, "Adj Close") in data.columns:
                frames.append(data[(t, "Adj Close")].rename(t))
        if not frames:
            raise RuntimeError("No Adj Close columns found in downloaded data.")
        prices = pd.concat(frames, axis=1)
    else:
        # Single ticker case
        if "Adj Close" not in data.columns:
            raise RuntimeError("Adj Close column missing for single-ticker download.")
        prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    print(f"[build_full_wave_history] Prices shape: {prices.shape}")
    return prices


def build_full_wave_history():
    # 1) Determine overall ticker universe from WAVE_WEIGHTS
    all_tickers = set()
    for weights in WAVE_WEIGHTS.values():
        for t in weights.keys():
            all_tickers.add(t.strip().upper())

    end_date = dt.datetime.utcnow().date()
    start_date = end_date - dt.timedelta(days=LOOKBACK_DAYS + 5)

    prices = _fetch_prices_for_universe(all_tickers, start_date, end_date)

    # 2) Build per-position, per-date rows
    rows = []

    for wave_name, weights in WAVE_WEIGHTS.items():
        print(f"[build_full_wave_history] Building history for Wave: {wave_name}")
        # Normalize weights
        total_w = float(sum(float(w) for w in weights.values()))
        if total_w <= 0:
            continue

        for pos_idx, (ticker, w) in enumerate(weights.items(), start=1):
            ticker = ticker.strip().upper()
            weight = float(w) / total_w

            if ticker not in prices.columns:
                print(
                    f"[build_full_wave_history] WARNING: no prices for {ticker}; "
                    f"skipping this position in {wave_name}"
                )
                continue

            price_series = prices[ticker].dropna()
            if price_series.empty:
                print(
                    f"[build_full_wave_history] WARNING: empty price series for {ticker}; "
                    f"skipping this position in {wave_name}"
                )
                continue

            # For now we set "position size" = 1.0 unit of Wave capital.
            # MarketValue = weight * price; absolute scale doesn't matter,
            # engine normalizes per-Wave NAV to 1.0 at first date.
            for date, price in price_series.items():
                rows.append(
                    {
                        "Date": date.date(),
                        "Wave": wave_name,
                        "Position": pos_idx,
                        "Weight": weight,
                        "Ticker": ticker,
                        "Price": float(price),
                        "MarketValue": weight * float(price),
                    }
                )

    if not rows:
        raise RuntimeError("No rows generated; check WAVE_WEIGHTS and price data.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["Wave", "Date", "Position"]).reset_index(drop=True)

    print(f"[build_full_wave_history] Writing {len(df)} rows to {OUTPUT_FILE} ...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("[build_full_wave_history] Done.")


if __name__ == "__main__":
    build_full_wave_history()