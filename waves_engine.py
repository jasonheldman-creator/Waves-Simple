"""
waves_engine.py

WAVES Intelligence™ — Live Engine

This script:

1. Loads wave_weights.csv (Primary / Secondary baskets — full universe).
2. Auto-discovers all Waves.
3. Normalizes weights within each Wave (full basket).
4. Fetches historical prices & daily returns via yfinance.
5. Writes positions logs for each Wave:
     logs/positions/<Wave>_positions_YYYYMMDD.csv

   Columns: Ticker, Name, Sector, weight_pct

6. Writes performance logs for each Wave:
     logs/performance/<Wave>_performance_daily.csv

   Columns: date, portfolio_return, benchmark_return

These are exactly what app.py expects for:

  - Performance Curve
  - Metric strip
  - Alpha Dashboard
  - Matrices & Engine Logs tab

Run manually:

    python waves_engine.py

(or via GitHub Actions / Codespaces / Replit)
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ================================
# Paths & config
# ================================
BASE_DIR = Path(".")
WEIGHTS_FILE = BASE_DIR / "wave_weights.csv"

LOGS_PERF_DIR = BASE_DIR / "logs" / "performance"
LOGS_POS_DIR = BASE_DIR / "logs" / "positions"

LOGS_PERF_DIR.mkdir(parents=True, exist_ok=True)
LOGS_POS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DAYS = 365  # calendar days of history

BENCHMARK_MAP = {
    "S&P 500 Wave": "^GSPC",
    "Growth Wave": "QQQ",
    "Infinity Wave": "^GSPC",
    "Income Wave": "^GSPC",
    "Future Power & Energy Wave": "XLE",
    "Crypto Income Wave": "BTC-USD",
    "Quantum Computing Wave": "QQQ",
    "Clean Transit-Infrastructure Wave": "IGE",
    "Small/Mid Growth Wave": "IWM",
}


# ================================
# Helpers
# ================================
def load_weights() -> pd.DataFrame:
    """Load wave_weights.csv and return [wave, ticker, weight]."""
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"Missing weights file: {WEIGHTS_FILE}")

    df = pd.read_csv(WEIGHTS_FILE)
    if df.empty:
        raise ValueError("wave_weights.csv is empty")

    cols = {c.lower(): c for c in df.columns}

    wave_col = cols.get("wave") or cols.get("portfolio")
    if wave_col is None:
        raise ValueError("Need a 'wave' or 'portfolio' column in wave_weights.csv")

    ticker_col = cols.get("ticker") or cols.get("symbol")
    if ticker_col is None:
        raise ValueError("Need a 'ticker' or 'symbol' column in wave_weights.csv")

    weight_candidates = [
        "weight",
        "weight_pct",
        "weight_percent",
        "target_weight",
        "portfolio_weight",
    ]
    weight_col = None
    for cand in weight_candidates:
        if cand in cols:
            weight_col = cols[cand]
            break

    if weight_col is None:
        df["__weight__"] = 1.0
        weight_col = "__weight__"

    df = df.rename(
        columns={wave_col: "wave", ticker_col: "ticker", weight_col: "raw_weight"}
    )

    df["wave"] = df["wave"].astype(str)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["raw_weight"] = df["raw_weight"].astype(float)

    df["weight"] = df.groupby("wave")["raw_weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else 1.0 / len(x)
    )

    return df[["wave", "ticker", "weight"]]


def get_benchmark_for_wave(wave_name: str) -> str:
    return BENCHMARK_MAP.get(wave_name, "^GSPC")


def fetch_price_history(tickers, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch adjusted close prices for tickers between start & end."""
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=list(tickers),
        start=start.date(),
        end=(end + timedelta(days=1)).date(),
        auto_adjust=True,
        group_by="ticker",
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        cols = []
        for t in tickers:
            if (t, "Close") in data.columns:
                cols.append(data[(t, "Close")].rename(t))
        if not cols:
            return pd.DataFrame()
        prices = pd.concat(cols, axis=1)
    else:
        prices = data["Close"].to_frame()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all")
    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def build_wave_performance(
    wave_name: str, weights_df: pd.DataFrame, start: datetime, end: datetime
) -> pd.DataFrame:
    """Return DataFrame: date, portfolio_return, benchmark_return."""
    wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
    if wave_weights.empty:
        raise ValueError(f"No rows for wave '{wave_name}' in wave_weights.csv")

    tickers = sorted(wave_weights["ticker"].unique().tolist())
    bench_ticker = get_benchmark_for_wave(wave_name)

    basket_prices = fetch_price_history(tickers, start, end)
    bench_prices = fetch_price_history([bench_ticker], start, end)

    if basket_prices.empty or bench_prices.empty:
        raise ValueError(
            f"Missing price history for wave '{wave_name}' or benchmark '{bench_ticker}'"
        )

    basket_rets = compute_daily_returns(basket_prices)
    bench_rets = compute_daily_returns(bench_prices)

    common_dates = basket_rets.index.intersection(bench_rets.index)
    basket_rets = basket_rets.loc[common_dates]
    bench_rets = bench_rets.loc[common_dates]

    weight_map = {row["ticker"]: row["weight"] for _, row in wave_weights.iterrows()}
    aligned_weights = np.array([weight_map[t] for t in basket_rets.columns])

    port_ret = basket_rets.values @ aligned_weights

    perf_df = pd.DataFrame(
        {
            "date": common_dates,
            "portfolio_return": port_ret,
            "benchmark_return": bench_rets.iloc[:, 0].values,
        }
    )

    perf_df = perf_df.sort_values("date").reset_index(drop=True)
    return perf_df


def write_performance_log(wave_name: str, perf_df: pd.DataFrame) -> Path:
    """
    Write logs/performance/<Wave>_performance_daily.csv

    Also ensures any intermediate directories implied by '/' in wave_name exist.
    """
    out_path = LOGS_PERF_DIR / f"{wave_name}_performance_daily.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # handles names with '/'
    perf_df.to_csv(out_path, index=False)
    return out_path


def write_positions_log(
    wave_name: str, weights_df: pd.DataFrame, as_of_date: datetime
) -> Path:
    """
    Write logs/positions/<Wave>_positions_YYYYMMDD.csv

    Columns: Ticker, Name, Sector, weight_pct
    """
    wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
    if wave_weights.empty:
        raise ValueError(f"No rows for wave '{wave_name}' in wave_weights.csv")

    wave_weights["weight"] = wave_weights["weight"] / wave_weights["weight"].sum()
    wave_weights["weight_pct"] = wave_weights["weight"] * 100.0

    tickers = sorted(wave_weights["ticker"].unique().tolist())

    names = {}
    sectors = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names[t] = info.get("shortName") or info.get("longName") or ""
            sectors[t] = info.get("sector") or ""
        except Exception:
            names[t] = ""
            sectors[t] = ""

    rows = []
    for _, row in wave_weights.iterrows():
        t = row["ticker"]
        rows.append(
            {
                "Ticker": t,
                "Name": names.get(t, ""),
                "Sector": sectors.get(t, ""),
                "weight_pct": row["weight_pct"],
            }
        )

    pos_df = pd.DataFrame(rows).sort_values("weight_pct", ascending=False)

    date_str = as_of_date.strftime("%Y%m%d")
    out_path = LOGS_POS_DIR / f"{wave_name}_positions_{date_str}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # handles names with '/'
    pos_df.to_csv(out_path, index=False)
    return out_path


# ================================
# Main engine routine
# ================================
def run_engine():
    print("=== WAVES Intelligence™ Engine ===")
    print(f"Base dir: {BASE_DIR.resolve()}")
    print(f"Weights file: {WEIGHTS_FILE}")

    weights_df = load_weights()
    waves = sorted(weights_df["wave"].unique().tolist())
    print(f"Discovered {len(waves)} Waves: {waves}")

    end = datetime.utcnow()
    start = end - timedelta(days=HISTORY_DAYS)

    for wave_name in waves:
        print(f"\n--- Processing Wave: {wave_name} ---")
        try:
            perf_df = build_wave_performance(wave_name, weights_df, start, end)
            perf_path = write_performance_log(wave_name, perf_df)
            print(f"  Wrote performance log: {perf_path}")

            last_date = perf_df["date"].max()
            pos_path = write_positions_log(wave_name, weights_df, last_date)
            print(f"  Wrote positions log:   {pos_path}")

        except Exception as e:
            print(f"  ERROR processing {wave_name}: {e}")


if __name__ == "__main__":
    run_engine()