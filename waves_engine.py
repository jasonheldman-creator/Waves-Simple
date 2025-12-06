"""
WAVES Intelligence™ - Multi-Wave Engine

Reads:
  - list.csv            : master universe (Ticker, Company, Sector, etc.)
  - wave_weights.csv    : Wave, Ticker, Weight

Outputs (in ./logs):
  - <Wave>_positions_YYYYMMDD.csv
  - wave_performance_daily.csv
"""

import os
import datetime as dt
import pandas as pd
import yfinance as yf

# ----------- Config -----------

UNIVERSE_FILE = "list.csv"
WAVE_WEIGHTS_FILE = "wave_weights.csv"
LOG_DIR = "logs"

# You can customize benchmarks later if you want
DEFAULT_BENCHMARK = "SPY"


# ----------- Helpers -----------

def ensure_log_dir() -> None:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def load_universe() -> pd.DataFrame:
    """
    Load the global universe from list.csv and normalize columns.
    We deduplicate tickers here so each symbol appears once.
    """
    df = pd.read_csv(UNIVERSE_FILE)

    # Try to normalize likely column names
    cols = {c.lower(): c for c in df.columns}

    ticker_col = cols.get("ticker")
    name_col = cols.get("company") or cols.get("name")
    sector_col = cols.get("sector")

    if ticker_col is None:
        raise ValueError("Universe file must contain a 'Ticker' column.")

    df_universe = pd.DataFrame()
    df_universe["Ticker"] = df[ticker_col].astype(str).str.strip()

    if name_col is not None:
        df_universe["Name"] = df[name_col].astype(str).str.strip()
    else:
        df_universe["Name"] = df_universe["Ticker"]

    if sector_col is not None:
        df_universe["Sector"] = df[sector_col].astype(str).str.strip()
    else:
        df_universe["Sector"] = "None"

    # Deduplicate tickers – keep the first occurrence
    df_universe = (
        df_universe
        .drop_duplicates(subset="Ticker", keep="first")
        .reset_index(drop=True)
    )

    return df_universe


def load_wave_weights() -> pd.DataFrame:
    """
    Load all wave weights from wave_weights.csv.
    Expected columns: Wave, Ticker, Weight
    """
    df = pd.read_csv(WAVE_WEIGHTS_FILE)

    cols = {c.lower(): c for c in df.columns}

    wave_col = cols.get("wave")
    ticker_col = cols.get("ticker")
    weight_col = cols.get("weight")

    if wave_col is None or ticker_col is None or weight_col is None:
        raise ValueError(
            "wave_weights.csv must contain 'Wave', 'Ticker', and 'Weight' columns."
        )

    df_weights = pd.DataFrame()
    df_weights["Wave"] = df[wave_col].astype(str).str.strip()
    df_weights["Ticker"] = df[ticker_col].astype(str).str.strip()
    df_weights["Weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)

    # Drop rows with zero or negative weights
    df_weights = df_weights[df_weights["Weight"] > 0]

    return df_weights


def fetch_latest_prices(tickers) -> pd.DataFrame:
    """
    Fetch latest prices & 1-day change % for a list of tickers using yfinance.
    Returns DataFrame with columns: Ticker, LivePrice, LiveChangePct
    """
    rows = []

    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if hist.empty:
                continue

            last = hist["Close"].iloc[-1]
            if len(hist) > 1:
                prev = hist["Close"].iloc[-2]
                change_pct = float((last - prev) / prev * 100.0) if prev != 0 else 0.0
            else:
                change_pct = 0.0

            rows.append(
                {
                    "Ticker": ticker,
                    "LivePrice": float(last),
                    "LiveChangePct": float(change_pct),
                }
            )
        except Exception:
            # If price fails for one ticker, just skip it
            continue

    if not rows:
        return pd.DataFrame(columns=["Ticker", "LivePrice", "LiveChangePct"])

    return pd.DataFrame(rows)


def write_positions_log(wave_name: str, df_positions: pd.DataFrame) -> None:
    """
    Write daily positions file for a single wave.
    """
    today = dt.datetime.utcnow().strftime("%Y%m%d")
    filename = f"{wave_name}_positions_{today}.csv"
    path = os.path.join(LOG_DIR, filename)

    df_positions.to_csv(path, index=False)
    print(f"Wrote positions log for {wave_name}: {path}")


def append_performance_log(wave_name: str, nav: float, benchmark_symbol: str = DEFAULT_BENCHMARK) -> None:
    """
    Append a single row into wave_performance_daily.csv.
    For now we just store NAV=1.0 and benchmark_nav=1.0 – can be
    replaced later with real performance series.
    """
    perf_file = "wave_performance_daily.csv"
    path = os.path.join(LOG_DIR, perf_file)

    row = {
        "date": dt.datetime.utcnow().date().isoformat(),
        "wave": wave_name,
        "nav": float(nav),
        "benchmark": benchmark_symbol,
        "benchmark_nav": 1.0,
        "timestamp": dt.datetime.utcnow().isoformat(),
    }

    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)
    print(f"Appended performance log for {wave_name}: {path}")


# ----------- Core Wave Logic -----------

def build_wave_portfolio(wave_name: str, universe_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean holdings DataFrame for a single wave:
      Ticker, Name, Sector, TargetWeight
    We also group by Ticker to eliminate duplicates.
    """
    w = weights_df[weights_df["Wave"] == wave_name].copy()
    if w.empty:
        raise ValueError(f"No holdings found in wave_weights.csv for wave '{wave_name}'")

    # Merge to get names & sectors
    df = w.merge(universe_df, on="Ticker", how="left")

    if df["Name"].isna().all():
        raise ValueError(f"After joining with list.csv, no valid universe rows for '{wave_name}'")

    # Group by Ticker to avoid duplicates and sum weights
    df = (
        df.groupby("Ticker", as_index=False)
        .agg(
            {
                "Wave": "first",
                "Weight": "sum",
                "Name": "first",
                "Sector": "first",
            }
        )
    )

    df = df.rename(columns={"Weight": "TargetWeight"})

    return df[["Ticker", "Name", "Sector", "TargetWeight"]]


def run_wave_engine(wave_name: str, universe_df: pd.DataFrame, weights_df: pd.DataFrame) -> None:
    """
    Run one wave: build holdings, pull prices, write logs.
    """
    print(f"\n=== WAVES {wave_name} Engine Run ===")

    # 1) Build holdings
    df = build_wave_portfolio(wave_name, universe_df, weights_df)

    # 2) Fetch latest prices
    tickers = df["Ticker"].dropna().unique().tolist()
    prices_df = fetch_latest_prices(tickers)

    if not prices_df.empty:
        df = df.merge(prices_df, on="Ticker", how="left")
    else:
        # If we couldn't get prices, still log holdings with NaNs
        df["LivePrice"] = float("nan")
        df["LiveChangePct"] = float("nan")

    # 3) Compute simple NAV = 1.0 and DollarWeight column
    portfolio_nav = 1.0
    df["DollarWeight"] = df["TargetWeight"] * portfolio_nav

    # 4) Write positions log
    write_positions_log(wave_name, df)

    # 5) Append performance log
    append_performance_log(wave_name, nav=portfolio_nav)

    print(f"{wave_name} engine run completed.")


# ----------- Main -----------

def main():
    ensure_log_dir()

    universe_df = load_universe()
    weights_df = load_wave_weights()

    # Discover all waves dynamically from the weights file
    waves = sorted(weights_df["Wave"].unique())

    print("Found waves:")
    for w in waves:
        print(f"  - {w}")

    for wave_name in waves:
        run_wave_engine(wave_name, universe_df, weights_df)


if __name__ == "__main__":
    main()
