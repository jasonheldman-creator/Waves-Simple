"""
waves_engine.py

Multi-Wave Engine for WAVES:

- Loads universe from list.csv (optional – used for Name/Sector if available)
- Loads weights from wave_weights.csv (Wave,Ticker,Weight)
- Cleans & normalizes weights per wave
- Auto-discovers all waves
- Fetches latest prices via yfinance
- Writes positions logs and daily performance logs for each wave:
    logs/positions/<Wave>_positions_YYYYMMDD.csv
    logs/performance/<Wave>_performance_daily.csv
- Auto-organizes any "loose" log files into the correct folders
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UNIVERSE_FILE = BASE_DIR / "list.csv"
WEIGHTS_FILE = BASE_DIR / "wave_weights.csv"

LOGS_DIR = BASE_DIR / "logs"
POSITIONS_DIR = LOGS_DIR / "positions"
PERFORMANCE_DIR = LOGS_DIR / "performance"

# Simple benchmark mapping (fallback to SPY if not specified)
WAVE_BENCHMARKS = {
    "SP500_Wave": "SPY",
    "Growth_Wave": "QQQ",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def ensure_log_dirs() -> None:
    """
    Ensure logs/, logs/positions/, logs/performance/ exist.

    Also auto-move any loose log files in logs/ into the correct
    subfolders so Streamlit can find them.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)

    # Move any *_performance_daily.csv files into logs/performance/
    for f in LOGS_DIR.glob("*_performance_daily.csv"):
        target = PERFORMANCE_DIR / f.name
        if f.is_file() and not target.exists():
            f.rename(target)

    # Move any *_positions_*.csv files into logs/positions/
    for f in LOGS_DIR.glob("*_positions_*.csv"):
        target = POSITIONS_DIR / f.name
        if f.is_file() and not target.exists():
            f.rename(target)


def load_universe() -> pd.DataFrame:
    """
    Load list.csv (universe).

    Tries to find a ticker column among: 'Ticker', 'Symbol'.
    Returns empty DataFrame if file is missing or doesn't have those.
    """
    if not UNIVERSE_FILE.exists():
        print(f"[WARN] Universe file not found at {UNIVERSE_FILE}, "
              f"continuing without universe metadata.")
        return pd.DataFrame()

    df = pd.read_csv(UNIVERSE_FILE).copy()

    # Find ticker column
    ticker_col = None
    for col in df.columns:
        if col.lower() in ("ticker", "symbol"):
            ticker_col = col
            break

    if ticker_col is None:
        print(f"[WARN] No 'Ticker' or 'Symbol' column found in {UNIVERSE_FILE.name}; "
              f"continuing without metadata.")
        return pd.DataFrame()

    df["Ticker"] = df[ticker_col].astype(str).str.strip().str.upper()

    if "Name" not in df.columns:
        df["Name"] = ""
    if "Sector" not in df.columns:
        df["Sector"] = ""

    return df[["Ticker", "Name", "Sector"]]


def load_wave_weights() -> pd.DataFrame:
    """
    Load wave_weights.csv.

    Expected columns: Wave,Ticker,Weight
    - Strips whitespace
    - Uppercases tickers
    - Coerces Weight to numeric
    - Drops rows with missing/invalid Weight
    - Normalizes weights per wave so each sums to 1.0
    """
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"wave_weights.csv not found at {WEIGHTS_FILE}")

    df = pd.read_csv(WEIGHTS_FILE).copy()

    required_cols = {"Wave", "Ticker", "Weight"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"wave_weights.csv must contain columns {required_cols}, "
            f"but has {set(df.columns)}"
        )

    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Weight"])
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with non-numeric weights.")

    norm_frames = []
    for wave, group in df.groupby("Wave"):
        total = group["Weight"].sum()
        if total <= 0:
            print(f"[WARN] Wave '{wave}' has total weight <= 0; skipping.")
            continue
        g = group.copy()
        g["Weight"] = g["Weight"] / total
        norm_frames.append(g)

    if not norm_frames:
        raise ValueError("No valid waves found in wave_weights.csv after normalization.")

    norm_df = pd.concat(norm_frames, ignore_index=True)
    return norm_df


def fetch_latest_prices(tickers) -> dict:
    """
    Fetch latest closing prices for the given tickers using yfinance.

    Returns dict: {ticker: price}
    """
    if not tickers:
        return {}

    tickers = sorted(set(t for t in tickers if isinstance(t, str) and t.strip()))
    if not tickers:
        return {}

    print(f"[INFO] Fetching prices for {len(tickers)} tickers via yfinance...")

    data = yf.download(
        tickers=tickers,
        period="1d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    prices: dict[str, float] = {}

    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                series = data[(t, "Close")].dropna()
            else:
                # Single-ticker case
                series = data["Close"].dropna()

            if series.empty:
                print(f"[WARN] No price data for {t}.")
                continue

            prices[t] = float(series.iloc[-1])
        except Exception as e:
            print(f"[WARN] Could not fetch price for {t}: {e}")

    return prices


def sanitize_wave_name_for_file(wave_name: str) -> str:
    """Replace spaces with underscores for filenames."""
    return wave_name.replace(" ", "_")


def write_positions_log(wave_name: str, positions_df: pd.DataFrame) -> Path:
    """
    Write positions for this wave to logs/positions/<Wave>_positions_YYYYMMDD.csv.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    fname = f"{sanitize_wave_name_for_file(wave_name)}_positions_{date_str}.csv"
    path = POSITIONS_DIR / fname
    positions_df.to_csv(path, index=False)
    print(f"[LOG] Wrote positions log: {path}")
    return path


def append_performance_log(
    wave_name: str,
    portfolio_nav: float,
    benchmark_price: float,
    benchmark_ticker: str,
) -> Path:
    """
    Append a row to logs/performance/<Wave>_performance_daily.csv.

    Columns:
        timestamp, wave_name, benchmark, nav, benchmark_price,
        wave_return, benchmark_return, alpha
    """
    fname = f"{sanitize_wave_name_for_file(wave_name)}_performance_daily.csv"
    path = PERFORMANCE_DIR / fname

    timestamp = datetime.now().isoformat(timespec="seconds")

    if path.exists():
        hist = pd.read_csv(path)
        if not hist.empty:
            last = hist.iloc[-1]
            prev_nav = float(last.get("nav", portfolio_nav))
            prev_bench = float(last.get("benchmark_price", benchmark_price))
            wave_ret = (portfolio_nav / prev_nav - 1.0) if prev_nav > 0 else 0.0
            bench_ret = (benchmark_price / prev_bench - 1.0) if prev_bench > 0 else 0.0
        else:
            wave_ret = 0.0
            bench_ret = 0.0
    else:
        wave_ret = 0.0
        bench_ret = 0.0

    alpha = wave_ret - bench_ret

    row = {
        "timestamp": timestamp,
        "wave_name": wave_name,
        "benchmark": benchmark_ticker,
        "nav": portfolio_nav,
        "benchmark_price": benchmark_price,
        "wave_return": wave_ret,
        "benchmark_return": bench_ret,
        "alpha": alpha,
    }

    df_row = pd.DataFrame([row])
    header = not path.exists()
    df_row.to_csv(path, mode="a", header=header, index=False)
    print(f"[LOG] Appended performance log: {path}")
    return path


# ---------------------------------------------------------------------
# Core per-wave engine
# ---------------------------------------------------------------------

def run_wave_engine(
    wave_name: str,
    universe_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> None:
    """
    Run engine for a single wave:

    - Filter weights to this wave
    - Merge with universe metadata (Name, Sector) if available
    - Fetch latest prices for all tickers + benchmark
    - Compute NAV (unit portfolio with target_portfolio_value = 1.0)
    - Save positions log
    - Append performance log
    """
    print(f"\n===== WAVES {wave_name} Engine Run =====")

    wdf = weights_df[weights_df["Wave"] == wave_name].copy()
    if wdf.empty:
        print(f"[WARN] No weights found for wave '{wave_name}', skipping.")
        return

    # Attach metadata if we have a universe
    if not universe_df.empty:
        wdf = wdf.merge(universe_df, on="Ticker", how="left")
    else:
        wdf["Name"] = ""
        wdf["Sector"] = ""

    tickers = wdf["Ticker"].dropna().unique().tolist()
    benchmark_ticker = WAVE_BENCHMARKS.get(wave_name, "SPY")

    # Fetch prices for both wave tickers and benchmark
    price_map = fetch_latest_prices(tickers + [benchmark_ticker])

    # Attach prices
    wdf["Price"] = wdf["Ticker"].map(price_map)

    missing_prices = wdf[wdf["Price"].isna()]
    if not missing_prices.empty:
        print("[WARN] Missing prices for the following tickers; they will be dropped:")
        print(missing_prices["Ticker"].tolist())
        wdf = wdf.dropna(subset=["Price"])

    if wdf.empty:
        print(f"[ERROR] No valid tickers with prices for wave '{wave_name}', skipping.")
        return

    # Compute portfolio NAV (unit portfolio)
    target_portfolio_value = 1.0
    wdf["TargetWeight"] = wdf["Weight"]
    wdf["DollarWeight"] = wdf["TargetWeight"] * target_portfolio_value
    wdf["Shares"] = wdf["DollarWeight"] / wdf["Price"]

    portfolio_nav = wdf["DollarWeight"].sum()

    benchmark_price = price_map.get(benchmark_ticker, 0.0)
    if benchmark_price == 0.0:
        print(f"[WARN] Could not fetch benchmark price for {benchmark_ticker}; "
              f"setting benchmark_price = 0 for logging.")

    # Save positions log
    positions_cols = [
        "Ticker",
        "Name",
        "Sector",
        "TargetWeight",
        "Price",
        "DollarWeight",
        "Shares",
    ]
    positions_log = wdf[positions_cols].copy()
    write_positions_log(wave_name, positions_log)

    # Append performance log
    append_performance_log(
        wave_name=wave_name,
        portfolio_nav=portfolio_nav,
        benchmark_price=benchmark_price,
        benchmark_ticker=benchmark_ticker,
    )

    print(f"[OK] {wave_name} engine run completed.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ensure_log_dirs()

    universe_df = load_universe()
    weights_df = load_wave_weights()

    # Discover waves dynamically
    waves = sorted(weights_df["Wave"].dropna().unique().tolist())
    print(f"Found waves: {waves}")

    for wave_name in waves:
        try:
            run_wave_engine(wave_name, universe_df, weights_df)
        except Exception as e:
            print(f"[ERROR] Wave '{wave_name}' failed: {e}")


if __name__ == "__main__":
    main()
