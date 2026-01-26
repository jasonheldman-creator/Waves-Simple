#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

AUTHORITATIVE LIVE SNAPSHOT GENERATOR

This script:
- Reads CURRENT prices from data/cache/prices_cache.parquet
- Uses EXISTING researched weights from data/wave_weights.csv
- Computes live returns per Wave
- Writes data/live_snapshot.csv (consumed by Streamlit app)

IMPORTANT:
- Does NOT modify weights
- Does NOT rely on canonical_snapshot.csv
- Fails loudly if inputs are invalid
"""

from pathlib import Path
from sys import exit
import logging
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PRICES_CACHE_PATH = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS_PATH = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
REQUIRED_WEIGHT_COLUMNS = {"wave", "ticker", "weight"}

RETURN_PERIODS = {
    "Return_1D": 1,
    "Return_30D": 21,
    "Return_60D": 42,
    "Return_365D": 252,
}

OUTPUT_COLUMNS = [
    "Wave_ID",
    "Wave_Name",
    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_prices_cache(path: Path) -> pd.DataFrame:
    logger.info("Loading prices cache...")
    if not path.exists():
        raise FileNotFoundError(f"Price cache not found: {path}")

    prices = pd.read_parquet(path)

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices_cache.parquet must have a DatetimeIndex")

    if prices.empty:
        raise ValueError("prices_cache.parquet is empty")

    logger.info("Prices cache loaded (%d rows, %d tickers)", *prices.shape)
    return prices


def load_wave_weights(path: Path) -> pd.DataFrame:
    logger.info("Loading wave_weights.csv...")
    if not path.exists():
        raise FileNotFoundError(f"wave_weights.csv not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_WEIGHT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing required columns: {missing}")

    if df.empty:
        raise ValueError("wave_weights.csv is empty")

    logger.info(
        "Wave weights loaded (%d rows, %d waves)",
        len(df),
        df["wave"].nunique(),
    )
    return df


def compute_return(series: pd.Series, lookback: int) -> float:
    if len(series) <= lookback:
        return np.nan
    return (series.iloc[-1] / series.iloc[-(lookback + 1)]) - 1.0


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def process_wave(
    wave_name: str,
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict | None:

    tickers = weights_df["ticker"].tolist()
    weights = weights_df["weight"].astype(float).values

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.warning("Skipping wave %s (missing tickers: %s)", wave_name, missing)
        return None

    price_slice = prices[tickers].dropna()
    if price_slice.empty:
        logger.warning("No usable prices for wave %s", wave_name)
        return None

    weight_sum = weights.sum()
    if weight_sum <= 0:
        logger.warning("Invalid weights for wave %s", wave_name)
        return None

    weights = weights / weight_sum
    weighted_prices = (price_slice * weights).sum(axis=1)

    row = {
        "Wave_ID": wave_name.lower().replace(" ", "_").replace("&", "and"),
        "Wave_Name": wave_name,
    }

    for col, days in RETURN_PERIODS.items():
        row[col] = compute_return(weighted_prices, days)

    return row


def generate_live_snapshot() -> None:
    prices = load_prices_cache(PRICES_CACHE_PATH)
    weights = load_wave_weights(WAVE_WEIGHTS_PATH)

    rows: list[dict] = []

    for wave_name, group in weights.groupby("wave"):
        snapshot = process_wave(
            wave_name=wave_name,
            weights_df=group,
            prices=prices,
        )
        if snapshot:
            rows.append(snapshot)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.critical(
            "NO VALID WAVES COMPUTED — writing headers only to live_snapshot.csv"
        )
        df.to_csv(OUTPUT_PATH, index=False)
        return

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(
        "Live snapshot written: %s (%d waves)",
        OUTPUT_PATH,
        len(df),
    )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main():
    try:
        generate_live_snapshot()
    except Exception:
        logger.exception("Live snapshot generation FAILED")
        exit(1)


if __name__ == "__main__":
    main()