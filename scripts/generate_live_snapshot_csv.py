#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

Purpose:
- Authoritative LIVE snapshot generator for Waves-Simple.
- Computes and exports live snapshot metrics for each wave.

Design Requirements:
1. Assume prices_cache.parquet is wide-form (index = date, columns = tickers, values = prices).
2. Explicitly validate wave_weights.csv schema; raise a ValueError if columns are missing.
3. Use column names: wave_id, display_name, Return_1D, Return_30D, Return_60D, Return_365D.
4. Normalize weights before computing weighted prices.
5. Use 1, 21, 42, and 252 as return-period trading-day equivalents.
6. Never silently write an empty file; write headers only and log a CRITICAL warning if no valid waves are computed.
7. Use clear logging for all major steps and exit non-zero on any failure.
"""

import logging
from pathlib import Path
from sys import exit

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
# Schema + constants
# ---------------------------------------------------------------------
REQUIRED_WEIGHT_COLUMNS = {"wave_id", "display_name", "ticker", "weight"}

RETURN_PERIODS = {
    "Return_1D": 1,
    "Return_30D": 21,
    "Return_60D": 42,
    "Return_365D": 252,
}

OUTPUT_COLUMNS = [
    "wave_id",
    "display_name",
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

    logger.info(
        "Prices cache loaded (%d dates, %d tickers)",
        prices.shape[0],
        prices.shape[1],
    )
    return prices


def load_and_validate_wave_weights(path: Path) -> pd.DataFrame:
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
        df["wave_id"].nunique(),
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
    wave_id: str,
    display_name: str,
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict | None:
    logger.info("Processing wave: %s", wave_id)

    tickers = weights_df["ticker"].tolist()
    weights = weights_df["weight"].astype(float).values

    # Ensure tickers exist
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.warning(
            "Skipping wave %s due to missing tickers: %s",
            wave_id,
            missing,
        )
        return None

    price_slice = prices[tickers].dropna()
    if price_slice.empty:
        logger.warning("No usable price data for wave %s", wave_id)
        return None

    # Normalize weights
    weight_sum = weights.sum()
    if weight_sum <= 0:
        logger.warning("Invalid weights for wave %s (sum <= 0)", wave_id)
        return None

    weights = weights / weight_sum

    weighted_prices = (price_slice * weights).sum(axis=1)

    snapshot = {
        "wave_id": wave_id,
        "display_name": display_name,
    }

    for col, days in RETURN_PERIODS.items():
        snapshot[col] = compute_return(weighted_prices, days)

    return snapshot


def generate_live_snapshot() -> None:
    prices = load_prices_cache(PRICES_CACHE_PATH)
    wave_weights = load_and_validate_wave_weights(WAVE_WEIGHTS_PATH)

    rows: list[dict] = []

    for wave_id, group in wave_weights.groupby("wave_id"):
        display_name = group["display_name"].iloc[0]

        snapshot = process_wave(
            wave_id=wave_id,
            display_name=display_name,
            weights_df=group,
            prices=prices,
        )

        if snapshot is not None:
            rows.append(snapshot)

    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if output_df.empty:
        logger.critical(
            "No valid waves computed — writing headers only to live_snapshot.csv"
        )
        output_df.to_csv(OUTPUT_PATH, index=False)
        return

    output_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(
        "Live snapshot written: %s (%d waves)",
        OUTPUT_PATH,
        len(output_df),
    )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main():
    try:
        generate_live_snapshot()
    except Exception as e:
        logger.exception("Live snapshot generation FAILED")
        exit(1)


if __name__ == "__main__":
    main()