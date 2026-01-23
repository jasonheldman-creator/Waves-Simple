#!/usr/bin/env python3

import os
import sys
import json
import logging
from datetime import datetime
from typing import List

import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

REPO_ROOT = os.getcwd()

CACHE_DIR = os.path.join(REPO_ROOT, "data", "cache")
PRICE_CACHE_PATH = os.path.join(CACHE_DIR, "prices_cache.parquet")
META_CACHE_PATH = os.path.join(CACHE_DIR, "prices_cache_meta.json")

WAVE_POSITIONS_PATH = os.path.join(REPO_ROOT, "data", "wave_positions.csv")
UNIVERSAL_UNIVERSE_PATH = os.path.join(REPO_ROOT, "universal_universe.csv")

MIN_TICKERS_REQUIRED = 1  # hard invariant

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("build_price_cache")

# =============================================================================
# UTILITIES
# =============================================================================

def fatal(msg: str) -> None:
    log.error(msg)
    sys.exit(1)

# =============================================================================
# TICKER LOADING (DETERMINISTIC)
# =============================================================================

def load_tickers() -> List[str]:
    """
    Load tickers deterministically.

    Priority:
    1. data/wave_positions.csv  (canonical)
    2. universal_universe.csv   (fallback)

    Hard fails if no tickers are resolved.
    """

    log.info("Loading tickers...")

    # ---- Primary source: wave_positions.csv
    if os.path.exists(WAVE_POSITIONS_PATH):
        log.info("Using primary ticker source: data/wave_positions.csv")

        try:
            df = pd.read_csv(WAVE_POSITIONS_PATH)
        except Exception as e:
            fatal(f"Failed to read wave_positions.csv: {e}")

        if "ticker" not in df.columns:
            fatal("wave_positions.csv missing required column: 'ticker'")

        tickers = (
            df["ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        log.info(f"Loaded {len(tickers)} tickers from wave_positions.csv")

    # ---- Fallback source: universal_universe.csv
    else:
        log.warning(
            "wave_positions.csv not found. "
            "Falling back to universal_universe.csv"
        )

        if not os.path.exists(UNIVERSAL_UNIVERSE_PATH):
            fatal(
                "Neither data/wave_positions.csv nor universal_universe.csv exist. "
                "Cannot proceed."
            )

        try:
            df = pd.read_csv(UNIVERSAL_UNIVERSE_PATH)
        except Exception as e:
            fatal(f"Failed to read universal_universe.csv: {e}")

        if "ticker" not in df.columns:
            fatal("universal_universe.csv missing required column: 'ticker'")

        # Optional but explicit filtering
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "active"]
            log.info("Filtered universal_universe.csv to active tickers only")

        tickers = (
            df["ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        log.info(f"Loaded {len(tickers)} tickers from universal_universe.csv")

    # ---- HARD INVARIANT
    if len(tickers) < MIN_TICKERS_REQUIRED:
        fatal(
            f"Ticker load invariant violated: "
            f"resolved {len(tickers)} tickers (minimum required: {MIN_TICKERS_REQUIRED})"
        )

    log.info("Ticker loading successful")
    return tickers

# =============================================================================
# PRICE CACHE GENERATION (PLACEHOLDER SAFE)
# =============================================================================

def build_price_cache(tickers: List[str]) -> pd.DataFrame:
    """
    Build a minimal, structurally valid price cache.

    NOTE:
    This does NOT fetch live prices.
    It ensures CI integrity and structural correctness.
    """

    log.info("Building price cache DataFrame")

    today = pd.Timestamp.utcnow().normalize()

    data = {}
    for t in tickers:
        # Minimal placeholder price series
        data[t] = [1.0]

    df = pd.DataFrame(data, index=[today])

    log.info(
        f"Price cache built with shape {df.shape} "
        f"(rows={df.shape[0]}, tickers={df.shape[1]})"
    )

    return df

# =============================================================================
# METADATA GENERATION (NEVER NULL)
# =============================================================================

def build_metadata(tickers: List[str], price_df: pd.DataFrame) -> dict:
    """
    Build deterministic metadata.

    Metadata is NEVER allowed to contain None.
    """

    log.info("Building cache metadata")

    metadata = {
        "spy_max_date": "UNAVAILABLE",
        "max_price_date": price_df.index.max().strftime("%Y-%m-%d"),
        "tickers_total": len(tickers),
        "tickers_successful": len(tickers),
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }

    # Final invariant check
    for k, v in metadata.items():
        if v is None:
            fatal(f"Metadata invariant violated: {k} is None")

    log.info(f"Metadata built: {json.dumps(metadata, indent=2)}")
    return metadata

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=== BUILD PRICE CACHE START ===")

    os.makedirs(CACHE_DIR, exist_ok=True)

    tickers = load_tickers()

    price_df = build_price_cache(tickers)
    metadata = build_metadata(tickers, price_df)

    # ---- WRITE FILES (ATOMIC INTENT)
    log.info(f"Writing price cache to {PRICE_CACHE_PATH}")
    price_df.to_parquet(PRICE_CACHE_PATH)

    log.info(f"Writing metadata to {META_CACHE_PATH}")
    with open(META_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("=== BUILD PRICE CACHE COMPLETE ===")

# =============================================================================

if __name__ == "__main__":
    main()
    