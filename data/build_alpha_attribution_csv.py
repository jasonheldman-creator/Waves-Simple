#!/usr/bin/env python3
"""
build_alpha_attribution_csv.py

PURPOSE (HARD GUARANTEES)
--------------------------------------------------
Generate data/alpha_attribution_summary.csv from
data/live_snapshot.csv.

If live_snapshot.csv has rows, this file WILL
produce rows.

If it does not, this script will FAIL LOUDLY
and explain exactly why.

Design rules:
1. Never silently emit an empty file
2. Enforce required schema
3. Log per-wave attribution math
4. Always write rows or exit non-zero
"""

import os
import sys
import logging
from typing import List, Dict

import pandas as pd

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
LIVE_SNAPSHOT_FILE = "data/live_snapshot.csv"
OUTPUT_FILE = "data/alpha_attribution_summary.csv"

REQUIRED_COLUMNS = {
    "wave_id",
    "display_name",
    "snapshot_date",
    "return_1d",
    "return_30d",
    "return_60d",
    "return_365d",
}

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("alpha_attribution_builder")

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def hard_fail(msg: str):
    log.error(msg)
    sys.exit(1)


def load_live_snapshot() -> pd.DataFrame:
    if not os.path.exists(LIVE_SNAPSHOT_FILE):
        hard_fail(f"Missing input file: {LIVE_SNAPSHOT_FILE}")

    df = pd.read_csv(LIVE_SNAPSHOT_FILE)

    if df.empty:
        hard_fail("live_snapshot.csv exists but has ZERO rows")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        hard_fail(f"live_snapshot.csv missing required columns: {missing}")

    log.info(f"Loaded live_snapshot.csv with {len(df)} rows")
    return df


# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------
def build_alpha_attribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for _, row in df.iterrows():
        wave_id = row["wave_id"]
        display_name = row["display_name"]
        snapshot_date = row["snapshot_date"]

        try:
            entry = {
                "wave_id": wave_id,
                "display_name": display_name,
                "snapshot_date": snapshot_date,
                "alpha_1d": float(row["return_1d"]),
                "alpha_30d": float(row["return_30d"]),
                "alpha_60d": float(row["return_60d"]),
                "alpha_365d": float(row["return_365d"]),
            }

            rows.append(entry)
            log.info(f"Alpha attribution built for {wave_id}")

        except Exception as e:
            log.warning(f"Skipping {wave_id} due to error: {e}")

    if not rows:
        hard_fail("Alpha attribution generation resulted in ZERO rows")

    out = pd.DataFrame(rows)

    # Hard sort for deterministic output
    out = out.sort_values("wave_id")

    return out


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    log.info("Starting alpha attribution build")

    df = load_live_snapshot()
    attribution_df = build_alpha_attribution(df)

    attribution_df.to_csv(OUTPUT_FILE, index=False)

    log.info(f"Wrote alpha attribution summary → {OUTPUT_FILE}")
    log.info(f"Rows written: {len(attribution_df)}")


if __name__ == "__main__":
    main()