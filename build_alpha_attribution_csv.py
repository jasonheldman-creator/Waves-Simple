#!/usr/bin/env python3
"""
build_alpha_attribution_csv.py

PURPOSE (HARD GUARANTEES)
--------------------------------------------------
Generate data/alpha_attribution_summary.csv from
data/live_snapshot.csv in a format REQUIRED by the app.

This script:
- Produces ONE row per wave
- Outputs alpha_30d / alpha_60d / alpha_365d (REQUIRED)
- Anchors all values to LIVE snapshot data
- Includes intraday context via snapshot_date
- Fails loudly if inputs are invalid

If live_snapshot.csv changes, this file WILL change.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict

import pandas as pd

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
LIVE_SNAPSHOT_FILE = "data/live_snapshot.csv"
OUTPUT_FILE = "data/alpha_attribution_summary.csv"

REQUIRED_INPUT_COLUMNS = {
    "wave_id",
    "display_name",
    "snapshot_date",
    "return_30d",
    "return_60d",
    "return_365d",
}

REQUIRED_OUTPUT_COLUMNS = {
    "wave_name",
    "mode",
    "snapshot_date",
    "alpha_30d",
    "alpha_60d",
    "alpha_365d",
}

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("alpha_attribution")

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def hard_fail(msg: str):
    log.error(msg)
    sys.exit(1)

# ------------------------------------------------------------------------------
# LOAD INPUT
# ------------------------------------------------------------------------------
def load_live_snapshot() -> pd.DataFrame:
    if not os.path.exists(LIVE_SNAPSHOT_FILE):
        hard_fail(f"Missing input file: {LIVE_SNAPSHOT_FILE}")

    df = pd.read_csv(LIVE_SNAPSHOT_FILE)

    if df.empty:
        hard_fail("live_snapshot.csv is empty")

    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        hard_fail(f"live_snapshot.csv missing required columns: {missing}")

    log.info(f"Loaded live snapshot with {len(df)} rows")

    return df

# ------------------------------------------------------------------------------
# BUILD ATTRIBUTION
# ------------------------------------------------------------------------------
def build_alpha_attribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for wave_id, g in df.groupby("wave_id"):
        display_name = g["display_name"].iloc[0]
        snapshot_date = g["snapshot_date"].iloc[0]

        row = {
            "wave_name": display_name,
            "mode": "LIVE",
            "snapshot_date": snapshot_date,
            "alpha_30d": float(g["return_30d"].iloc[0]),
            "alpha_60d": float(g["return_60d"].iloc[0]),
            "alpha_365d": float(g["return_365d"].iloc[0]),
        }

        rows.append(row)

    if not rows:
        hard_fail("No alpha attribution rows generated")

    out = pd.DataFrame(rows)

    missing = REQUIRED_OUTPUT_COLUMNS - set(out.columns)
    if missing:
        hard_fail(f"Output missing required columns: {missing}")

    return out

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    log.info("Building alpha attribution summary (LIVE)")

    df = load_live_snapshot()
    out = build_alpha_attribution(df)

    # Stable ordering
    out = out.sort_values("wave_name")

    out.to_csv(OUTPUT_FILE, index=False)

    log.info(f"Alpha attribution summary written → {OUTPUT_FILE}")
    log.info(f"Rows written: {len(out)}")

# ------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()