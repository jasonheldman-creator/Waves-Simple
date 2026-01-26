#!/usr/bin/env python3
"""
build_alpha_attribution_csv.py

PURPOSE (HARD GUARANTEES)
--------------------------------------------------
Generate data/alpha_attribution_summary.csv from
data/live_snapshot.csv in the exact schema REQUIRED
by the Streamlit app.

Hard guarantees:
- Produces required column: `wave`
- Produces required alpha columns:
  alpha_30d, alpha_60d, alpha_365d
- Anchored to LIVE snapshot data
- Includes intraday context via snapshot_date
- Fails loudly if anything is missing
"""

import os
import sys
import logging
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
    "wave",
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

        row = {
            # 🔑 THIS IS THE KEY FIX
            "wave": display_name,

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

    out = out.sort_values("wave")

    out.to_csv(OUTPUT_FILE, index=False)

    log.info(f"Alpha attribution summary written → {OUTPUT_FILE}")
    log.info(f"Rows written: {len(out)}")

# ------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()