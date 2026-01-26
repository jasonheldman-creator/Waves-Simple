#!/usr/bin/env python3
"""
build_alpha_attribution_csv.py

PURPOSE (HARD GUARANTEES)
--------------------------------------------------
Generate data/alpha_attribution_summary.csv from
data/live_snapshot.csv.

This script:
- Validates live_snapshot.csv schema explicitly
- Never silently drops all rows
- Logs every skip reason per wave
- Fails loudly if output would be empty

If live_snapshot.csv has rows, this file WILL
produce rows — or explain exactly why it cannot.
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

DAYS_MAP = {
    "return_30d": 30,
    "return_60d": 60,
    "return_365d": 365,
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


def load_live_snapshot() -> pd.DataFrame:
    if not os.path.exists(LIVE_SNAPSHOT_FILE):
        hard_fail(f"Missing live snapshot file: {LIVE_SNAPSHOT_FILE}")

    df = pd.read_csv(LIVE_SNAPSHOT_FILE)

    if df.empty:
        hard_fail("live_snapshot.csv exists but has zero rows")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        hard_fail(f"live_snapshot.csv missing required columns: {missing}")

    log.info(f"Loaded live snapshot: {len(df)} rows")

    return df


# ------------------------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------------------------
def main():
    df = load_live_snapshot()

    rows: List[Dict] = []

    for _, row in df.iterrows():
        wave_id = row["wave_id"]
        display_name = row["display_name"]

        for return_col, days in DAYS_MAP.items():
            value = row.get(return_col)

            if pd.isna(value):
                log.warning(
                    f"{wave_id}: {return_col} is NaN — skipping attribution for {days}D"
                )
                continue

            rows.append({
                "wave_name": display_name,
                "mode": "LIVE",
                "days": days,
                "total_alpha": float(value),
                "total_wave_return": float(value),
            })

    if not rows:
        hard_fail(
            "Alpha attribution produced ZERO rows.\n"
            "This means live_snapshot.csv did not contain any usable return values.\n"
            "Check return_30d / return_60d / return_365d columns."
        )

    out_df = pd.DataFrame(rows)

    # Stable ordering
    out_df = out_df.sort_values(["wave_name", "days"])

    out_df.to_csv(OUTPUT_FILE, index=False)

    log.info(f"Alpha attribution written → {OUTPUT_FILE}")
    log.info(f"Rows written: {len(out_df)}")


# ------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()