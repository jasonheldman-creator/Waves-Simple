#!/usr/bin/env python3
"""
build_alpha_attribution_csv.py

Authoritative Alpha Attribution CSV builder.

Consumes:
- data/live_snapshot.csv

Produces:
- data/alpha_attribution_summary.csv

Design goals:
- Never crash
- Never silently skip valid data
- Always emit a schema-valid CSV
- Populate attribution whenever live snapshot has alpha columns
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
LIVE_SNAPSHOT = Path("data/live_snapshot.csv")
OUTPUT_PATH = Path("data/alpha_attribution_summary.csv")

# -----------------------------
# Locked Output Schema
# -----------------------------
OUTPUT_COLUMNS = [
    "Wave_ID",
    "Wave",
    "Horizon",
    "Return",
    "Alpha",
    "Alpha_Share"
]

# -----------------------------
# Helpers
# -----------------------------
def empty_output_df() -> pd.DataFrame:
    """Return empty but schema-valid output."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def load_live_snapshot() -> pd.DataFrame:
    if not LIVE_SNAPSHOT.exists():
        print("[WARN] live_snapshot.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(LIVE_SNAPSHOT)
    if df.empty:
        print("[WARN] live_snapshot.csv is empty")
    return df


def build_alpha_rows(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []

    HORIZONS = [
        ("1D", "Return_1D", "Alpha_1D"),
        ("30D", "Return_30D", "Alpha_30D"),
        ("60D", "Return_60D", "Alpha_60D"),
        ("365D", "Return_365D", "Alpha_365D"),
    ]

    for _, row in snapshot.iterrows():
        wave_id = row.get("Wave_ID")
        wave_name = row.get("Wave")

        for horizon, ret_col, alpha_col in HORIZONS:
            ret = row.get(ret_col)
            alpha = row.get(alpha_col)

            if pd.isna(ret) or pd.isna(alpha):
                continue

            rows.append({
                "Wave_ID": wave_id,
                "Wave": wave_name,
                "Horizon": horizon,
                "Return": float(ret),
                "Alpha": float(alpha),
                "Alpha_Share": float(alpha)  # direct attribution (can be refined later)
            })

    return pd.DataFrame(rows)


# -----------------------------
# Main Builder
# -----------------------------
def build_alpha_attribution_summary() -> pd.DataFrame:
    print("[INFO] Building alpha attribution summary")

    snapshot = load_live_snapshot()
    if snapshot.empty:
        print("[WARN] No live snapshot data — writing empty attribution CSV")
        return empty_output_df()

    df = build_alpha_rows(snapshot)

    if df.empty:
        print("[WARN] No valid alpha rows computed — writing empty attribution CSV")
        return empty_output_df()

    df = df[OUTPUT_COLUMNS]
    print(f"[INFO] Built {len(df)} alpha attribution rows")
    return df


def main():
    df = build_alpha_attribution_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Wrote alpha attribution CSV → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()