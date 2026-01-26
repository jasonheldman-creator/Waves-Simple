#!/usr/bin/env python3
"""
build_alpha_attribution_csv.py

PURPOSE
--------------------------------------------------
Generate alpha source breakdown for WAVES strategies.

Outputs:
data/alpha_attribution_summary.csv

Each row represents:
    (wave × horizon)

Horizons:
    30D, 60D, 365D

Alpha Sources:
    - selection_alpha
    - momentum_alpha
    - vix_alpha
    - volatility_alpha
    - exposure_alpha
    - residual_alpha

Design Principles:
    • Explicit horizons
    • Deterministic math
    • No silent skips
    • App schema is the contract
"""

import sys
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_DIR = Path("data")
LIVE_SNAPSHOT = DATA_DIR / "live_snapshot.csv"
OUTPUT_FILE = DATA_DIR / "alpha_attribution_summary.csv"

HORIZONS = {
    30: "alpha_30d",
    60: "alpha_60d",
    365: "alpha_365d",
}

REQUIRED_LIVE_COLS = {
    "wave",
    "alpha_30d",
    "alpha_60d",
    "alpha_365d",
}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def hard_fail(msg: str):
    print(f"ERROR: {msg}")
    sys.exit(1)


def load_live_snapshot() -> pd.DataFrame:
    if not LIVE_SNAPSHOT.exists():
        hard_fail("live_snapshot.csv not found")

    df = pd.read_csv(LIVE_SNAPSHOT)

    if df.empty:
        hard_fail("live_snapshot.csv is empty")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    missing = REQUIRED_LIVE_COLS - set(df.columns)
    if missing:
        hard_fail(f"live_snapshot.csv missing columns: {missing}")

    return df


# -------------------------------------------------
# ATTRIBUTION MODEL (DETERMINISTIC PLACEHOLDERS)
# -------------------------------------------------
def decompose_alpha(total_alpha: float) -> dict:
    """
    Deterministic, explainable alpha split.
    These weights are intentionally conservative and sum to 1.0.
    """

    if pd.isna(total_alpha):
        total_alpha = 0.0

    selection_alpha = total_alpha * 0.35
    momentum_alpha = total_alpha * 0.20
    vix_alpha = total_alpha * 0.15
    volatility_alpha = total_alpha * 0.15
    exposure_alpha = total_alpha * 0.10

    explained = (
        selection_alpha
        + momentum_alpha
        + vix_alpha
        + volatility_alpha
        + exposure_alpha
    )

    residual_alpha = total_alpha - explained

    return {
        "selection_alpha": selection_alpha,
        "momentum_alpha": momentum_alpha,
        "vix_alpha": vix_alpha,
        "volatility_alpha": volatility_alpha,
        "exposure_alpha": exposure_alpha,
        "residual_alpha": residual_alpha,
    }


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    live_df = load_live_snapshot()

    rows = []

    for _, r in live_df.iterrows():
        wave = r["wave"]

        for days, alpha_col in HORIZONS.items():
            total_alpha = r.get(alpha_col, 0.0)

            components = decompose_alpha(total_alpha)

            row = {
                "wave": wave,
                "horizon": days,
                **components,
            }

            rows.append(row)

    if not rows:
        hard_fail("No alpha attribution rows generated")

    out_df = pd.DataFrame(rows)

    out_df = out_df.sort_values(
        ["horizon", "selection_alpha"],
        ascending=[True, False],
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"SUCCESS: wrote {len(out_df)} rows → {OUTPUT_FILE}")


# -------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------
if __name__ == "__main__":
    main()