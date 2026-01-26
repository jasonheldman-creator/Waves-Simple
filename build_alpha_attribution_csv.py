# build_alpha_attribution_csv.py
# WAVES Intelligence — Alpha Source Attribution Builder (LONG FORMAT)
#
# PURPOSE:
# Build a long-format alpha attribution table:
#   one row per (wave × horizon)
#
# OUTPUT:
#   data/alpha_attribution_summary.csv
#
# INPUT:
#   data/live_snapshot.csv
#
# AUTHOR:
#   Stabilized institutional rewrite (v3)

import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"
OUTPUT_PATH = DATA_DIR / "alpha_attribution_summary.csv"

# -----------------------------
# Config
# -----------------------------
HORIZONS = {
    "30D": "return_30d",
    "60D": "return_60d",
    "365D": "return_365d",
}

REQUIRED_COLUMNS = {
    "display_name",
    "return_30d",
    "return_60d",
    "return_365d",
}

# -----------------------------
# Helpers
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def validate_inputs(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in live_snapshot.csv: {missing}"
        )


def decompose_alpha(total_alpha: float) -> dict:
    """
    TEMPORARY, TRANSPARENT attribution split.
    These weights are explicit placeholders until
    strategy-level signal attribution is wired in.
    """
    return {
        "selection_alpha": total_alpha * 0.45,
        "momentum_alpha": total_alpha * 0.25,
        "exposure_alpha": total_alpha * 0.15,
        "vix_alpha": total_alpha * 0.10,
        "volatility_alpha": total_alpha * 0.05,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    if not LIVE_SNAPSHOT_PATH.exists():
        raise FileNotFoundError("live_snapshot.csv not found")

    snapshot_df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    snapshot_df = normalize_columns(snapshot_df)

    validate_inputs(snapshot_df)

    rows = []

    for _, row in snapshot_df.iterrows():
        wave = row["display_name"]

        for horizon, col in HORIZONS.items():
            total_alpha = row[col]

            components = decompose_alpha(total_alpha)
            residual_alpha = total_alpha - sum(components.values())

            rows.append({
                "wave": wave,
                "horizon": horizon,
                "selection_alpha": components["selection_alpha"],
                "momentum_alpha": components["momentum_alpha"],
                "exposure_alpha": components["exposure_alpha"],
                "vix_alpha": components["vix_alpha"],
                "volatility_alpha": components["volatility_alpha"],
                "residual_alpha": residual_alpha,
                "total_alpha": total_alpha,
            })

    out_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Alpha attribution built successfully: "
        f"{len(out_df)} rows written to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()