# build_alpha_attribution_csv.py
# WAVES Intelligence — Alpha Source Attribution Builder
# PURPOSE: Generate long-format alpha source breakdown by Wave and Horizon
# OUTPUT: data/alpha_attribution_summary.csv
# AUTHOR: Stabilized institutional rewrite

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
    30: "alpha_30d",
    60: "alpha_60d",
    365: "alpha_365d",
}

# Proportional attribution model (stable first-pass)
ATTRIBUTION_WEIGHTS = {
    "selection_alpha": 0.50,
    "momentum_alpha": 0.20,
    "vix_alpha": 0.15,
    "volatility_alpha": 0.10,
    "exposure_alpha": 0.05,
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
    required = {"wave"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in live_snapshot.csv: {missing}")


# -----------------------------
# Main Build Logic
# -----------------------------
def main():
    if not LIVE_SNAPSHOT_PATH.exists():
        raise FileNotFoundError("live_snapshot.csv not found")

    snapshot_df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    snapshot_df = normalize_columns(snapshot_df)
    validate_inputs(snapshot_df)

    rows = []

    for _, row in snapshot_df.iterrows():
        wave = row["wave"]

        for horizon, alpha_col in HORIZONS.items():
            if alpha_col not in snapshot_df.columns:
                continue

            total_alpha = row.get(alpha_col, 0.0)

            # Defensive numeric handling
            try:
                total_alpha = float(total_alpha)
            except Exception:
                total_alpha = 0.0

            # Allocate alpha sources
            allocated = {}
            allocated_sum = 0.0

            for source, weight in ATTRIBUTION_WEIGHTS.items():
                value = total_alpha * weight
                allocated[source] = value
                allocated_sum += value

            residual_alpha = total_alpha - allocated_sum

            rows.append({
                "wave": wave,
                "horizon": horizon,
                "total_alpha": total_alpha,
                "selection_alpha": allocated["selection_alpha"],
                "momentum_alpha": allocated["momentum_alpha"],
                "vix_alpha": allocated["vix_alpha"],
                "volatility_alpha": allocated["volatility_alpha"],
                "exposure_alpha": allocated["exposure_alpha"],
                "residual_alpha": residual_alpha,
            })

    # Always write a non-empty, schema-valid CSV
    output_df = pd.DataFrame(rows)

    if output_df.empty:
        output_df = pd.DataFrame(columns=[
            "wave",
            "horizon",
            "total_alpha",
            "selection_alpha",
            "momentum_alpha",
            "vix_alpha",
            "volatility_alpha",
            "exposure_alpha",
            "residual_alpha",
        ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Alpha attribution summary written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()