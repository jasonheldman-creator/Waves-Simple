# build_alpha_attribution_csv.py
# WAVES Intelligence — Alpha Source Attribution Builder
# PURPOSE: Generate long-format alpha source breakdown for Waves + Portfolio
# OUTPUT: data/alpha_attribution_summary.csv
# AUTHOR: Stabilized institutional rewrite (v3)

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
    30: "return_30d",
    60: "return_60d",
    365: "return_365d",
}

ALPHA_SOURCES = [
    "selection_alpha",
    "momentum_alpha",
    "vix_regime_alpha",
    "volatility_alpha",
    "exposure_alpha",
    "residual_alpha",
]

# Heuristic allocation weights (can be replaced later with real signals)
ALLOCATION_WEIGHTS = {
    "selection_alpha": 0.40,
    "momentum_alpha": 0.25,
    "vix_regime_alpha": 0.15,
    "volatility_alpha": 0.10,
    "exposure_alpha": 0.10,
}

# -----------------------------
# Helpers
# -----------------------------
def validate_snapshot(df: pd.DataFrame):
    required = {"display_name"} | set(HORIZONS.values())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in live_snapshot.csv: {missing}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def decompose_alpha(total_alpha: float) -> dict:
    """Split total alpha into sources + residual."""
    components = {}
    allocated = 0.0

    for k, w in ALLOCATION_WEIGHTS.items():
        v = total_alpha * w
        components[k] = v
        allocated += v

    components["residual_alpha"] = total_alpha - allocated
    return components


# -----------------------------
# Main
# -----------------------------
def main():
    if not LIVE_SNAPSHOT_PATH.exists():
        raise FileNotFoundError("live_snapshot.csv not found")

    snapshot_df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    snapshot_df = normalize_columns(snapshot_df)
    validate_snapshot(snapshot_df)

    rows = []

    # -------- Wave-level rows --------
    for _, row in snapshot_df.iterrows():
        wave_name = row["display_name"]

        for horizon, col in HORIZONS.items():
            total_alpha = float(row[col])

            components = decompose_alpha(total_alpha)

            out = {
                "wave": wave_name,
                "horizon": horizon,
                "total_alpha": total_alpha,
            }
            out.update(components)
            rows.append(out)

    # -------- Portfolio rows --------
    for horizon, col in HORIZONS.items():
        total_alpha = snapshot_df[col].mean()

        components = decompose_alpha(total_alpha)

        out = {
            "wave": "Portfolio",
            "horizon": horizon,
            "total_alpha": total_alpha,
        }
        out.update(components)
        rows.append(out)

    # -------- Write output --------
    output_df = pd.DataFrame(rows)

    # Column order (explicit)
    ordered_cols = (
        ["wave", "horizon"]
        + ALPHA_SOURCES
        + ["total_alpha"]
    )

    output_df = output_df[ordered_cols]
    output_df.sort_values(["wave", "horizon"], inplace=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"[OK] Wrote {len(output_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()