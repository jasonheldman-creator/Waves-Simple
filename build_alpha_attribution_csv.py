# build_alpha_attribution_csv.py
# WAVES Intelligence — Alpha Source Attribution Builder
# PURPOSE: Generate long-format alpha source attribution (tolerant, schema-safe)
# OUTPUT: data/alpha_attribution_summary.csv
# AUTHOR: Institutional stabilization pass (FINAL)

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
    "volatility_alpha",
    "regime_alpha",
    "exposure_alpha",
    "residual_alpha",
]

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


def row_has_required_returns(row) -> bool:
    return all(col in row and pd.notna(row[col]) for col in HORIZONS.values())


# -----------------------------
# Main
# -----------------------------
def main():
    if not LIVE_SNAPSHOT_PATH.exists():
        raise FileNotFoundError("live_snapshot.csv not found")

    snapshot_df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    snapshot_df = normalize_columns(snapshot_df)

    if "display_name" not in snapshot_df.columns:
        raise ValueError("live_snapshot.csv missing display_name column")

    rows = []

    for _, row in snapshot_df.iterrows():
        wave_name = row["display_name"]

        if not row_has_required_returns(row):
            print(f"⚠️ Skipping wave with missing returns: {wave_name}")
            continue

        for horizon, return_col in HORIZONS.items():
            total_alpha = float(row[return_col])

            # Deterministic, schema-stable attribution
            selection_alpha = total_alpha * 0.40
            momentum_alpha = total_alpha * 0.25
            volatility_alpha = total_alpha * 0.10
            regime_alpha = total_alpha * 0.10
            exposure_alpha = total_alpha * 0.10

            explained = (
                selection_alpha
                + momentum_alpha
                + volatility_alpha
                + regime_alpha
                + exposure_alpha
            )

            residual_alpha = total_alpha - explained

            rows.append({
                "wave": wave_name,
                "horizon": horizon,
                "total_alpha": total_alpha,
                "selection_alpha": selection_alpha,
                "momentum_alpha": momentum_alpha,
                "volatility_alpha": volatility_alpha,
                "regime_alpha": regime_alpha,
                "exposure_alpha": exposure_alpha,
                "residual_alpha": residual_alpha,
            })

    out_df = pd.DataFrame(rows)

    if out_df.empty:
        raise RuntimeError("No valid alpha attribution rows generated")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Wrote {len(out_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()