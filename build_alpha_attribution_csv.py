# build_alpha_attribution_csv.py
# WAVES Intelligence — Alpha Attribution Builder
# PURPOSE: Generate horizon-based alpha attribution summary
# OUTPUT: data/alpha_attribution_summary.csv
# AUTHOR: Institutional-stable rewrite (schema-tolerant)

import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"
OUTPUT_PATH = DATA_DIR / "alpha_attribution_summary.csv"

# -------------------------------------------------
# Config
# -------------------------------------------------
HORIZONS = {
    30: "alpha_30d",
    60: "alpha_60d",
    365: "alpha_365d",
}

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def resolve_wave_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve wave identifier column into canonical 'wave'.
    Accepts: wave, wave_name, display_name, wave_id
    """
    if "wave" in df.columns:
        return df

    for alt in ["wave_name", "display_name", "wave_id"]:
        if alt in df.columns:
            df["wave"] = df[alt]
            return df

    raise ValueError(
        "Missing wave identifier column. "
        "Expected one of: wave, wave_name, display_name, wave_id"
    )


def validate_inputs(df: pd.DataFrame) -> None:
    required_alpha_cols = set(HORIZONS.values())
    missing = required_alpha_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in live_snapshot.csv: {missing}"
        )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main() -> None:
    if not LIVE_SNAPSHOT_PATH.exists():
        raise FileNotFoundError("live_snapshot.csv not found")

    snapshot_df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    snapshot_df = normalize_columns(snapshot_df)
    snapshot_df = resolve_wave_column(snapshot_df)

    validate_inputs(snapshot_df)

    rows = []

    for days, alpha_col in HORIZONS.items():
        grouped = (
            snapshot_df
            .groupby("wave", dropna=False)[alpha_col]
            .sum()
            .reset_index()
        )

        for _, r in grouped.iterrows():
            rows.append({
                "wave": r["wave"],
                "horizon": days,
                alpha_col: r[alpha_col],
            })

    if not rows:
        raise RuntimeError("No alpha attribution rows generated")

    out_df = pd.DataFrame(rows)

    # Pivot to wide format for app compatibility
    out_df = (
        out_df
        .pivot(index="wave", columns="horizon")
        .reset_index()
    )

    # Flatten column names
    out_df.columns = [
        f"{col[0]}_{col[1]}d" if isinstance(col, tuple) and col[1] else col
        for col in out_df.columns
    ]

    # Clean final column names
    out_df.columns = (
        out_df.columns
        .str.replace("__", "_")
        .str.strip("_")
    )

    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Alpha attribution written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()