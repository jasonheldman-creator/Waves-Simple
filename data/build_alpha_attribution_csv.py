import pandas as pd
from pathlib import Path

# ============================
# CONFIG
# ============================

LIVE_SNAPSHOT_PATH = Path("data/live_snapshot.csv")
OUTPUT_PATH = Path("data/alpha_attribution_summary.csv")

REQUIRED_COLUMNS = [
    "Wave",
    "Weight",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",
]

OUTPUT_COLUMNS = [
    "Wave",
    "Attribution_30D",
    "Attribution_60D",
    "Attribution_365D",
]

# ============================
# HELPERS
# ============================

def empty_output_df():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)

def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"live_snapshot.csv missing required columns: {missing}")

# ============================
# CORE BUILDER
# ============================

def build_alpha_attribution_summary():
    if not LIVE_SNAPSHOT_PATH.exists():
        print("[WARN] live_snapshot.csv not found — writing empty attribution file")
        return empty_output_df()

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)

    if df.empty:
        print("[WARN] live_snapshot.csv empty — writing empty attribution file")
        return empty_output_df()

    validate_columns(df)

    # Coerce numeric fields safely
    for col in ["Weight", "Alpha_30D", "Alpha_60D", "Alpha_365D"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Compute attribution
    df["Attribution_30D"] = df["Alpha_30D"] * df["Weight"]
    df["Attribution_60D"] = df["Alpha_60D"] * df["Weight"]
    df["Attribution_365D"] = df["Alpha_365D"] * df["Weight"]

    result = (
        df.groupby("Wave", as_index=False)[
            ["Attribution_30D", "Attribution_60D", "Attribution_365D"]
        ]
        .sum()
    )

    return result[OUTPUT_COLUMNS]

# ============================
# ENTRYPOINT
# ============================

def main():
    output_df = build_alpha_attribution_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Alpha attribution CSV written: {OUTPUT_PATH}")
    print(f"[INFO] Rows: {len(output_df)}")

if __name__ == "__main__":
    main()