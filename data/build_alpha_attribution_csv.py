import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================
# Paths
# ============================

ROOT = Path(__file__).resolve().parents[1]

LIVE_SNAPSHOT = ROOT / "data" / "live_snapshot.csv"
WAVE_WEIGHTS = ROOT / "data" / "wave_weights.csv"
OUTPUT = ROOT / "data" / "alpha_attribution_summary.csv"

# ============================
# Helpers
# ============================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def resolve_columns(df: pd.DataFrame, mapping: dict) -> dict:
    resolved = {}
    for logical, options in mapping.items():
        for opt in options:
            if opt in df.columns:
                resolved[logical] = opt
                break
    missing = set(mapping.keys()) - set(resolved.keys())
    if missing:
        raise ValueError(f"Missing required logical fields: {missing}")
    return resolved


def empty_output() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "wave",
            "wave_name",
            "alpha_30d",
            "alpha_60d",
            "alpha_365d",
            "as_of_date",
        ]
    )


# ============================
# Main builder
# ============================

def build_alpha_attribution_summary():
    print("▶ Building alpha attribution summary")

    if not LIVE_SNAPSHOT.exists():
        print("[WARN] live_snapshot.csv not found")
        return empty_output()

    if not WAVE_WEIGHTS.exists():
        print("[WARN] wave_weights.csv not found")
        return empty_output()

    # ----------------------------
    # Load inputs
    # ----------------------------

    snap = normalize_columns(pd.read_csv(LIVE_SNAPSHOT))
    weights = normalize_columns(pd.read_csv(WAVE_WEIGHTS))

    if snap.empty:
        print("[WARN] live_snapshot.csv is empty")
        return empty_output()

    if weights.empty:
        print("[WARN] wave_weights.csv is empty")
        return empty_output()

    # ----------------------------
    # Resolve schemas
    # ----------------------------

    SNAP_MAP = {
        "wave": ["wave", "wave_id"],
        "wave_name": ["wave_name", "display_name"],
        "alpha_30d": ["alpha_30d", "alpha30d"],
        "alpha_60d": ["alpha_60d", "alpha60d"],
        "alpha_365d": ["alpha_365d", "alpha365d"],
    }

    WEIGHT_MAP = {
        "wave": ["wave", "wave_id"],
        "weight": ["weight", "allocation", "pct", "percent"],
    }

    snap_cols = resolve_columns(snap, SNAP_MAP)
    weight_cols = resolve_columns(weights, WEIGHT_MAP)

    snap = snap.rename(columns={v: k for k, v in snap_cols.items()})
    weights = weights.rename(columns={v: k for k, v in weight_cols.items()})

    # ----------------------------
    # Merge & compute attribution
    # ----------------------------

    merged = snap.merge(
        weights[["wave", "weight"]],
        on="wave",
        how="left",
    )

    if merged["weight"].isna().all():
        print("[WARN] All weights missing after merge")
        return empty_output()

    for col in ["alpha_30d", "alpha_60d", "alpha_365d"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col] * merged["weight"]

    result = (
        merged.groupby(["wave", "wave_name"], as_index=False)
        .agg(
            alpha_30d=("alpha_30d", "sum"),
            alpha_60d=("alpha_60d", "sum"),
            alpha_365d=("alpha_365d", "sum"),
        )
    )

    result["as_of_date"] = datetime.utcnow().date().isoformat()

    return result


# ============================
# Entrypoint
# ============================

if __name__ == "__main__":
    try:
        df = build_alpha_attribution_summary()
        df.to_csv(OUTPUT, index=False)
        print(f"✅ Wrote alpha attribution CSV → {OUTPUT}")
    except Exception as e:
        print(f"❌ Alpha attribution build failed: {e}")
        raise