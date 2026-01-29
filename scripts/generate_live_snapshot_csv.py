# scripts/generate_live_snapshot_csv.py
# WAVES Intelligence™ — Canonical Live Snapshot Generator
# PURPOSE:
# Produce a UI-safe, attribution-complete live_snapshot.csv every run

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
WEIGHTS_PATH = DATA_DIR / "wave_weights.csv"
OUTPUT_PATH = DATA_DIR / "live_snapshot.csv"

# ---- REQUIRED UI COLUMNS ----

BASE_COLUMNS = [
    "wave_name",
    "asof",
    "return_1d", "alpha_1d",
    "return_30d", "alpha_30d",
    "return_60d", "alpha_60d",
    "return_365d", "alpha_365d",
]

ATTRIBUTION_COLUMNS = [
    # Residual
    "alpha_residual_30d", "alpha_residual_60d", "alpha_residual_365d",
    # Momentum
    "alpha_momentum_30d", "alpha_momentum_60d", "alpha_momentum_365d",
    # Volatility
    "alpha_volatility_30d", "alpha_volatility_60d", "alpha_volatility_365d",
    # Beta
    "alpha_beta_30d", "alpha_beta_60d", "alpha_beta_365d",
    # Allocation
    "alpha_allocation_30d", "alpha_allocation_60d", "alpha_allocation_365d",
]

ALL_COLUMNS = BASE_COLUMNS + ATTRIBUTION_COLUMNS

# ---- LOAD WAVES ----

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError("data/wave_weights.csv not found")

weights_df = pd.read_csv(WEIGHTS_PATH)
if "wave_name" not in weights_df.columns:
    raise ValueError("wave_weights.csv must contain 'wave_name' column")

waves = sorted(weights_df["wave_name"].unique())

# ---- SNAPSHOT ROW BUILDER ----

def build_snapshot_row(wave_name: str, asof: str) -> dict:
    """
    Build a UI-safe snapshot row.
    Real engines can be wired later without breaking schema.
    """

    row = {
        "wave_name": wave_name,
        "asof": asof,

        # Returns (safe defaults)
        "return_1d": 0.0,
        "alpha_1d": 0.0,
        "return_30d": np.nan,
        "alpha_30d": np.nan,
        "return_60d": np.nan,
        "alpha_60d": np.nan,
        "return_365d": np.nan,
        "alpha_365d": np.nan,
    }

    # Attribution drivers — MUST EXIST for UI
    for col in ATTRIBUTION_COLUMNS:
        row[col] = np.nan

    return row

# ---- BUILD SNAPSHOT ----

asof_date = datetime.utcnow().strftime("%Y-%m-%d")
rows = []

for wave in waves:
    try:
        rows.append(build_snapshot_row(wave, asof_date))
    except Exception as e:
        print(f"[WARN] Failed snapshot row for {wave}: {e}")

snapshot_df = pd.DataFrame(rows)

# ---- FINAL SAFETY CHECK ----

for col in ALL_COLUMNS:
    if col not in snapshot_df.columns:
        snapshot_df[col] = np.nan

snapshot_df = snapshot_df[ALL_COLUMNS]

# ---- WRITE OUTPUT ----

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
snapshot_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ live_snapshot.csv written with {len(snapshot_df)} rows")
print(f"✅ Columns: {len(snapshot_df.columns)}")