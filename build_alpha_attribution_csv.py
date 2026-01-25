"""
build_alpha_attribution_csv.py
WAVES Intelligence™ — Alpha Attribution CSV Builder (Summary-Only)

PURPOSE
-------
Export REAL, reconciled alpha attribution summaries for each Wave
using the canonical alpha attribution engine.

This script:
• Uses alpha_attribution.py as the sole source of truth
• Produces NO synthetic attribution
• Enforces reconciliation discipline
• Outputs a stable, institution-grade CSV schema

This file performs EXPORT ONLY.
All math lives in alpha_attribution.py.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List, Optional

from alpha_attribution import compute_alpha_attribution_series_safe


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Input data
HISTORY_DIR = Path("data/history")
DIAGNOSTICS_DIR = Path("data/diagnostics")
WAVE_REGISTRY_CSV = Path("data/wave_registry.csv")

# Output
OUTPUT_CSV = Path("data/alpha_attribution_summary.csv")

# Defaults
DEFAULT_MODE = "Standard"
DEFAULT_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_wave_registry() -> pd.DataFrame:
    if not WAVE_REGISTRY_CSV.exists():
        raise FileNotFoundError(f"Wave registry not found: {WAVE_REGISTRY_CSV}")
    return pd.read_csv(WAVE_REGISTRY_CSV)


def load_history_df(wave_id: str) -> Optional[pd.DataFrame]:
    """
    Load per-wave history.
    Returns None if history is missing (never raises).
    """
    path = HISTORY_DIR / f"{wave_id}_history.csv"
    if not path.exists():
        print(f"[WARN] Skipping wave '{wave_id}' — history file not found")
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df


def load_diagnostics_df(wave_id: str) -> Optional[pd.DataFrame]:
    """
    Load optional diagnostics data.
    Missing diagnostics are allowed.
    """
    path = DIAGNOSTICS_DIR / f"{wave_id}_diagnostics.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df


# ---------------------------------------------------------------------
# MAIN BUILD
# ---------------------------------------------------------------------

def build_alpha_attribution_summary() -> pd.DataFrame:
    registry = load_wave_registry()
    rows: List[dict] = []

    for _, row in registry.iterrows():
        wave_id = row["wave_id"]
        wave_name = row["wave_name"]
        mode = row.get("mode", DEFAULT_MODE)

        history_df = load_history_df(wave_id)
        if history_df is None:
            continue  # ← critical: skip safely

        diagnostics_df = load_diagnostics_df(wave_id)

        # Enforce lookback window
        if len(history_df) > DEFAULT_LOOKBACK_DAYS:
            history_df = history_df.tail(DEFAULT_LOOKBACK_DAYS)

        # Compute REAL attribution (engine-owned math)
        _, summary = compute_alpha_attribution_series_safe(
            wave_name=wave_name,
            mode=mode,
            history_df=history_df,
            diagnostics_df=diagnostics_df
        )

        rows.append(summary.to_dict())

    if not rows:
        raise RuntimeError("No waves produced attribution output")

    df = pd.DataFrame(rows)

    # Canonical column order (LOCKED)
    ordered_cols = [
        "wave_name",
        "mode",
        "days",
        "total_alpha",
        "total_wave_return",
        "total_benchmark_return",
        "exposure_timing_alpha",
        "regime_vix_alpha",
        "momentum_trend_alpha",
        "volatility_control_alpha",
        "asset_selection_alpha",
        "sum_of_components",
        "reconciliation_error",
        "reconciliation_pct_error",
        "exposure_timing_contribution_pct",
        "regime_vix_contribution_pct",
        "momentum_trend_contribution_pct",
        "volatility_control_contribution_pct",
        "asset_selection_contribution_pct",
    ]

    return df[ordered_cols]


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = build_alpha_attribution_summary()
    df.to_csv(OUTPUT_CSV, index=False)

    print("Alpha attribution summary CSV built successfully")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Waves processed: {len(df)}")


if __name__ == "__main__":
    main()