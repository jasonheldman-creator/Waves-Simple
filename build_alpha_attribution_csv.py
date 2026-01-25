"""
build_alpha_attribution_csv.py
WAVES Intelligence™ — Alpha Attribution CSV Builder (Summary-Only)

PURPOSE
-------
Export REAL, reconciled alpha attribution summaries for each Wave
using the canonical alpha attribution engine.

GUARANTEES
----------
• Uses alpha_attribution.py as the sole source of truth
• Produces NO synthetic attribution
• Skips waves with missing inputs (expected during development)
• NEVER fails CI due to missing data
• ALWAYS writes data/alpha_attribution_summary.csv
• ALWAYS writes a valid CSV with a locked schema
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List

from alpha_attribution import compute_alpha_attribution_series_safe


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

HISTORY_DIR = Path("data/history")
DIAGNOSTICS_DIR = Path("data/diagnostics")
WAVE_REGISTRY_CSV = Path("data/wave_registry.csv")

OUTPUT_CSV = Path("data/alpha_attribution_summary.csv")

DEFAULT_MODE = "Standard"
DEFAULT_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------
# LOCKED OUTPUT SCHEMA
# ---------------------------------------------------------------------

ORDERED_COLS = [
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


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def empty_output_df() -> pd.DataFrame:
    """Return empty DataFrame with locked schema."""
    return pd.DataFrame({col: [] for col in ORDERED_COLS})


def load_wave_registry() -> pd.DataFrame:
    if not WAVE_REGISTRY_CSV.exists():
        print(f"[WARN] Wave registry missing: {WAVE_REGISTRY_CSV}")
        return pd.DataFrame()

    return pd.read_csv(WAVE_REGISTRY_CSV)


def load_history_df(wave_id: str) -> pd.DataFrame | None:
    path = HISTORY_DIR / f"{wave_id}_history.csv"
    if not path.exists():
        print(f"[WARN] Skipping wave '{wave_id}' — history missing")
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    if "date" in df.columns:
        df.set_index("date", inplace=True)

    if len(df) > DEFAULT_LOOKBACK_DAYS:
        df = df.tail(DEFAULT_LOOKBACK_DAYS)

    return df


def load_diagnostics_df(wave_id: str) -> pd.DataFrame | None:
    path = DIAGNOSTICS_DIR / f"{wave_id}_diagnostics.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    if "date" in df.columns:
        df.set_index("date", inplace=True)

    return df


# ---------------------------------------------------------------------
# CORE BUILD LOGIC
# ---------------------------------------------------------------------

def build_alpha_attribution_summary() -> pd.DataFrame:
    registry = load_wave_registry()
    rows: List[dict] = []

    if registry.empty:
        print("[WARN] Wave registry empty — writing empty attribution CSV")
        return empty_output_df()

    for _, r in registry.iterrows():
        wave_id = r.get("wave_id")
        wave_name = r.get("wave_name", wave_id)
        mode = r.get("mode", DEFAULT_MODE)

        if not wave_id:
            continue

        history_df = load_history_df(wave_id)
        if history_df is None or history_df.empty:
            continue

        diagnostics_df = load_diagnostics_df(wave_id)

        try:
            _, summary = compute_alpha_attribution_series_safe(
                wave_name=wave_name,
                mode=mode,
                history_df=history_df,
                diagnostics_df=diagnostics_df,
            )

            summary_dict = summary.to_dict()

            # Force schema alignment
            row = {col: summary_dict.get(col) for col in ORDERED_COLS}
            rows.append(row)

        except Exception as e:
            print(f"[WARN] Attribution failed for '{wave_id}': {e}")
            continue

    if not rows:
        print("[WARN] No waves produced attribution — writing empty CSV")
        return empty_output_df()

    return pd.DataFrame(rows, columns=ORDERED_COLS)


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = build_alpha_attribution_summary()
    df.to_csv(OUTPUT_CSV, index=False)

    print("======================================")
    print("Alpha Attribution CSV build complete")
    print(f"Path: {OUTPUT_CSV}")
    print(f"Rows written: {len(df)}")
    print("======================================")


if __name__ == "__main__":
    main()