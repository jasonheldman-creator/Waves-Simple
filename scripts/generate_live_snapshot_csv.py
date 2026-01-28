# ============================================================
# generate_live_snapshot_csv.py
# WAVES Intelligence™ — Canonical Live Snapshot Generator
# PURPOSE:
#   Aggregate returns, alpha, and attribution diagnostics
#   into ONE authoritative snapshot consumed by the UI.
#
# GUARANTEES:
#   • Aggregation ONLY (no attribution math)
#   • Explicit attribution propagation
#   • wave_id–based joins only
#   • Schema-stable output
#   • No truncation, no inference, no fabrication
# ============================================================

import os
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_DIR = "data"
LIVE_SNAPSHOT_PATH = os.path.join(DATA_DIR, "live_snapshot.csv")
ATTRIBUTION_PATH = os.path.join(DATA_DIR, "alpha_attribution_snapshot.csv")

# ------------------------------------------------------------
# Attribution components expected by the UI
# ------------------------------------------------------------
ATTR_COMPONENTS = [
    "momentum",
    "volatility",
    "beta",
    "allocation",
    "residual",
]

HORIZONS = [1, 30, 60, 365]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def _safe_float(val):
    try:
        if pd.isna(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan


# ------------------------------------------------------------
# Main Generator
# ------------------------------------------------------------
def generate_live_snapshot(
    output_path: str = LIVE_SNAPSHOT_PATH,
    session_state: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Canonical live snapshot generator.

    This function:
      • computes returns + total alpha
      • merges attribution fields IF PRESENT
      • writes a UI-ready snapshot CSV

    It does NOT:
      • compute attribution
      • infer missing values
      • rename attribution semantics
    """

    from waves_engine import compute_history_nav
    from waves_engine import get_all_wave_ids
    from waves_engine import get_display_name_from_wave_id
    from waves_engine import compute_data_ready_status

    # --------------------------------------------------------
    # Safety guards
    # --------------------------------------------------------
    if session_state is not None:
        if session_state.get("safe_mode_no_fetch", False):
            print("⚠️ Safe mode active — snapshot suppressed")
            return pd.DataFrame()

        if session_state.get("safe_demo_mode", False):
            print("⚠️ Demo mode active — snapshot suppressed")
            return pd.DataFrame()

    print("=" * 72)
    print("Generating Live Snapshot (Canonical, Attribution-Aware)")
    print("=" * 72)

    # --------------------------------------------------------
    # Load attribution snapshot (OPTIONAL)
    # --------------------------------------------------------
    attribution_df = None

    if os.path.exists(ATTRIBUTION_PATH):
        try:
            attribution_df = pd.read_csv(ATTRIBUTION_PATH)
            attribution_df = _normalize_columns(attribution_df)

            if "wave_id" not in attribution_df.columns:
                print("⚠️ attribution snapshot missing wave_id — ignored")
                attribution_df = None
            else:
                attribution_df = attribution_df.set_index("wave_id")

        except Exception as e:
            print(f"⚠️ Failed to load attribution snapshot: {e}")
            attribution_df = None
    else:
        print("ℹ️ No attribution snapshot found")

    # --------------------------------------------------------
    # Build snapshot rows
    # --------------------------------------------------------
    rows: List[Dict] = []

    for wave_id in get_all_wave_ids():
        wave_name = get_display_name_from_wave_id(wave_id)

        try:
            readiness = compute_data_ready_status(wave_id)

            row = {
                "wave_id": wave_id,
                "wave_name": wave_name,
                "readiness_status": readiness.get("readiness_status", "unavailable"),
                "coverage_pct": readiness.get("coverage_pct", np.nan),
                "data_regime": readiness.get("readiness_status", "unavailable"),
            }

            # ------------------------------------------------
            # Returns + total alpha
            # ------------------------------------------------
            for d in HORIZONS:
                try:
                    nav_df = compute_history_nav(
                        wave_name,
                        mode="Standard",
                        days=d,
                        include_diagnostics=False,
                        session_state=session_state,
                    )

                    if not nav_df.empty and len(nav_df) >= 2:
                        wave_ret = (
                            nav_df["wave_nav"].iloc[-1] / nav_df["wave_nav"].iloc[0] - 1
                            if "wave_nav" in nav_df else np.nan
                        )
                        bm_ret = (
                            nav_df["bm_nav"].iloc[-1] / nav_df["bm_nav"].iloc[0] - 1
                            if "bm_nav" in nav_df else np.nan
                        )
                        row[f"return_{d}d"] = wave_ret
                        row[f"benchmark_return_{d}d"] = bm_ret
                        row[f"alpha_{d}d"] = (
                            wave_ret - bm_ret
                            if not np.isnan(wave_ret) and not np.isnan(bm_ret)
                            else np.nan
                        )
                    else:
                        row[f"return_{d}d"] = np.nan
                        row[f"benchmark_return_{d}d"] = np.nan
                        row[f"alpha_{d}d"] = np.nan

                except Exception:
                    row[f"return_{d}d"] = np.nan
                    row[f"benchmark_return_{d}d"] = np.nan
                    row[f"alpha_{d}d"] = np.nan

            # ------------------------------------------------
            # Attribution propagation (NO COMPUTATION)
            # ------------------------------------------------
            for d in HORIZONS:
                for comp in ATTR_COMPONENTS:
                    row[f"alpha_{comp}_{d}d"] = np.nan

            if attribution_df is not None and wave_id in attribution_df.index:
                attr = attribution_df.loc[wave_id]

                for d in HORIZONS:
                    for comp in ATTR_COMPONENTS:
                        col = f"alpha_{comp}_{d}d"
                        if col in attr:
                            row[col] = _safe_float(attr[col])

            rows.append(row)
            print(f"  ✓ {wave_name}")

        except Exception as e:
            print(f"  ✗ Error processing {wave_name}: {e}")

    # --------------------------------------------------------
    # Finalize
    # --------------------------------------------------------
    snapshot_df = pd.DataFrame(rows)
    snapshot_df.to_csv(output_path, index=False)

    print("\n✓ Live snapshot written")
    print(f"  Path: {output_path}")
    print(f"  Waves: {len(snapshot_df)}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return snapshot_df