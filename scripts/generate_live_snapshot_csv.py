"""
WAVES Intelligence™
Generate Live Snapshot CSV

PURPOSE
-------
Generate the canonical live_snapshot.csv used by the Streamlit app.
This script:
- Calls compute_history_nav() using its EXISTING signature
- Extracts the latest NAV row
- Persists diagnostics (VIX, momentum, exposure, etc.)
- Writes a flat CSV snapshot

CRITICAL GUARANTEES
-------------------
- DOES NOT modify strategy logic
- DOES NOT pass unsupported keyword arguments
- FAILS loudly if diagnostics are missing
"""

import sys
import traceback
from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "live_snapshot.csv"

# -----------------------------
# Imports (deliberately minimal)
# -----------------------------
try:
    from waves_engine import compute_history_nav
except Exception as e:
    print("❌ Failed to import compute_history_nav")
    raise

# -----------------------------
# Helpers
# -----------------------------
def normalize_history_result(result):
    """
    Normalize output of compute_history_nav into:
    - history_df (pd.DataFrame)
    - diagnostics (dict)

    Supports:
    1) dict with keys
    2) tuple/list
    """
    diagnostics = None
    history_df = None

    if isinstance(result, dict):
        history_df = result.get("history") or result.get("df")
        diagnostics = result.get("diagnostics")

    elif isinstance(result, (tuple, list)):
        if len(result) >= 2:
            history_df = result[0]
            diagnostics = result[-1]

    if history_df is None or diagnostics is None:
        raise ValueError(
            "compute_history_nav did not return history + diagnostics. "
            "Refusing to generate snapshot."
        )

    return history_df, diagnostics


def latest_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("History DataFrame is empty")
    return df.iloc[-1]


# -----------------------------
# Main
# -----------------------------
def main():
    print("▶ Generating live snapshot...")

    try:
        # 🔒 Call with NO keyword arguments
        result = compute_history_nav()

        history_df, diagnostics = normalize_history_result(result)
        last = latest_row(history_df)

        # -----------------------------
        # Base snapshot fields
        # -----------------------------
        snapshot = {
            "date": last.name if hasattr(last, "name") else None,
            "nav": last.get("nav") or last.get("NAV"),
            "wave_return_1d": last.get("return_1d"),
            "wave_return_30d": last.get("return_30d"),
            "wave_return_60d": last.get("return_60d"),
            "wave_return_365d": last.get("return_365d"),
            "bm_return_1d": last.get("bm_return_1d"),
            "bm_return_30d": last.get("bm_return_30d"),
            "bm_return_60d": last.get("bm_return_60d"),
            "bm_return_365d": last.get("bm_return_365d"),
        }

        # -----------------------------
        # Diagnostics (raw, not re-engineered)
        # -----------------------------
        snapshot.update({
            "vix": diagnostics.get("vix"),
            "regime": diagnostics.get("regime"),
            "tilt_factor": diagnostics.get("tilt_factor"),
            "vix_exposure": diagnostics.get("vix_exposure"),
            "vol_adjust": diagnostics.get("vol_adjust"),
            "safe_fraction": diagnostics.get("safe_fraction"),
            "exposure": diagnostics.get("exposure"),
            "aggregated_risk_state": diagnostics.get("aggregated_risk_state"),
        })

        # -----------------------------
        # Alpha (true economic alpha)
        # -----------------------------
        if snapshot["wave_return_365d"] is not None and snapshot["bm_return_365d"] is not None:
            snapshot["alpha_365d"] = (
                snapshot["wave_return_365d"] - snapshot["bm_return_365d"]
            )
        else:
            snapshot["alpha_365d"] = None

        # -----------------------------
        # Write CSV
        # -----------------------------
        DATA_DIR.mkdir(exist_ok=True)
        pd.DataFrame([snapshot]).to_csv(OUTPUT_PATH, index=False)

        print(f"✅ live_snapshot.csv written to {OUTPUT_PATH}")

    except Exception as e:
        print("❌ Failed to generate live snapshot")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()