"""
WAVES Intelligence™
Generate Live Snapshot CSV

PURPOSE
-------
Generate the canonical live_snapshot.csv used by the Streamlit app.

This script is intentionally defensive:
- Explicitly fixes PYTHONPATH
- Calls compute_history_nav() with ZERO unsupported kwargs
- Extracts diagnostics without re-engineering strategy logic
- Fails loudly if expectations are violated
"""

import sys
import traceback
from pathlib import Path
import pandas as pd

# ============================================================
# 🔒 FIX PYTHON PATH (THIS IS THE MISSING PIECE)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================
# Paths
# ============================================================
DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "live_snapshot.csv"

# ============================================================
# Import engine (now guaranteed to work)
# ============================================================
try:
    from waves_engine import compute_history_nav
except Exception:
    print("❌ Failed to import compute_history_nav from waves_engine")
    raise

# ============================================================
# Helpers
# ============================================================
def normalize_history_result(result):
    """
    Normalize compute_history_nav output into:
    - history_df
    - diagnostics dict
    """
    history_df = None
    diagnostics = None

    if isinstance(result, dict):
        history_df = result.get("history") or result.get("df")
        diagnostics = result.get("diagnostics")

    elif isinstance(result, (tuple, list)):
        if len(result) >= 2:
            history_df = result[0]
            diagnostics = result[-1]

    if history_df is None or diagnostics is None:
        raise ValueError(
            "compute_history_nav did not return (history, diagnostics). "
            "Snapshot generation aborted."
        )

    return history_df, diagnostics


def get_latest_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("History DataFrame is empty")
    return df.iloc[-1]


# ============================================================
# Main
# ============================================================
def main():
    print("▶ Generating live snapshot...")

    try:
        # 🔒 Call EXACT engine signature (no kwargs)
        result = compute_history_nav()

        history_df, diagnostics = normalize_history_result(result)
        last = get_latest_row(history_df)

        snapshot = {
            # Core
            "date": last.name if hasattr(last, "name") else None,
            "nav": last.get("nav") or last.get("NAV"),

            # Wave returns
            "wave_return_1d": last.get("return_1d"),
            "wave_return_30d": last.get("return_30d"),
            "wave_return_60d": last.get("return_60d"),
            "wave_return_365d": last.get("return_365d"),

            # Benchmark returns
            "bm_return_1d": last.get("bm_return_1d"),
            "bm_return_30d": last.get("bm_return_30d"),
            "bm_return_60d": last.get("bm_return_60d"),
            "bm_return_365d": last.get("bm_return_365d"),

            # Diagnostics (RAW, AS COMPUTED)
            "vix": diagnostics.get("vix"),
            "regime": diagnostics.get("regime"),
            "tilt_factor": diagnostics.get("tilt_factor"),
            "vix_exposure": diagnostics.get("vix_exposure"),
            "vol_adjust": diagnostics.get("vol_adjust"),
            "safe_fraction": diagnostics.get("safe_fraction"),
            "exposure": diagnostics.get("exposure"),
            "aggregated_risk_state": diagnostics.get("aggregated_risk_state"),
        }

        # True economic alpha
        if snapshot["wave_return_365d"] is not None and snapshot["bm_return_365d"] is not None:
            snapshot["alpha_365d"] = (
                snapshot["wave_return_365d"]
                - snapshot["bm_return_365d"]
            )
        else:
            snapshot["alpha_365d"] = None

        DATA_DIR.mkdir(exist_ok=True)
        pd.DataFrame([snapshot]).to_csv(OUTPUT_PATH, index=False)

        print(f"✅ live_snapshot.csv written to {OUTPUT_PATH}")

    except Exception:
        print("❌ Failed to generate live snapshot")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()