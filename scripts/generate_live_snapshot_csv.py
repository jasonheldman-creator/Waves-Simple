"""
WAVES Intelligence™
Generate Live Snapshot CSV

PURPOSE
-------
Builds data/live_snapshot.csv from compute_history_nav diagnostics
and wires in alpha attribution components.

This file is executed as a SCRIPT via GitHub Actions, so it must
manually ensure repo-root import resolution.

DO NOT CONVERT TO PACKAGE IMPORTS.
"""

# ---------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CRITICAL FIX)
# ---------------------------------------------------------
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------
# Standard imports
# ---------------------------------------------------------
import pandas as pd
import traceback

# ---------------------------------------------------------
# Internal imports (now safe)
# ---------------------------------------------------------
from analytics.alpha_attribution_adapter import AlphaAttributionAdapter

# NOTE:
# compute_history_nav is assumed to already exist and return
# diagnostics-rich history data. We do NOT reimplement it here.
from waves_engine import compute_history_nav


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def build_snapshot_from_diagnostics(diagnostics: dict) -> dict:
    """
    Convert diagnostics dict into a single snapshot row,
    including alpha attribution components.
    """

    adapter = AlphaAttributionAdapter(diagnostics)
    alpha_components = adapter.convert_to_alpha_attribution()

    snapshot = {
        # Core returns
        "return_1d": diagnostics.get("return_1d"),
        "return_30d": diagnostics.get("return_30d"),
        "return_60d": diagnostics.get("return_60d"),
        "return_365d": diagnostics.get("return_365d"),

        # Alpha headline
        "alpha_365d": diagnostics.get("alpha_365d"),

        # Risk diagnostics
        "vix": diagnostics.get("vix"),
        "regime": diagnostics.get("regime"),
        "exposure": diagnostics.get("exposure"),
        "safe_fraction": diagnostics.get("safe_fraction"),
        "tilt_factor": diagnostics.get("tilt_factor"),
        "aggregated_risk_state": diagnostics.get("aggregated_risk_state"),
    }

    # Merge alpha attribution components
    snapshot.update(alpha_components)

    return snapshot


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
def main():
    print("▶ Generating live snapshot...")

    try:
        # -------------------------------------------------
        # Compute history + diagnostics
        # -------------------------------------------------
        history_result = compute_history_nav(
            return_diagnostics=True,
            return_latest_only=True,
        )

        if not history_result:
            raise RuntimeError("compute_history_nav returned no data")

        diagnostics = history_result.get("diagnostics")
        if not diagnostics:
            raise RuntimeError("No diagnostics found in compute_history_nav output")

        # -------------------------------------------------
        # Build snapshot row
        # -------------------------------------------------
        snapshot_row = build_snapshot_from_diagnostics(diagnostics)

        df = pd.DataFrame([snapshot_row])

        # -------------------------------------------------
        # Write CSV
        # -------------------------------------------------
        df.to_csv(LIVE_SNAPSHOT_PATH, index=False)

        print(f"✅ Live snapshot written to {LIVE_SNAPSHOT_PATH}")

    except Exception as e:
        print("❌ Failed to generate live snapshot")
        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    main()