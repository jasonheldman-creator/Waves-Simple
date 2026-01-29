"""
generate_live_snapshot_csv.py

Builds live_snapshot.csv from canonical WAVES engine outputs.

This script:
- Calls compute_history_nav (the ECONOMICALLY CAUSAL engine)
- Extracts diagnostics (momentum, VIX, exposure, safe allocation, etc.)
- Converts diagnostics into explicit alpha attribution components
- Writes a flat snapshot CSV for Streamlit consumption

NO strategy math is invented here.
NO returns are recomputed here.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from analytics.alpha_attribution_adapter import AlphaAttributionAdapter
from waves_engine import compute_history_nav
from wave_registry import ALL_WAVES

OUTPUT_PATH = Path("data/live_snapshot.csv")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _safe_get(d: Dict[str, Any], key: str, default=0.0):
    val = d.get(key, default)
    if val is None:
        return default
    return val


# -------------------------------------------------------------------
# Main snapshot builder
# -------------------------------------------------------------------

def build_live_snapshot() -> pd.DataFrame:
    rows = []

    for wave_name in ALL_WAVES:
        try:
            # -------------------------------------------------------
            # 1. Call the REAL engine (this is the causal path)
            # -------------------------------------------------------
            result = compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                include_diagnostics=True,
            )

            if result is None or result.empty:
                continue

            diagnostics = result.attrs.get("diagnostics", {})
            if not diagnostics:
                continue

            # -------------------------------------------------------
            # 2. Convert diagnostics → attribution components
            # -------------------------------------------------------
            adapter = AlphaAttributionAdapter(diagnostics)
            attribution = adapter.convert()

            # -------------------------------------------------------
            # 3. Build snapshot row
            # -------------------------------------------------------
            row = {
                "Wave": wave_name,

                # Core returns
                "Return_30D": _safe_get(diagnostics, "ret_30d"),
                "Return_60D": _safe_get(diagnostics, "ret_60d"),
                "Return_365D": _safe_get(diagnostics, "ret_365d"),

                # Total alpha
                "Alpha_30D": _safe_get(diagnostics, "alpha_30d"),
                "Alpha_60D": _safe_get(diagnostics, "alpha_60d"),
                "Alpha_365D": _safe_get(diagnostics, "alpha_365d"),

                # ---------------------------------------------------
                # Strategy attribution (THE IMPORTANT PART)
                # ---------------------------------------------------
                "Alpha_Selection_365D": attribution["selection"],
                "Alpha_Momentum_365D": attribution["momentum"],
                "Alpha_Volatility_365D": attribution["volatility"],
                "Alpha_Beta_365D": attribution["beta"],
                "Alpha_Allocation_365D": attribution["allocation"],
                "Alpha_Residual_365D": attribution["residual"],

                # Diagnostics (optional but useful)
                "Tilt_Factor": _safe_get(diagnostics, "tilt_factor"),
                "VIX": _safe_get(diagnostics, "vix"),
                "Exposure": _safe_get(diagnostics, "exposure"),
                "Safe_Fraction": _safe_get(diagnostics, "safe_fraction"),
                "Aggregated_Risk_State": diagnostics.get("aggregated_risk_state", "unknown"),
            }

            rows.append(row)

        except Exception as e:
            print(f"[WARN] Failed snapshot for {wave_name}: {e}")
            continue

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    df = build_live_snapshot()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Live snapshot written to {OUTPUT_PATH}")