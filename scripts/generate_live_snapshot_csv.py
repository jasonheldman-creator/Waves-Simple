# scripts/generate_live_snapshot_csv.py
# WAVES Intelligence™ — Canonical Live Snapshot Generator
# CONTRACT-ALIGNED WITH compute_history_nav (baseline API)

import os
import sys
import pandas as pd
from typing import Dict, Any

# ------------------------------------------------------------
# Path bootstrap (repo-root safe)
# ------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ------------------------------------------------------------
# Imports (ENGINE CONTRACT)
# ------------------------------------------------------------

from waves_engine import (
    compute_history_nav,
    get_all_waves,
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "live_snapshot.csv")
HORIZONS = [1, 30, 60, 365]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _calc_return(nav: pd.Series, days: int):
    if nav is None or len(nav) <= days:
        return None
    start = float(nav.iloc[-days])
    end = float(nav.iloc[-1])
    if start <= 0:
        return None
    return (end / start) - 1.0


# ------------------------------------------------------------
# Main generator
# ------------------------------------------------------------

def generate_live_snapshot():
    waves = get_all_waves()
    rows = []

    for wave in waves:
        try:
            df = compute_history_nav(
                wave_name=wave,
                mode="Standard",
                days=365,
                include_diagnostics=True,
            )

            # Engine contract: EMPTY DF = safe skip
            if df is None or df.empty:
                print(f"[WARN] {wave} skipped: empty history")
                continue

            # Required columns (baseline API)
            if not {"wave_nav", "bm_nav"}.issubset(df.columns):
                print(f"[WARN] {wave} skipped: missing NAV columns")
                continue

            wave_nav = df["wave_nav"]
            bm_nav = df["bm_nav"]

            row: Dict[str, Any] = {
                "wave": wave,
            }

            # -----------------------------
            # Returns
            # -----------------------------

            for h in HORIZONS:
                w_ret = _calc_return(wave_nav, h)
                b_ret = _calc_return(bm_nav, h)

                row[f"wave_return_{h}d"] = w_ret
                row[f"bm_return_{h}d"] = b_ret

                if h == 365 and w_ret is not None and b_ret is not None:
                    row["alpha_365d"] = w_ret - b_ret

            # -----------------------------
            # Diagnostics (ATTRS CONTRACT)
            # -----------------------------

            diagnostics = df.attrs.get("diagnostics")

            if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
                latest = diagnostics.iloc[-1]

                row.update(
                    {
                        "regime": latest.get("regime"),
                        "vix": latest.get("vix"),
                        "safe_fraction": latest.get("safe_fraction"),
                        "exposure": latest.get("exposure"),
                        "vol_adjust": latest.get("vol_adjust"),
                        "vix_exposure": latest.get("vix_exposure"),
                        "aggregated_risk_state": latest.get("aggregated_risk_state"),
                    }
                )

            rows.append(row)

        except Exception as e:
            print(f"[WARN] {wave} skipped: {e}")
            continue

    # ------------------------------------------------------------
    # Write snapshot
    # ------------------------------------------------------------

    if not rows:
        raise RuntimeError("No valid waves produced snapshot")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Live snapshot written: {OUTPUT_PATH} ({len(out)} waves)")


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    try:
        generate_live_snapshot()
    except Exception as e:
        print(f"[FATAL] Failed to generate live snapshot: {e}")
        sys.exit(1)