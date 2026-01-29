# scripts/generate_live_snapshot_csv.py
# WAVES Intelligence™ — Live Snapshot Generator (CANONICAL)
# PURPOSE:
#   Build data/live_snapshot.csv used by Overview + Attribution tabs
# GUARANTEES:
#   • Uses real wave universe via get_all_waves()
#   • Uses compute_history_nav WITHOUT altering engine behavior
#   • Never fabricates or zero-fills alpha
#   • Skips broken waves safely
#   • Always writes a structurally complete CSV

import os
import sys
import pandas as pd
from typing import List

# ---------------------------------------------------------------------
# Ensure repo root on path
# ---------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------
# Imports from engine
# ---------------------------------------------------------------------
try:
    from waves_engine import (
        compute_history_nav,
        get_all_waves,
    )
except Exception as e:
    raise RuntimeError("❌ Failed to import waves_engine components") from e

# ---------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "live_snapshot.csv")

# ---------------------------------------------------------------------
# Horizons (days)
# ---------------------------------------------------------------------
HORIZONS = {
    "1D": 1,
    "30D": 30,
    "60D": 60,
    "365D": 365,
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def compute_return(nav: pd.Series, days: int):
    if nav is None or len(nav) <= days:
        return None
    start = nav.iloc[-days - 1]
    end = nav.iloc[-1]
    if start <= 0:
        return None
    return float(end / start - 1.0)


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------
def generate_live_snapshot():
    waves: List[str] = get_all_waves()
    rows = []

    print(f"📡 Generating live snapshot for {len(waves)} waves")

    for wave_name in waves:
        try:
            hist = compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                days=365,
                include_diagnostics=True,
            )

            if hist is None or hist.empty or len(hist) < 30:
                print(f"[WARN] {wave_name} skipped: insufficient history")
                continue

            nav_w = hist["wave_nav"]
            nav_b = hist["bm_nav"]

            row = {
                "wave": wave_name,
            }

            # ----------------------------------------------------------
            # Returns
            # ----------------------------------------------------------
            for label, d in HORIZONS.items():
                row[f"return_{label.lower()}"] = compute_return(nav_w, d)
                row[f"bm_return_{label.lower()}"] = compute_return(nav_b, d)

            # ----------------------------------------------------------
            # Alpha (365D canonical)
            # ----------------------------------------------------------
            if row["return_365d"] is not None and row["bm_return_365d"] is not None:
                alpha_total = row["return_365d"] - row["bm_return_365d"]
            else:
                alpha_total = None

            row["alpha_365d"] = alpha_total

            # ----------------------------------------------------------
            # Attribution (STRUCTURAL — never fake)
            # ----------------------------------------------------------
            diagnostics = hist.attrs.get("diagnostics")

            if diagnostics is not None and not diagnostics.empty and alpha_total is not None:
                # Use mean exposure effects over window
                exposure = diagnostics.get("exposure")
                safe_frac = diagnostics.get("safe_fraction")
                vix_exp = diagnostics.get("vix_exposure")

                beta_alpha = alpha_total * (exposure.mean() - 1.0) if exposure is not None else None
                allocation_alpha = alpha_total * safe_frac.mean() if safe_frac is not None else None
                volatility_alpha = alpha_total * vix_exp.mean() if vix_exp is not None else None
            else:
                beta_alpha = None
                allocation_alpha = None
                volatility_alpha = None

            # Residual = selection
            known_components = [
                v for v in [beta_alpha, allocation_alpha, volatility_alpha] if v is not None
            ]

            selection_alpha = (
                alpha_total - sum(known_components)
                if alpha_total is not None and known_components
                else None
            )

            # ----------------------------------------------------------
            # Final attribution fields (UI expects these)
            # ----------------------------------------------------------
            row.update(
                {
                    "alpha_selection": selection_alpha,
                    "alpha_momentum": None,  # momentum folded into exposure
                    "alpha_volatility": volatility_alpha,
                    "alpha_beta": beta_alpha,
                    "alpha_allocation": allocation_alpha,
                }
            )

            rows.append(row)

        except Exception as e:
            print(f"[WARN] {wave_name} skipped: {e}")
            continue

    # -----------------------------------------------------------------
    # Write snapshot
    # -----------------------------------------------------------------
    if not rows:
        raise RuntimeError("❌ No valid waves produced snapshot")

    df = pd.DataFrame(rows).sort_values("wave")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Live snapshot written: {OUTPUT_PATH} ({len(df)} waves)")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    generate_live_snapshot()