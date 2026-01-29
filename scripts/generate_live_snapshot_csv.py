# scripts/generate_live_snapshot_csv.py
# WAVES Intelligence — Canonical Live Snapshot Generator
# SAFE FULL REWRITE — NO TRUNCATION, NO MAGIC IMPORTS

import os
import sys
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------
# Ensure repo root is on PYTHONPATH
# ---------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------
# Imports (explicit, defensive)
# ---------------------------------------------------------
try:
    from waves_engine import compute_history_nav
except Exception as e:
    raise ImportError(
        "Failed to import compute_history_nav from waves_engine.py. "
        "Verify waves_engine.py exists at repo root."
    ) from e

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "live_snapshot.csv")
HORIZONS = [1, 30, 60, 365]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def compute_return(nav: pd.Series, days: int):
    """
    Compute return using last vs N-trading-days-ago.
    """
    if nav is None or nav.empty:
        return None
    if len(nav) <= days:
        return None
    try:
        return float(nav.iloc[-1] / nav.iloc[-(days + 1)] - 1.0)
    except Exception:
        return None

# ---------------------------------------------------------
# Main Snapshot Builder
# ---------------------------------------------------------
def generate_live_snapshot():
    print("▶ Generating live snapshot...")

    # 🔑 IMPORTANT:
    # Replace this with your real wave registry if/when available
    # For now, keep explicit to avoid silent failures
    waves = [
        "S&P 500 Wave",
        # add others explicitly as needed
    ]

    rows = []

    for wave_name in waves:
        try:
            print(f"  → Processing {wave_name}")

            # Core engine call (baseline, unchanged behavior)
            result = compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                days=max(HORIZONS)
            )

            if result is None or result.empty:
                print(f"    ⚠ No NAV returned for {wave_name}")
                continue

            # Expected columns
            if "wave_nav" not in result.columns or "bm_nav" not in result.columns:
                raise ValueError(
                    f"Expected wave_nav and bm_nav columns missing for {wave_name}"
                )

            wave_nav = result["wave_nav"]
            bm_nav = result["bm_nav"]

            # Diagnostics live in attrs
            diagnostics = result.attrs.get("diagnostics", {})

            row = {
                "wave_name": wave_name,
                "as_of": result.index[-1],
            }

            # Returns + alpha
            for d in HORIZONS:
                w_ret = compute_return(wave_nav, d)
                b_ret = compute_return(bm_nav, d)

                row[f"wave_return_{d}d"] = w_ret
                row[f"bm_return_{d}d"] = b_ret

                if w_ret is not None and b_ret is not None:
                    row[f"alpha_{d}d"] = w_ret - b_ret
                else:
                    row[f"alpha_{d}d"] = None

            # 🔍 Persist diagnostics (raw, untouched)
            for key in [
                "vix",
                "regime",
                "tilt_factor",
                "vix_exposure",
                "vol_adjust",
                "safe_fraction",
                "exposure",
                "aggregated_risk_state",
            ]:
                row[key] = diagnostics.get(key)

            rows.append(row)

        except Exception as e:
            print(f"    ❌ Failed {wave_name}: {e}")

    if not rows:
        raise RuntimeError("No waves produced snapshot rows — aborting.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Live snapshot written: {OUTPUT_PATH} ({len(df)} waves)")

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        generate_live_snapshot()
    except Exception as e:
        print(f"❌ Live snapshot generation failed: {e}")
        sys.exit(1)