import os
import sys
import pandas as pd
from typing import List, Dict, Any

# -----------------------------------------------------------------------------
# Ensure repo root is on sys.path
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# -----------------------------------------------------------------------------
# Imports from core engine (CANONICAL)
# -----------------------------------------------------------------------------
try:
    from waves_engine import compute_history_nav, get_all_waves
except ImportError as e:
    raise ImportError(
        "Failed to import compute_history_nav / get_all_waves from waves_engine. "
        "Ensure waves_engine.py exists at repo root and is importable."
    ) from e

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "live_snapshot.csv")
HORIZONS = [1, 30, 60, 365]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def calc_return(nav: pd.Series, days: int):
    """
    Compute simple return over `days`.
    Returns None if insufficient history.
    """
    if nav is None or len(nav) <= days:
        return None
    try:
        return float(nav.iloc[-1] / nav.iloc[-days] - 1.0)
    except Exception:
        return None


def extract_nav_columns(nav_df: pd.DataFrame):
    """
    Standardize NAV column access.
    """
    if "wave_nav" not in nav_df.columns or "bm_nav" not in nav_df.columns:
        raise ValueError("NAV dataframe missing required columns: wave_nav / bm_nav")
    return nav_df["wave_nav"], nav_df["bm_nav"]


# -----------------------------------------------------------------------------
# Main generator
# -----------------------------------------------------------------------------
def generate_live_snapshot():
    waves: List[str] = get_all_waves()

    if not waves:
        raise RuntimeError("get_all_waves() returned no waves")

    rows: List[Dict[str, Any]] = []

    for wave_name in waves:
        try:
            # -----------------------------------------------------------------
            # Canonical history call (DO NOT GUESS SIGNATURE)
            # -----------------------------------------------------------------
            result = compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                days=365,
                include_diagnostics=True,
            )

            # -----------------------------------------------------------------
            # Normalize return shape
            # -----------------------------------------------------------------
            if isinstance(result, tuple):
                nav_df, diagnostics = result
            elif isinstance(result, dict):
                nav_df = result.get("nav")
                diagnostics = result.get("diagnostics", {})
            else:
                raise ValueError(f"Unexpected return type: {type(result)}")

            if nav_df is None or nav_df.empty:
                raise ValueError("Empty NAV dataframe")

            wave_nav, bm_nav = extract_nav_columns(nav_df)

            # -----------------------------------------------------------------
            # Base snapshot row
            # -----------------------------------------------------------------
            row: Dict[str, Any] = {
                "wave_name": wave_name,
            }

            # -----------------------------------------------------------------
            # Returns + alpha
            # -----------------------------------------------------------------
            for d in HORIZONS:
                w_ret = calc_return(wave_nav, d)
                b_ret = calc_return(bm_nav, d)

                row[f"wave_return_{d}d"] = w_ret
                row[f"bm_return_{d}d"] = b_ret

                if w_ret is not None and b_ret is not None:
                    row[f"alpha_{d}d"] = w_ret - b_ret
                else:
                    row[f"alpha_{d}d"] = None

            # -----------------------------------------------------------------
            # Diagnostics passthrough (NO SYNTHETIC MATH HERE)
            # These are raw signals used later by attribution adapters
            # -----------------------------------------------------------------
            if isinstance(diagnostics, dict):
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
            else:
                # Diagnostics missing or malformed
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
                    row[key] = None

            rows.append(row)

        except Exception as e:
            print(f"[WARN] {wave_name} skipped: {e}")
            continue

    if not rows:
        raise RuntimeError("No valid waves produced snapshot rows")

    # -------------------------------------------------------------------------
    # Write CSV
    # -------------------------------------------------------------------------
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[OK] Live snapshot written: {OUTPUT_PATH} ({len(df)} waves)")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        generate_live_snapshot()
    except Exception as e:
        print(f"[FATAL] Failed to generate live snapshot: {e}")
        sys.exit(1)