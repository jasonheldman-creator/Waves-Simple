# scripts/generate_live_snapshot_csv.py
# WAVES Intelligence™ — Live Snapshot Generator (Correct Attribution)
# AUTHOR: Canonical, non-truncated, attribution-safe rewrite

import os
import sys
import pandas as pd

# -------------------------------------------------------------------
# Path setup (repo-root safe)
# -------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
try:
    from waves_engine import compute_history_nav
except ImportError as e:
    raise ImportError(
        "compute_history_nav could not be imported. "
        "Ensure waves_engine.py exists at repo root."
    ) from e

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "live_snapshot.csv")
HORIZONS = [1, 30, 60, 365]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def calc_return(series: pd.Series, days: int):
    if series is None or len(series) <= days:
        return None
    try:
        return series.iloc[-1] / series.iloc[-days] - 1
    except Exception:
        return None


def calc_alpha_delta(base_nav: pd.Series, overlay_nav: pd.Series, days: int):
    """
    Alpha contribution = overlay return minus base return
    This is the ONLY valid way to compute attribution.
    """
    if base_nav is None or overlay_nav is None:
        return None

    base_r = calc_return(base_nav, days)
    overlay_r = calc_return(overlay_nav, days)

    if base_r is None or overlay_r is None:
        return None

    return overlay_r - base_r


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def generate_live_snapshot():
    # TODO: replace with canonical wave registry if available
    wave_names = ["Wave1", "Wave2"]

    rows = []

    for wave_name in wave_names:
        try:
            result = compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                days=365,
                include_diagnostics=True
            )

            # ----------------------------------------------------------
            # Normalize return structure
            # ----------------------------------------------------------
            if isinstance(result, tuple):
                nav_df, diagnostics = result
            elif isinstance(result, dict):
                nav_df = result.get("nav")
                diagnostics = result.get("diagnostics", {})
            else:
                raise ValueError("Unexpected compute_history_nav return type")

            if nav_df is None or nav_df.empty:
                raise ValueError("Empty NAV dataframe")

            # ----------------------------------------------------------
            # Required NAVs
            # ----------------------------------------------------------
            wave_nav = nav_df.get("wave_nav")
            bm_nav = nav_df.get("bm_nav")

            if wave_nav is None or bm_nav is None:
                raise ValueError("wave_nav or bm_nav missing")

            row = {"wave_name": wave_name}

            # ----------------------------------------------------------
            # Returns + total alpha
            # ----------------------------------------------------------
            for d in HORIZONS:
                row[f"wave_return_{d}d"] = calc_return(wave_nav, d)
                row[f"bm_return_{d}d"] = calc_return(bm_nav, d)

            if row["wave_return_365d"] is not None and row["bm_return_365d"] is not None:
                row["alpha_total_365d"] = (
                    row["wave_return_365d"] - row["bm_return_365d"]
                )
            else:
                row["alpha_total_365d"] = None

            # ----------------------------------------------------------
            # Attribution via REAL overlay NAVs (if present)
            # ----------------------------------------------------------
            base_nav = bm_nav

            row["alpha_momentum_365d"] = calc_alpha_delta(
                base_nav,
                nav_df.get("momentum_nav"),
                365
            )

            row["alpha_volatility_365d"] = calc_alpha_delta(
                base_nav,
                nav_df.get("volatility_nav"),
                365
            )

            row["alpha_beta_365d"] = calc_alpha_delta(
                base_nav,
                nav_df.get("beta_nav"),
                365
            )

            row["alpha_allocation_365d"] = calc_alpha_delta(
                base_nav,
                nav_df.get("allocation_nav"),
                365
            )

            row["alpha_regime_365d"] = calc_alpha_delta(
                base_nav,
                nav_df.get("regime_nav"),
                365
            )

            # ----------------------------------------------------------
            # Residual selection (ONLY after real components)
            # ----------------------------------------------------------
            explained = [
                row.get("alpha_momentum_365d"),
                row.get("alpha_volatility_365d"),
                row.get("alpha_beta_365d"),
                row.get("alpha_allocation_365d"),
                row.get("alpha_regime_365d"),
            ]

            explained_sum = sum(x for x in explained if x is not None)

            if row["alpha_total_365d"] is not None:
                row["alpha_selection_365d"] = (
                    row["alpha_total_365d"] - explained_sum
                )
            else:
                row["alpha_selection_365d"] = None

            # ----------------------------------------------------------
            # Diagnostics (metadata only — NOT math inputs)
            # ----------------------------------------------------------
            for k in [
                "tilt_factor",
                "vix_exposure",
                "exposure",
                "safe_fraction",
                "aggregated_risk_state",
                "vix"
            ]:
                row[k] = diagnostics.get(k)

            rows.append(row)

        except Exception as e:
            print(f"[WARN] {wave_name} skipped: {e}")

    # -------------------------------------------------------------------
    # Write snapshot
    # -------------------------------------------------------------------
    if not rows:
        raise RuntimeError("No valid waves produced snapshot")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Live snapshot written → {OUTPUT_PATH} ({len(df)} waves)")


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        generate_live_snapshot()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)