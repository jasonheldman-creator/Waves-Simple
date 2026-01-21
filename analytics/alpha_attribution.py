# analytics/alpha_attribution.py

import os
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ATTRIBUTION_PATH = "data/alpha_attribution_snapshot.csv"
LIVE_SNAPSHOT_PATH = "data/live_snapshot.csv"


# -------------------------------------------------------------------
# Core generator (SAFE to call from app.py)
# -------------------------------------------------------------------

def build_alpha_attribution_snapshot():
    """
    Generate strategy-level alpha attribution by source.

    Output columns:
        - wave_name
        - alpha_market
        - alpha_vix
        - alpha_momentum
        - alpha_rotation
        - alpha_stock_selection
        - alpha_total

    This function is SAFE TO CALL from app.py.
    It is schema-tolerant and will gracefully degrade if
    strategy-state columns are missing.

    Returns:
        (bool, str): success flag and status message
    """

    # ---------------------------------------------------------------
    # Preconditions
    # ---------------------------------------------------------------
    if not os.path.exists(LIVE_SNAPSHOT_PATH):
        return False, "live_snapshot.csv not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)

    # ---------------------------------------------------------------
    # Normalize wave identifier (CRITICAL FIX)
    # ---------------------------------------------------------------
    if "wave_name" not in df.columns:
        if "display_name" in df.columns:
            df["wave_name"] = df["display_name"]
        elif "wave" in df.columns:
            df["wave_name"] = df["wave"]
        elif "wave_id" in df.columns:
            df["wave_name"] = df["wave_id"]
        else:
            return False, "Missing wave identifier (wave_name / display_name / wave_id)"

    # ---------------------------------------------------------------
    # Minimal required columns
    # ---------------------------------------------------------------
    if "Alpha_1D" not in df.columns:
        return False, "Missing required column: Alpha_1D"

    # Optional strategy-state columns (graceful fallback)
    has_benchmark = "benchmark_return_1D" in df.columns
    has_strategy = "strategy_return_1D" in df.columns
    has_vix = "vix_regime" in df.columns
    has_momentum = "momentum_state" in df.columns
    has_rotation = "rotation_state" in df.columns

    # ---------------------------------------------------------------
    # Attribution logic
    # ---------------------------------------------------------------
    rows = []

    for _, r in df.iterrows():
        try:
            alpha_total = float(r["Alpha_1D"])

            # Market-relative alpha
            if has_benchmark and has_strategy:
                alpha_market = float(
                    r["strategy_return_1D"] - r["benchmark_return_1D"]
                )
            else:
                alpha_market = alpha_total

            # VIX / volatility overlay
            alpha_vix = (
                alpha_total * 0.30
                if has_vix and str(r.get("vix_regime", "")).upper() == "RISK_OFF"
                else 0.0
            )

            # Momentum overlay
            alpha_momentum = (
                alpha_total * 0.25
                if has_momentum and str(r.get("momentum_state", "")).upper() == "ON"
                else 0.0
            )

            # Rotation / factor overlay
            alpha_rotation = (
                alpha_total * 0.20
                if has_rotation and str(r.get("rotation_state", "")).upper() == "ON"
                else 0.0
            )

            # Residual = stock selection
            alpha_stock_selection = (
                alpha_total
                - alpha_market
                - alpha_vix
                - alpha_momentum
                - alpha_rotation
            )

            rows.append(
                {
                    "wave_name": r["wave_name"],
                    "alpha_market": alpha_market,
                    "alpha_vix": alpha_vix,
                    "alpha_momentum": alpha_momentum,
                    "alpha_rotation": alpha_rotation,
                    "alpha_stock_selection": alpha_stock_selection,
                    "alpha_total": alpha_total,
                }
            )

        except Exception:
            # Skip malformed rows, never crash the app
            continue

    if not rows:
        return False, "No valid rows produced for attribution"

    # ---------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(ATTRIBUTION_PATH), exist_ok=True)
    out_df.to_csv(ATTRIBUTION_PATH, index=False)

    return True, f"Wrote {len(out_df)} rows to {ATTRIBUTION_PATH}"


# -------------------------------------------------------------------
# Optional CLI execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    ok, msg = build_alpha_attribution_snapshot()
    print(msg)