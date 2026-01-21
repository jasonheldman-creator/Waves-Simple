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

    Snapshot schema assumptions (from live_snapshot.csv):
        - Wave_ID   (canonical identifier)
        - Wave      (display name)
        - Alpha_1D
        - Benchmark_Return_1D
        - Return_1D
        - VIX_Regime
        - strategy_state
        - strategy_stack_applied

    Output columns:
        - wave_id
        - wave_name
        - alpha_market
        - alpha_vix
        - alpha_momentum
        - alpha_rotation
        - alpha_stock_selection
        - alpha_total

    Returns:
        (bool, str)
    """

    # ---------------------------------------------------------------
    # Preconditions
    # ---------------------------------------------------------------
    if not os.path.exists(LIVE_SNAPSHOT_PATH):
        return False, "live_snapshot.csv not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)

    # Required identifiers
    required_cols = ["Wave_ID", "Wave", "Alpha_1D"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    # Optional columns (graceful degradation)
    has_benchmark = "Benchmark_Return_1D" in df.columns
    has_return = "Return_1D" in df.columns
    has_vix = "VIX_Regime" in df.columns
    has_strategy_stack = "strategy_stack_applied" in df.columns

    # ---------------------------------------------------------------
    # Attribution logic
    # ---------------------------------------------------------------
    rows = []

    for _, r in df.iterrows():
        try:
            alpha_total = float(r["Alpha_1D"])

            # -------------------------------------------------------
            # Market alpha
            # -------------------------------------------------------
            if has_benchmark and has_return:
                alpha_market = float(r["Return_1D"] - r["Benchmark_Return_1D"])
            else:
                alpha_market = alpha_total

            # -------------------------------------------------------
            # VIX overlay attribution
            # -------------------------------------------------------
            alpha_vix = 0.0
            if has_vix and str(r["VIX_Regime"]).lower() in {"low", "risk_off", "risk-off"}:
                alpha_vix = alpha_total * 0.30

            # -------------------------------------------------------
            # Strategy overlays (momentum / rotation)
            # -------------------------------------------------------
            alpha_momentum = 0.0
            alpha_rotation = 0.0

            if has_strategy_stack and isinstance(r["strategy_stack_applied"], str):
                stack = r["strategy_stack_applied"].lower()

                if "momentum" in stack:
                    alpha_momentum = alpha_total * 0.25

                if "rotation" in stack or "trend_confirmation" in stack:
                    alpha_rotation = alpha_total * 0.20

            # -------------------------------------------------------
            # Residual = stock selection
            # -------------------------------------------------------
            alpha_stock_selection = (
                alpha_total
                - alpha_market
                - alpha_vix
                - alpha_momentum
                - alpha_rotation
            )

            rows.append(
                {
                    "wave_id": r["Wave_ID"],
                    "wave_name": r["Wave"],
                    "alpha_market": alpha_market,
                    "alpha_vix": alpha_vix,
                    "alpha_momentum": alpha_momentum,
                    "alpha_rotation": alpha_rotation,
                    "alpha_stock_selection": alpha_stock_selection,
                    "alpha_total": alpha_total,
                }
            )

        except Exception:
            # Never crash the app because of one bad row
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
# CLI support
# -------------------------------------------------------------------

if __name__ == "__main__":
    ok, msg = build_alpha_attribution_snapshot()
    print(msg)