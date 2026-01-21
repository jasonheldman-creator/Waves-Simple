import os
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ATTRIBUTION_PATH = "data/alpha_attribution_snapshot.csv"
LIVE_SNAPSHOT_PATH = "data/live_snapshot.csv"


# -------------------------------------------------------------------
# Core generator
# -------------------------------------------------------------------

def generate_alpha_attribution_snapshot():
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
    It writes data/alpha_attribution_snapshot.csv if inputs are valid.
    """

    # --- Preconditions ------------------------------------------------
    if not os.path.exists(LIVE_SNAPSHOT_PATH):
        return False, "live_snapshot.csv not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)

    required_cols = [
        "wave_name",
        "Alpha_1D",
        "benchmark_return_1D",
        "strategy_return_1D",
        "vix_regime",
        "momentum_state",
        "rotation_state",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    # --- Attribution logic -------------------------------------------
    rows = []

    for _, r in df.iterrows():
        alpha_total = float(r["Alpha_1D"])

        # Market-relative alpha
        alpha_market = float(
            r["strategy_return_1D"] - r["benchmark_return_1D"]
        )

        # VIX / volatility overlay attribution
        alpha_vix = 0.0
        if str(r["vix_regime"]).upper() == "RISK_OFF":
            alpha_vix = alpha_total * 0.30

        # Momentum overlay attribution
        alpha_momentum = 0.0
        if str(r["momentum_state"]).upper() == "ON":
            alpha_momentum = alpha_total * 0.25

        # Rotation / factor overlay attribution
        alpha_rotation = 0.0
        if str(r["rotation_state"]).upper() == "ON":
            alpha_rotation = alpha_total * 0.20

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

    # --- Write output -------------------------------------------------
    out_df = pd.DataFrame(rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(ATTRIBUTION_PATH), exist_ok=True)

    out_df.to_csv(ATTRIBUTION_PATH, index=False)

    return True, f"Wrote {len(out_df)} rows to {ATTRIBUTION_PATH}"


# -------------------------------------------------------------------
# Optional CLI entry (safe if someone runs file directly)
# -------------------------------------------------------------------

if __name__ == "__main__":
    ok, msg = generate_alpha_attribution_snapshot()
    print(msg)