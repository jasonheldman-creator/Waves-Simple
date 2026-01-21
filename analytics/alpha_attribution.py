import os
import pandas as pd
import streamlit as st


ATTRIBUTION_PATH = "data/alpha_attribution_snapshot.csv"
LIVE_SNAPSHOT_PATH = "data/live_snapshot.csv"


def build_alpha_attribution_snapshot():
    """
    Builds strategy-level alpha attribution by source:
    - Market
    - VIX / Volatility overlay
    - Momentum overlay
    - Rotation / factor overlay
    - Stock selection (residual)
    """

    if not os.path.exists(LIVE_SNAPSHOT_PATH):
        st.warning("Alpha attribution unavailable — live snapshot missing")
        return

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)

    required = [
        "wave_name",
        "Alpha_1D",
        "benchmark_return_1D",
        "strategy_return_1D",
        "vix_regime",
        "momentum_state",
        "rotation_state",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Alpha attribution unavailable — missing columns: {missing}")
        return

    out = []

    for _, r in df.iterrows():
        alpha_total = r["Alpha_1D"]

        alpha_market = r["strategy_return_1D"] - r["benchmark_return_1D"]

        alpha_vix = 0.0
        if r["vix_regime"] == "RISK_OFF":
            alpha_vix = alpha_total * 0.30

        alpha_momentum = 0.0
        if r["momentum_state"] == "ON":
            alpha_momentum = alpha_total * 0.25

        alpha_rotation = 0.0
        if r["rotation_state"] == "ON":
            alpha_rotation = alpha_total * 0.20

        alpha_stock_selection = (
            alpha_total
            - alpha_market
            - alpha_vix
            - alpha_momentum
            - alpha_rotation
        )

        out.append(
            {
                "wave_name": r["wave_name"],
                "alpha_total": alpha_total,
                "alpha_market": alpha_market,
                "alpha_vix": alpha_vix,
                "alpha_momentum": alpha_momentum,
                "alpha_rotation": alpha_rotation,
                "alpha_stock_selection": alpha_stock_selection,
            }
        )

    out_df = pd.DataFrame(out)
    out_df.to_csv(ATTRIBUTION_PATH, index=False)