# intelligence/adaptive_intelligence.py
# WAVES Intelligence™ — Adaptive Intelligence Module

import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
ALPHA_HISTORY_PATH = DATA_DIR / "alpha_history.csv"
LIVE_ATTRIBUTION_PATH = DATA_DIR / "live_snapshot_attribution.csv"


def render_alpha_quality_and_confidence(*args, **kwargs):
    st.subheader("Alpha Quality & Confidence")

    if not ALPHA_HISTORY_PATH.exists():
        st.info("Alpha history not available.")
        return

    df = pd.read_csv(ALPHA_HISTORY_PATH)

    required_cols = [
        "date",
        "alpha_1d",
        "alpha_30d",
        "alpha_90d",
        "alpha_365d",
        "confidence_score",
    ]

    for col in required_cols:
        if col not in df.columns:
            st.info("Alpha history schema incomplete.")
            return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    latest = df.iloc[-1]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Alpha 1D", f"{latest['alpha_1d']:.2%}")
    c2.metric("Alpha 30D", f"{latest['alpha_30d']:.2%}")
    c3.metric("Alpha 90D", f"{latest['alpha_90d']:.2%}")
    c4.metric("Alpha 1Y", f"{latest['alpha_365d']:.2%}")
    c5.metric("Confidence", f"{latest['confidence_score']:.1f}")

    st.caption("Confidence reflects consistency, drawdown control, and persistence of alpha.")


def render_adaptive_intelligence_preview(*args, **kwargs):
    st.subheader("Adaptive Intelligence Preview")

    st.markdown(
        """
This panel represents the live reasoning layer of WAVES Intelligence™.

Adaptive Intelligence continuously evaluates:
- Source of alpha (momentum, volatility control, allocation, residual skill)
- Stability vs regime change
- Signal decay and reinforcement
- Confidence-adjusted decision weighting

The outputs shown here are diagnostic, not prescriptive.
"""
    )


def render_alpha_attribution_drivers(*args, **kwargs):
    st.subheader("Alpha Attribution Drivers (Intraday)")

    if not LIVE_ATTRIBUTION_PATH.exists():
        st.info("Live alpha attribution data not available.")
        return

    df = pd.read_csv(LIVE_ATTRIBUTION_PATH)

    required_cols = [
        "wave_id",
        "timestamp",
        "alpha_beta",
        "alpha_momentum",
        "alpha_volatility",
        "alpha_allocation",
        "alpha_residual",
    ]

    for col in required_cols:
        if col not in df.columns:
            st.info("Alpha attribution schema incomplete.")
            return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    latest_ts = df["timestamp"].max()
    df_latest = df[df["timestamp"] == latest_ts]

    st.caption(f"Last update: {latest_ts}")

    for _, row in df_latest.iterrows():
        with st.container():
            st.markdown(f"**{row['wave_id']}**")

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("Beta", f"{row['alpha_beta']:.2%}")
            c2.metric("Momentum", f"{row['alpha_momentum']:.2%}")
            c3.metric("Volatility", f"{row['alpha_volatility']:.2%}")
            c4.metric("Allocation", f"{row['alpha_allocation']:.2%}")
            c5.metric("Residual", f"{row['alpha_residual']:.2%}")

            attribution_sum = (
                row["alpha_beta"]
                + row["alpha_momentum"]
                + row["alpha_volatility"]
                + row["alpha_allocation"]
                + row["alpha_residual"]
            )

            st.progress(min(max((attribution_sum + 0.05) / 0.10, 0), 1))


def render_adaptive_intelligence_panel(*args, **kwargs):
    st.header("Adaptive Intelligence")

    render_alpha_quality_and_confidence(*args, **kwargs)
    st.divider()
    render_alpha_attribution_drivers(*args, **kwargs)
    st.divider()
    render_adaptive_intelligence_preview(*args, **kwargs)