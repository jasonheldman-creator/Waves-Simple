# intelligence/adaptive_intelligence.py
# WAVES Intelligence™
# Alpha Quality + Adaptive Intelligence (Production Safe)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------------------------------
# Alpha Quality & Confidence — GOVERNANCE SAFE
# -------------------------------------------------
def render_alpha_quality_and_confidence(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    st.subheader("Alpha Quality & Confidence")

    if selected_wave is None:
        st.info("No wave selected.")
        return

    alpha_history_path = Path("data") / "alpha_history.csv"

    # -------------------------------------------------
    # Load & validate alpha history
    # -------------------------------------------------
    try:
        if not alpha_history_path.exists():
            _render_placeholders()
            return

        df = pd.read_csv(alpha_history_path)

        required_cols = {"date", "wave_id", "alpha_1d"}
        if not required_cols.issubset(df.columns):
            _render_placeholders()
            return

        df = df[df["wave_id"] == selected_wave].copy()
        if df.empty:
            _render_placeholders()
            return

        # -------------------------------------------------
        # Deduplicate → one row per day (keep LAST)
        # -------------------------------------------------
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")
        df = df.groupby("date").tail(1)

        # -------------------------------------------------
        # Enforce minimum observations
        # -------------------------------------------------
        MIN_OBS = 5
        WINDOW = 30

        if len(df) < MIN_OBS:
            _render_placeholders(note=f"Requires ≥ {MIN_OBS} daily observations")
            return

        # -------------------------------------------------
        # Rolling window (last 30 observations)
        # -------------------------------------------------
        df = df.tail(WINDOW)

        alpha = df["alpha_1d"].astype(float)

        # -------------------------------------------------
        # Core metrics
        # -------------------------------------------------
        alpha_mean = alpha.mean()
        alpha_vol = alpha.std()

        quality_score = (
            (alpha_mean / alpha_vol) if alpha_vol and not np.isnan(alpha_vol) else 0.0
        )
        quality_score = float(np.clip(quality_score * 10, -100, 100))

        confidence_level = min(len(df) / 60, 1.0) * 100

        hit_rate = (alpha > 0).mean() * 100

        # Longest negative streak
        neg = alpha < 0
        streak = 0
        max_streak = 0
        for val in neg:
            if val:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        rolling_vol = alpha.std()

        # -------------------------------------------------
        # Regime detection (slope of rolling mean)
        # -------------------------------------------------
        x = np.arange(len(alpha))
        slope = np.polyfit(x, alpha.values, 1)[0]

        if slope > 0:
            regime = "Improving"
            regime_color = "🟢"
        elif slope < 0:
            regime = "Deteriorating"
            regime_color = "🔴"
        else:
            regime = "Neutral"
            regime_color = "⚪️"

    except Exception:
        _render_placeholders()
        return

    # -------------------------------------------------
    # Render — Executive Layout
    # -------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Alpha Quality Score", f"{quality_score:.1f}")
    with c2:
        st.metric("Confidence Level", f"{confidence_level:.0f}%")

    st.divider()

    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("Hit Rate (30D)", f"{hit_rate:.0f}%")
    with c4:
        st.metric("Max Drawdown Streak", f"{max_streak} days")
    with c5:
        st.metric("Alpha Volatility", f"{rolling_vol:.4f}")

    st.divider()

    st.markdown(f"**Alpha Regime:** {regime_color} {regime}")

    # -------------------------------------------------
    # Sparkline (30D alpha)
    # -------------------------------------------------
    spark_df = pd.DataFrame({"Alpha": alpha.values}, index=df["date"])
    st.line_chart(spark_df, height=120)


# -------------------------------------------------
# Helper: Placeholder Rendering
# -------------------------------------------------
def _render_placeholders(note: str | None = None):
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Alpha Quality Score", "—")
    with c2:
        st.metric("Confidence Level", "—")

    st.divider()

    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("Hit Rate (30D)", "—")
    with c4:
        st.metric("Max Drawdown Streak", "—")
    with c5:
        st.metric("Alpha Volatility", "—")

    st.divider()

    st.markdown("**Alpha Regime:** ⚪️ Neutral")

    if note:
        st.caption(note)


# -------------------------------------------------
# Adaptive Intelligence — PREVIEW (Read-Only)
# -------------------------------------------------
def render_adaptive_intelligence_preview(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    st.subheader("Adaptive Intelligence — Preview")
    st.caption("Interpretive layer derived from Alpha Attribution (read-only)")

    if selected_wave is None:
        st.info("No wave selected.")
        return

    render_alpha_quality_and_confidence(
        snapshot_df=snapshot_df,
        source_df=source_df,
        selected_wave=selected_wave,
        return_cols=return_cols,
        benchmark_cols=benchmark_cols,
    )

    st.divider()

    st.metric("Adaptive Bias", "Neutral")
    st.metric("Regime Read", "Undetermined")
    st.metric("Signal Strength", "—")

    st.info("Preview mode only. No adaptive actions executed.")