# ============================================================
# intelligence/adaptive_intelligence.py
# WAVES Intelligence™
# Alpha Quality + Adaptive Intelligence
# ============================================================

from pathlib import Path
import pandas as pd
import streamlit as st


# -------------------------------------------------
# Alpha Quality & Confidence (LIVE, HISTORY-AWARE)
# -------------------------------------------------
def render_alpha_quality_and_confidence(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    """
    Renders Alpha Quality Score and Confidence Level.

    Behavior:
    - Consumes append-only data/alpha_history.csv if present
    - Deduplicates to one row per wave per day (keeps last)
    - Requires >= 20 daily observations
    - Gracefully degrades to placeholders if insufficient data
    - No writes, no side effects, no upstream dependencies
    """

    # Explicitly mark unused params (signature preserved by design)
    _ = snapshot_df, source_df, return_cols, benchmark_cols

    st.subheader("Alpha Quality & Confidence")

    if selected_wave is None:
        st.info("No wave selected.")
        return

    # ------------------------------------------------------------
    # Load alpha history
    # ------------------------------------------------------------
    alpha_history_path = Path("data") / "alpha_history.csv"

    try:
        if not alpha_history_path.exists():
            st.metric("Alpha Quality Score", "—")
            st.metric("Confidence Level", "—")
            return

        df = pd.read_csv(alpha_history_path)

        # Required schema
        required_cols = {"date", "wave_id", "alpha_1d"}
        if not required_cols.issubset(df.columns):
            st.metric("Alpha Quality Score", "—")
            st.metric("Confidence Level", "—")
            return

        # ------------------------------------------------------------
        # Filter to selected wave
        # ------------------------------------------------------------
        df = df[df["wave_id"] == selected_wave].copy()

        if df.empty:
            st.metric("Alpha Quality Score", "—")
            st.metric("Confidence Level", "—")
            return

        # ------------------------------------------------------------
        # Deduplicate: one row per day (keep last)
        # ------------------------------------------------------------
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")
        df = df.groupby("date", as_index=False).tail(1)

        # ------------------------------------------------------------
        # Require minimum observations
        # ------------------------------------------------------------
        MIN_OBS = 20
        if len(df) < MIN_OBS:
            st.metric("Alpha Quality Score", "—")
            st.metric("Confidence Level", "—")
            st.caption(f"Requires ≥ {MIN_OBS} daily observations")
            return

        # ------------------------------------------------------------
        # Compute Alpha Quality Score (mean / volatility)
        # ------------------------------------------------------------
        alpha_series = df["alpha_1d"].astype(float)

        alpha_mean = alpha_series.mean()
        alpha_vol = alpha_series.std()

        if pd.isna(alpha_vol) or alpha_vol == 0:
            raw_quality = 0.0
        else:
            raw_quality = alpha_mean / alpha_vol

        # Normalize to 0–100 (institutional-friendly)
        quality_score = max(min((raw_quality + 2.0) * 25.0, 100.0), 0.0)

        # ------------------------------------------------------------
        # Confidence Level (sample-size based, capped)
        # ------------------------------------------------------------
        confidence_level = min(len(df) / 60.0, 1.0) * 100.0

    except Exception:
        # Silent fail — never block the UI
        st.metric("Alpha Quality Score", "—")
        st.metric("Confidence Level", "—")
        return

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Alpha Quality Score",
            f"{quality_score:.1f}"
        )

    with col2:
        st.metric(
            "Confidence Level",
            f"{confidence_level:.0f}%"
        )


# -------------------------------------------------
# Adaptive Intelligence — PREVIEW (UNCHANGED)
# -------------------------------------------------
def render_adaptive_intelligence_preview(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    """
    Read-only interpretive preview derived from Alpha Attribution.
    No trading logic. No persistence. No actions.
    """

    st.subheader("Adaptive Intelligence — Preview")
    st.caption("Interpretive layer derived from Alpha Attribution (read-only)")

    if selected_wave is None:
        st.info("No wave selected.")
        return

    st.write(f"Selected Wave: {selected_wave}")

    # Reuse Alpha Quality safely
    render_alpha_quality_and_confidence(
        snapshot_df=snapshot_df,
        source_df=source_df,
        selected_wave=selected_wave,
        return_cols=return_cols,
        benchmark_cols=benchmark_cols,
    )

    st.divider()

    # Preview-only interpretive outputs
    st.metric("Adaptive Bias", "Neutral")
    st.metric("Regime Read", "Undetermined")
    st.metric("Signal Strength", "—")

    st.info("Preview mode only. No adaptive actions executed.")