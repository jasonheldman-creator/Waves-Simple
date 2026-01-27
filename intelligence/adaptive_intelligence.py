# intelligence/adaptive_intelligence.py
# WAVES Intelligence™
# Alpha Quality + Adaptive Intelligence (Governance-First)

import streamlit as st
import pandas as pd
import numpy as np


# -------------------------------------------------
# Alpha Quality & Confidence
# -------------------------------------------------
def render_alpha_quality_and_confidence(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
    *,
    min_observations: int = 20,
):
    """
    Governance-first Alpha Quality & Confidence renderer.

    • Computes metrics ONLY when sufficient alpha history exists
    • Degrades silently to placeholders when data is insufficient
    • Never fabricates or infers confidence
    """

    st.subheader("Alpha Quality & Confidence")

    if selected_wave is None:
        st.info("No wave selected.")
        return

    st.caption(f"Attribution profile for the selected wave: {selected_wave}")
    st.divider()

    # -------------------------------------------------
    # No alpha history available → graceful degradation
    # -------------------------------------------------
    if source_df is None or not isinstance(source_df, pd.DataFrame):
        _render_alpha_placeholders()
        return

    if "alpha" not in source_df.columns:
        _render_alpha_placeholders()
        return

    alpha_series = source_df["alpha"].dropna()

    if len(alpha_series) < min_observations:
        _render_alpha_placeholders(
            note=f"Insufficient observations ({len(alpha_series)}/{min_observations})"
        )
        return

    # -------------------------------------------------
    # Compute metrics (truth-gated)
    # -------------------------------------------------
    mean_alpha = alpha_series.mean()
    alpha_vol = alpha_series.std(ddof=0)

    # Quality: normalized mean vs volatility
    alpha_quality_score = (
        mean_alpha / alpha_vol if alpha_vol > 0 else np.nan
    )

    # Confidence: stability proxy (bounded, interpretable)
    confidence_level = min(1.0, len(alpha_series) / (2 * min_observations))

    # -------------------------------------------------
    # Render metrics
    # -------------------------------------------------
    cols = st.columns(2)

    with cols[0]:
        st.metric(
            "Alpha Quality Score",
            f"{alpha_quality_score:.2f}" if np.isfinite(alpha_quality_score) else "—",
        )

    with cols[1]:
        st.metric(
            "Confidence Level",
            f"{confidence_level:.0%}" if np.isfinite(confidence_level) else "—",
        )


def _render_alpha_placeholders(note: str | None = None):
    """Unified placeholder renderer (no warnings, no noise)."""
    cols = st.columns(2)

    with cols[0]:
        st.metric("Alpha Quality Score", "—")

    with cols[1]:
        st.metric("Confidence Level", "—")

    if note:
        st.caption(note)


# -------------------------------------------------
# Adaptive Intelligence — PREVIEW (B2)
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

    # Reuse Alpha Quality safely (single truth path)
    render_alpha_quality_and_confidence(
        snapshot_df=snapshot_df,
        source_df=source_df,
        selected_wave=selected_wave,
        return_cols=return_cols,
        benchmark_cols=benchmark_cols,
    )

    st.divider()

    # Preview-only interpretive outputs (static by design in B2)
    st.metric("Adaptive Bias", "Neutral")
    st.metric("Regime Read", "Undetermined")
    st.metric("Signal Strength", "—")

    st.info("Preview mode only. No adaptive actions executed.")