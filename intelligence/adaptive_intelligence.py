# intelligence/adaptive_intelligence.py
# WAVES Intelligence™
# Alpha Quality + Adaptive Intelligence (Preview Only)

import streamlit as st


# -------------------------------------------------
# Alpha Quality & Confidence (EXISTING / UNCHANGED)
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

    st.write(f"Selected Wave: {selected_wave}")

    # Safe placeholders (no execution logic)
    st.metric("Alpha Quality Score", "—")
    st.metric("Confidence Level", "—")


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

    # Preview-only interpretive outputs (static by design in B2)
    st.metric("Adaptive Bias", "Neutral")
    st.metric("Regime Read", "Undetermined")
    st.metric("Signal Strength", "—")

    st.info("Preview mode only. No adaptive actions executed.")