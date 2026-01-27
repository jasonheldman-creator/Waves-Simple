# ============================================================
# intelligence/adaptive_intelligence.py
# WAVES Intelligence™
# Alpha Quality, Attribution Drivers & Adaptive Intelligence
# ============================================================

from pathlib import Path
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Alpha Quality & Confidence
# ------------------------------------------------------------
def render_alpha_quality_and_confidence(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    """
    Renders Alpha Quality & Confidence diagnostics for a selected wave.

    • Consumes append-only data/alpha_history.csv
    • Deduplicates to one row per wave per day
    • Requires a minimum observation threshold
    • Degrades gracefully when data is missing
    • No trading logic, no persistence, no side effects
    """

    with st.container():
        st.subheader("Alpha Quality & Confidence")
        st.caption("Risk-adjusted alpha diagnostics derived from daily realized performance")

        if selected_wave is None:
            st.info("No wave selected.")
            return

        alpha_history_path = Path("data") / "alpha_history.csv"

        try:
            if not alpha_history_path.exists():
                _render_alpha_placeholders()
                return

            df = pd.read_csv(alpha_history_path)

            required_cols = {"date", "wave_id", "alpha_1d"}
            if not required_cols.issubset(df.columns):
                _render_alpha_placeholders()
                return

            df = df[df["wave_id"] == selected_wave].copy()
            if df.empty:
                _render_alpha_placeholders()
                return

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            if df.empty:
                _render_alpha_placeholders()
                return

            df = df.sort_values("date")
            df = df.groupby("date").tail(1)

            MIN_OBS = 5
            if len(df) < MIN_OBS:
                _render_alpha_placeholders(
                    caption=f"Requires ≥ {MIN_OBS} daily observations"
                )
                return

            alpha_series = df["alpha_1d"]

            alpha_mean = alpha_series.mean()
            alpha_vol = alpha_series.std()

            quality_score = 0.0 if pd.isna(alpha_vol) or alpha_vol == 0 else alpha_mean / alpha_vol
            quality_score = max(min(quality_score * 10, 100), -100)

            confidence_level = min(len(df) / 60, 1.0) * 100

            window = min(20, len(alpha_series))
            rolling_vol = alpha_series.rolling(window=window).std()
            alpha_vol_metric = rolling_vol.iloc[-1] if not rolling_vol.empty else float("nan")

            hit_rate = (alpha_series > 0).mean() * 100

            if alpha_mean > 0:
                alpha_regime = "Positive"
            elif alpha_mean < 0:
                alpha_regime = "Negative"
            else:
                alpha_regime = "Neutral"

        except Exception:
            _render_alpha_placeholders()
            return

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Alpha Quality Score", f"{quality_score:.1f}")
        with col2:
            st.metric("Confidence Level", f"{confidence_level:.0f}%")

        st.divider()

        st.line_chart(
            df.set_index("date")["alpha_1d"],
            use_container_width=True,
        )

        st.divider()

        col3, col4, col5 = st.columns(3)

        with col3:
            st.metric(
                "Alpha Volatility (Rolling)",
                "—" if pd.isna(alpha_vol_metric) else f"{alpha_vol_metric:.4f}",
            )

        with col4:
            st.metric("Hit Rate", f"{hit_rate:.0f}%")

        with col5:
            st.metric("Alpha Regime", alpha_regime)


# ------------------------------------------------------------
# Alpha Attribution Drivers (EXPLANATORY SECTION)
# ------------------------------------------------------------
def render_alpha_attribution_drivers(
    snapshot_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    """
    Explanatory decomposition of realized alpha.
    This is a UI-only placeholder until full driver data is wired.
    """

    with st.container():
        if selected_wave is None:
            st.info("No wave selected.")
            return

        st.caption("Strategy-level attribution of realized alpha")

        # NOTE:
        # The computational driver logic already exists elsewhere in the repo.
        # This renderer is intentionally conservative and non-fabricating.

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("VIX / Volatility Overlay", "—")

        with col2:
            st.metric("Momentum / Trend Overlay", "—")

        with col3:
            st.metric("Stock Selection / Residual", "—")

        st.caption(
            "Driver attribution will populate automatically once the "
            "strategy-level decomposition stream is reconnected."
        )


# ------------------------------------------------------------
# Helper: Placeholder renderer
# ------------------------------------------------------------
def _render_alpha_placeholders(caption: str | None = None):
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Alpha Quality Score", "—")

    with col2:
        st.metric("Confidence Level", "—")

    if caption:
        st.caption(caption)


# ------------------------------------------------------------
# Adaptive Intelligence — Preview (Read-Only)
# ------------------------------------------------------------
def render_adaptive_intelligence_preview(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    """
    Interpretive preview derived from Alpha Attribution.
    No trading logic. No persistence. No adaptive actions.
    """

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