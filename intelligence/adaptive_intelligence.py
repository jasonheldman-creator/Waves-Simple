import pandas as pd
import numpy as np
import streamlit as st


def _safe_progress(value: float) -> float:
    if pd.isna(value):
        return 0.0
    scaled = (value / 0.001) * 0.5 + 0.5
    if scaled < 0.0:
        scaled = 0.0
    if scaled > 1.0:
        scaled = 1.0
    return float(scaled)


def _format_attribution_value(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.4f}"


def _format_timestamp(ts):
    try:
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def _render_attribution_dimension(
    label: str,
    value: float,
    emphasize: bool = False,
) -> None:
    label_col, bar_col, value_col = st.columns([3, 4, 2])

    with label_col:
        if emphasize:
            st.markdown(f"**{label}**")
        else:
            st.markdown(label)

    with bar_col:
        if pd.isna(value):
            st.markdown("—")
        else:
            st.progress(_safe_progress(value))

    with value_col:
        st.markdown(_format_attribution_value(value))


def _render_tactical_timing_dimension() -> None:
    label_col, text_col = st.columns([3, 6])

    with label_col:
        st.markdown("Tactical Timing")

    with text_col:
        st.markdown(
            "Currently expressed within **Stock Selection (Primary Alpha)** "
            "as part of the overall decision signal."
        )


def _render_attribution_block(
    scope_label: str,
    timestamp: pd.Timestamp,
    components: dict,
) -> None:
    st.markdown(f"**Attribution Scope:** {scope_label}")
    st.markdown(f"**As of:** {_format_timestamp(timestamp)}")

    st.markdown("---")

    _render_attribution_dimension(
        "Stock Selection (Primary Alpha)",
        components.get("alpha_residual", np.nan),
        emphasize=True,
    )

    _render_attribution_dimension(
        "Momentum Overlay",
        components.get("alpha_momentum", np.nan),
    )

    _render_attribution_dimension(
        "Volatility / VIX Control",
        components.get("alpha_volatility", np.nan),
    )

    _render_attribution_dimension(
        "Market Exposure Control",
        components.get("alpha_beta", np.nan),
    )

    _render_attribution_dimension(
        "Allocation & Positioning",
        components.get("alpha_allocation", np.nan),
    )

    _render_tactical_timing_dimension()


def _compute_scope_components(df: pd.DataFrame) -> dict:
    cols = [
        "alpha_residual",
        "alpha_momentum",
        "alpha_volatility",
        "alpha_beta",
        "alpha_allocation",
    ]

    components = {}
    for col in cols:
        if col in df.columns:
            components[col] = df[col].sum(min_count=1)
        else:
            components[col] = np.nan

    return components


def render_alpha_attribution_intraday(attribution_df: pd.DataFrame) -> None:
    if attribution_df is None or attribution_df.empty:
        return

    if "timestamp" not in attribution_df.columns:
        return

    st.subheader("Alpha Attribution Drivers (Intraday)")

    latest_timestamp = attribution_df["timestamp"].max()
    latest_slice = attribution_df[attribution_df["timestamp"] == latest_timestamp]

    if latest_slice.empty:
        return

    if "wave_id" in latest_slice.columns:
        wave_ids = (
            latest_slice["wave_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        wave_ids = sorted(wave_ids)
    else:
        wave_ids = []

    scope_options = ["Portfolio (All Waves)"]
    scope_options.extend(wave_ids)

    selected_scope = st.selectbox(
        "Attribution view",
        scope_options,
        index=0,
    )

    if selected_scope == "Portfolio (All Waves)":
        scope_df = latest_slice
        scope_label = "Portfolio (All Waves)"
    else:
        scope_df = latest_slice[
            latest_slice["wave_id"].astype(str) == selected_scope
        ]
        scope_label = f"Wave {selected_scope}"

    components = _compute_scope_components(scope_df)

    _render_attribution_block(
        scope_label=scope_label,
        timestamp=latest_timestamp,
        components=components,
    )


def render_adaptive_intelligence_console(
    attribution_intraday: pd.DataFrame,
) -> None:
    st.header("WAVES Intelligence™")

    render_alpha_attribution_intraday(attribution_intraday)