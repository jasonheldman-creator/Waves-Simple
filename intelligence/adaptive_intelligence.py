# intelligence/adaptive_intelligence.py

import math
from typing import List

import pandas as pd
import numpy as np
import streamlit as st


EM_DASH = "—"


def _resolve_attribution_horizon():
    raw = st.session_state.get("alpha_attribution_horizon", "30D")
    horizon = raw.strip().upper() if isinstance(raw, str) else "30D"

    if horizon in ("INTRADAY", "1D"):
        return {"label": "INTRADAY", "suffix": None}
    if horizon == "30D":
        return {"label": "30D", "suffix": "30d"}
    if horizon == "60D":
        return {"label": "60D", "suffix": "60d"}
    if horizon == "365D":
        return {"label": "365D", "suffix": "365d"}

    return {"label": "30D", "suffix": "30d"}


def _safe_get_value(row: pd.Series, col: str):
    """
    Safely get a numeric value from a row.

    FIX:
    Accept numpy numeric types (np.floating, np.integer),
    which is what pandas uses internally.
    """
    if col not in row.index:
        return None

    value = row[col]

    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(value):
        return float(value)

    return None


def _render_progress_with_label(
    label: str,
    value: float | None,
    help_text: str | None = None,
    min_val: float = -1.0,
    max_val: float = 1.0,
):
    st.markdown(f"**{label}**")

    if help_text:
        st.caption(help_text)

    if value is None:
        st.write(EM_DASH)
        return

    clamped = max(min_val, min(max_val, float(value)))
    normalized = (clamped - min_val) / (max_val - min_val)

    progress_col, value_col = st.columns([4, 1])
    with progress_col:
        st.progress(normalized)
    with value_col:
        st.write(f"{value:.3f}")


def _render_tactical_timing_explainer():
    st.subheader("Tactical Timing")
    st.caption(
        "Tactical Timing decomposes realized alpha into distinct drivers—"
        "residual selection, momentum, volatility, beta, and allocation—"
        "over the selected attribution horizon."
    )


def _compute_residual_component(
    row: pd.Series,
    residual_col: str | None,
    alpha_col: str | None,
    other_component_cols: List[str],
):
    if residual_col:
        residual_val = _safe_get_value(row, residual_col)
        if residual_val is not None:
            return residual_val

    if alpha_col is None:
        return None

    alpha_val = _safe_get_value(row, alpha_col)
    if alpha_val is None:
        return None

    total = 0.0
    count = 0
    for col in other_component_cols:
        val = _safe_get_value(row, col)
        if val is not None:
            total += val
            count += 1

    if count == 0:
        return None

    return alpha_val - total


def _render_attribution_components(row: pd.Series, horizon_info: dict):
    suffix = horizon_info.get("suffix")

    if suffix is None:
        residual_col = "alpha_stock_selection"
        beta_col = "alpha_market"
        allocation_col = "alpha_rotation"
        alpha_col = "alpha_intraday"
    else:
        residual_col = "alpha_stock_selection"
        beta_col = "alpha_market"
        allocation_col = "alpha_rotation"
        alpha_col = f"alpha_{suffix}"

    momentum_col = "alpha_momentum"
    volatility_col = "alpha_volatility"

    momentum_val = _safe_get_value(row, momentum_col)
    volatility_val = _safe_get_value(row, volatility_col)
    beta_val = _safe_get_value(row, beta_col)
    allocation_val = _safe_get_value(row, allocation_col)

    residual_val = _compute_residual_component(
        row,
        residual_col=residual_col,
        alpha_col=alpha_col,
        other_component_cols=[
            momentum_col,
            volatility_col,
            beta_col,
            allocation_col,
        ],
    )

    st.markdown("### Alpha Attribution Drivers")

    col1, col2 = st.columns(2)

    with col1:
        _render_progress_with_label(
            "Residual Selection",
            residual_val,
            "Idiosyncratic stock selection beyond systematic factors.",
        )
        _render_progress_with_label(
            "Momentum",
            momentum_val,
            "Timing and trend-following effects captured in the portfolio.",
        )
        _render_progress_with_label(
            "Volatility",
            volatility_val,
            "Convexity and volatility harvesting contributions.",
        )

    with col2:
        _render_progress_with_label(
            "Beta",
            beta_val,
            "Exposure to broad market and factor betas.",
        )
        _render_progress_with_label(
            "Allocation",
            allocation_val,
            "Tactical tilts across sleeves, sectors, or themes.",
        )


def render_alpha_attribution_drivers(snapshot_df, selected_wave, RETURN_COLS, BENCHMARK_COLS):
    horizon_info = _resolve_attribution_horizon()

    st.markdown("## Alpha Attribution")
    if selected_wave:
        st.caption(f"Wave: **{selected_wave}** | Horizon: **{horizon_info['label']}**")

    row = snapshot_df[snapshot_df["display_name"] == selected_wave].iloc[0]
    _render_tactical_timing_explainer()
    _render_attribution_components(row, horizon_info)