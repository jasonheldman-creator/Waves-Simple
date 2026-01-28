# intelligence/adaptive_intelligence.py

import math
from typing import List

import pandas as pd
import streamlit as st


EM_DASH = "—"


# ------------------------------------------------------------
# Horizon resolution
# ------------------------------------------------------------
def _resolve_attribution_horizon():
    """
    Resolve the alpha attribution horizon from session state.

    Accepts:
      - INTRADAY
      - 1D
      - 30D
      - 60D
      - 365D

    Returns:
      {
        "label": one of "INTRADAY", "30D", "60D", "365D",
        "suffix": None for intraday/1D, or "30d"/"60d"/"365d"
      }
    """
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


# ------------------------------------------------------------
# Governance-safe helpers
# ------------------------------------------------------------
def _safe_get_value(row: pd.Series, col: str | None):
    """
    Safely extract a numeric value from a row.
    Column names are expected to be lowercase.
    """
    if col is None or col not in row.index:
        return None

    value = row[col]
    if isinstance(value, (int, float)) and math.isfinite(value):
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
    span = max_val - min_val if max_val != min_val else 1.0
    normalized = (clamped - min_val) / span

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


# ------------------------------------------------------------
# Residual logic (unchanged, governance-safe)
# ------------------------------------------------------------
def _compute_residual_component(
    row: pd.Series,
    residual_col: str | None,
    alpha_col: str | None,
    other_component_cols: List[str],
):
    # 1) Explicit residual
    if residual_col:
        residual_val = _safe_get_value(row, residual_col)
        if residual_val is not None:
            return residual_val

    # 2) Derived residual
    if not alpha_col:
        return None

    alpha_val = _safe_get_value(row, alpha_col)
    if alpha_val is None:
        return None

    known_sum = 0.0
    known_count = 0

    for col in other_component_cols:
        val = _safe_get_value(row, col)
        if val is not None:
            known_sum += val
            known_count += 1

    if known_count == 0:
        return None

    return alpha_val - known_sum


# ------------------------------------------------------------
# Core renderer
# ------------------------------------------------------------
def _render_attribution_components(row: pd.Series, horizon_info: dict):
    """
    IMPORTANT:
    All column names are LOWERCASE to match app_min.py normalization.
    """

    suffix = horizon_info.get("suffix")

    # Canonical (lowercase) column bindings
    momentum_col = "alpha_momentum"
    volatility_col = "alpha_volatility"
    beta_col = "alpha_market"
    allocation_col = "alpha_rotation"
    residual_fallback = "alpha_stock_selection"

    if suffix is None:
        residual_col = residual_fallback
        alpha_col = "alpha_intraday"
    else:
        residual_col = f"alpha_residual_{suffix}"
        alpha_col = f"alpha_{suffix}"

    momentum_val = _safe_get_value(row, momentum_col)
    volatility_val = _safe_get_value(row, volatility_col)
    beta_val = _safe_get_value(row, beta_col)
    allocation_val = _safe_get_value(row, allocation_col)

    residual_val = _compute_residual_component(
        row=row,
        residual_col=residual_col if residual_col in row.index else residual_fallback,
        alpha_col=alpha_col if alpha_col in row.index else "alpha_total",
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


# ------------------------------------------------------------
# Header & row selection
# ------------------------------------------------------------
def _render_header(selected_wave, horizon_info: dict):
    st.markdown("## Alpha Attribution")

    parts = []
    if selected_wave:
        parts.append(f"Wave: **{selected_wave}**")
    if horizon_info.get("label"):
        parts.append(f"Horizon: **{horizon_info['label']}**")

    if parts:
        st.caption(" | ".join(parts))


def _select_wave_row(snapshot_df: pd.DataFrame, selected_wave):
    if snapshot_df is None or snapshot_df.empty:
        return None

    if selected_wave and "display_name" in snapshot_df.columns:
        match = snapshot_df[snapshot_df["display_name"] == selected_wave]
        if not match.empty:
            return match.iloc[0]

    return snapshot_df.iloc[0]


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------
def render_alpha_attribution_drivers(snapshot_df, selected_wave, RETURN_COLS, BENCHMARK_COLS):
    horizon_info = _resolve_attribution_horizon()
    _render_header(selected_wave, horizon_info)

    row = _select_wave_row(snapshot_df, selected_wave)
    if row is None:
        st.info("No data available for alpha attribution.")
        return

    _render_tactical_timing_explainer()
    _render_attribution_components(row, horizon_info)