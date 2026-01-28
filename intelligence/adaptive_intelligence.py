# intelligence/adaptive_intelligence.py

import math
from typing import List

import pandas as pd
import streamlit as st


EM_DASH = "—"


# ============================================================
# Horizon Resolution
# ============================================================

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
        "suffix": None or "30d"/"60d"/"365d"
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


# ============================================================
# Safe Helpers (Governance-Safe)
# ============================================================

def _safe_get_value(row: pd.Series, col: str | None):
    if col is None or col not in row.index:
        return None
    value = row[col]
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _pick_first_existing(row: pd.Series, candidates: List[str | None]):
    for col in candidates:
        if col and col in row.index:
            return col
    return None


# ============================================================
# Rendering Helpers
# ============================================================

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
        "over the selected attribution horizon. This view is designed to "
        "keep the focus on governance and explainability, not prediction."
    )


# ============================================================
# Residual Logic (Never Fabricates)
# ============================================================

def _compute_residual_component(
    row: pd.Series,
    residual_col: str | None,
    alpha_col: str | None,
    other_component_cols: List[str | None],
):
    # Use explicit residual if present
    if residual_col:
        residual_val = _safe_get_value(row, residual_col)
        if residual_val is not None:
            return residual_val

    # Otherwise derive residual = alpha − known components
    if not alpha_col:
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


# ============================================================
# Core Attribution Renderer (FIXED)
# ============================================================

def _render_attribution_components(row: pd.Series, horizon_info: dict):
    suffix = horizon_info.get("suffix")

    # --------------------------------------------
    # Column Resolution (EXPLICIT + FALLBACK)
    # --------------------------------------------
    if suffix is None:
        residual_col = _pick_first_existing(
            row, ["alpha_residual", "alpha_stock_selection"]
        )
        momentum_col = _pick_first_existing(row, ["alpha_momentum"])
        volatility_col = _pick_first_existing(row, ["alpha_volatility"])
        beta_col = _pick_first_existing(row, ["alpha_beta", "alpha_market"])
        allocation_col = _pick_first_existing(row, ["alpha_allocation", "alpha_rotation"])
        alpha_col = _pick_first_existing(row, ["alpha_intraday", "alpha_total"])
    else:
        residual_col = _pick_first_existing(
            row, [f"alpha_residual_{suffix}", "alpha_stock_selection"]
        )
        momentum_col = _pick_first_existing(
            row, [f"alpha_momentum_{suffix}", "alpha_momentum"]
        )
        volatility_col = _pick_first_existing(
            row, [f"alpha_volatility_{suffix}", "alpha_volatility"]
        )
        beta_col = _pick_first_existing(
            row, [f"alpha_beta_{suffix}", "alpha_market"]
        )
        allocation_col = _pick_first_existing(
            row, [f"alpha_allocation_{suffix}", "alpha_rotation"]
        )
        alpha_col = _pick_first_existing(
            row, [f"alpha_{suffix}", "alpha_total"]
        )

    # --------------------------------------------
    # Fetch Values
    # --------------------------------------------
    momentum_val = _safe_get_value(row, momentum_col)
    volatility_val = _safe_get_value(row, volatility_col)
    beta_val = _safe_get_value(row, beta_col)
    allocation_val = _safe_get_value(row, allocation_col)

    residual_val = _compute_residual_component(
        row=row,
        residual_col=residual_col,
        alpha_col=alpha_col,
        other_component_cols=[
            momentum_col,
            volatility_col,
            beta_col,
            allocation_col,
        ],
    )

    # --------------------------------------------
    # Render
    # --------------------------------------------
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


# ============================================================
# Row Selection (FIXED — THIS WAS THE BUG)
# ============================================================

def _select_wave_row(snapshot_df: pd.DataFrame, selected_wave):
    if snapshot_df is None or snapshot_df.empty:
        return None

    if selected_wave is None:
        return snapshot_df.iloc[0]

    # Prefer wave_id match (authoritative)
    if "wave_id" in snapshot_df.columns:
        subset = snapshot_df[snapshot_df["wave_id"] == selected_wave]
        if not subset.empty:
            return subset.iloc[0]

    # Fallback to display_name
    if "display_name" in snapshot_df.columns:
        subset = snapshot_df[snapshot_df["display_name"] == selected_wave]
        if not subset.empty:
            return subset.iloc[0]

    return snapshot_df.iloc[0]


# ============================================================
# Public Entry Point
# ============================================================

def render_alpha_attribution_drivers(snapshot_df, selected_wave, RETURN_COLS, BENCHMARK_COLS):
    horizon_info = _resolve_attribution_horizon()

    st.markdown("## Alpha Attribution")
    st.caption(
        f"Wave: **{selected_wave}** | Horizon: **{horizon_info['label']}**"
        if selected_wave else f"Horizon: **{horizon_info['label']}**"
    )

    row = _select_wave_row(snapshot_df, selected_wave)
    if row is None:
        st.info("No data available for alpha attribution.")
        return

    _render_tactical_timing_explainer()
    _render_attribution_components(row, horizon_info)