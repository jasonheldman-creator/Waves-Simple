# intelligence/adaptive_intelligence.py

import math
from typing import List, Optional

import pandas as pd
import streamlit as st


EM_DASH = "—"


# ------------------------------------------------------------------
# Horizon resolution
# ------------------------------------------------------------------
def _resolve_attribution_horizon():
    """
    Resolve the alpha attribution horizon from session state.

    Accepts:
      - INTRADAY
      - 1D
      - 30D
      - 60D
      - 365D
    """
    raw = st.session_state.get("alpha_attribution_horizon", "30D")

    if isinstance(raw, str):
        horizon = raw.strip().upper()
    else:
        horizon = "30D"

    if horizon in ("INTRADAY", "1D"):
        return {"label": "INTRADAY", "suffix": None}

    if horizon == "30D":
        return {"label": "30D", "suffix": "30d"}

    if horizon == "60D":
        return {"label": "60D", "suffix": "60d"}

    if horizon == "365D":
        return {"label": "365D", "suffix": "365d"}

    return {"label": "30D", "suffix": "30d"}


# ------------------------------------------------------------------
# Safe value access
# ------------------------------------------------------------------
def _safe_get_value(row: pd.Series, col: Optional[str]) -> Optional[float]:
    if col is None or col not in row.index:
        return None

    value = row[col]
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)

    return None


def _pick_first_existing(row: pd.Series, candidates: List[str]) -> Optional[str]:
    """
    Governance-safe binding helper.
    Returns the first column that exists in the row.
    """
    for col in candidates:
        if col in row.index:
            return col
    return None


# ------------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------------
def _render_progress_with_label(
    label: str,
    value: Optional[float],
    help_text: Optional[str] = None,
    min_val: float = -1.0,
    max_val: float = 1.0,
):
    st.markdown(f"**{label}**")

    if help_text:
        st.caption(help_text)

    if value is None:
        st.write(EM_DASH)
        return

    clamped = max(min_val, min(max_val, value))
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
        "Tactical Timing decomposes realized alpha into residual selection, "
        "momentum, volatility, beta, and allocation over the selected horizon. "
        "This view is governance-first and purely explanatory."
    )


# ------------------------------------------------------------------
# Residual logic
# ------------------------------------------------------------------
def _compute_residual_component(
    row: pd.Series,
    residual_col: Optional[str],
    alpha_col: Optional[str],
    other_component_cols: List[Optional[str]],
):
    # 1) Explicit residual if present
    residual_val = _safe_get_value(row, residual_col)
    if residual_val is not None:
        return residual_val

    # 2) Derive from alpha − known components
    alpha_val = _safe_get_value(row, alpha_col)
    if alpha_val is None:
        return None

    total_known = 0.0
    known_count = 0

    for col in other_component_cols:
        val = _safe_get_value(row, col)
        if val is not None:
            total_known += val
            known_count += 1

    if known_count == 0:
        return None

    return alpha_val - total_known


# ------------------------------------------------------------------
# CORE FIX: Attribution component resolution
# ------------------------------------------------------------------
def _render_attribution_components(row: pd.Series, horizon_info: dict):
    """
    Correct, deterministic attribution binding.

    RULE (CRITICAL):
    If a horizon suffix exists, ALWAYS prefer suffixed columns first.
    """

    suffix = horizon_info.get("suffix")

    if suffix is None:
        # Intraday / 1D
        residual_col = _pick_first_existing(row, ["alpha_residual"])
        momentum_col = _pick_first_existing(row, ["alpha_momentum"])
        volatility_col = _pick_first_existing(row, ["alpha_volatility"])
        beta_col = _pick_first_existing(row, ["alpha_beta", "alpha_market"])
        allocation_col = _pick_first_existing(row, ["alpha_allocation", "alpha_rotation"])
        alpha_col = _pick_first_existing(row, ["alpha_intraday", "alpha_1d", "alpha_total"])
    else:
        # 30D / 60D / 365D — ALWAYS prefer suffixed
        residual_col = _pick_first_existing(
            row,
            [f"alpha_residual_{suffix}", "alpha_stock_selection"],
        )
        momentum_col = _pick_first_existing(
            row,
            [f"alpha_momentum_{suffix}", "alpha_momentum"],
        )
        volatility_col = _pick_first_existing(
            row,
            [f"alpha_volatility_{suffix}", "alpha_volatility"],
        )
        beta_col = _pick_first_existing(
            row,
            [f"alpha_beta_{suffix}", "alpha_market"],
        )
        allocation_col = _pick_first_existing(
            row,
            [f"alpha_allocation_{suffix}", "alpha_rotation"],
        )
        alpha_col = _pick_first_existing(
            row,
            [f"alpha_{suffix}", "alpha_total"],
        )

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
            "Trend-following and timing contribution.",
        )
        _render_progress_with_label(
            "Volatility",
            volatility_val,
            "Convexity and volatility harvesting contribution.",
        )

    with col2:
        _render_progress_with_label(
            "Beta",
            beta_val,
            "Market and factor exposure.",
        )
        _render_progress_with_label(
            "Allocation",
            allocation_val,
            "Tactical allocation and rotation effects.",
        )


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def render_alpha_attribution_drivers(
    snapshot_df: pd.DataFrame,
    selected_wave,
    RETURN_COLS,
    BENCHMARK_COLS,
):
    horizon_info = _resolve_attribution_horizon()

    st.markdown("## Alpha Attribution")

    caption = []
    if selected_wave:
        caption.append(f"Wave: **{selected_wave}**")
    caption.append(f"Horizon: **{horizon_info['label']}**")
    st.caption(" | ".join(caption))

    if snapshot_df is None or snapshot_df.empty:
        st.info("No data available for alpha attribution.")
        return

    if selected_wave is not None and "display_name" in snapshot_df.columns:
        subset = snapshot_df[snapshot_df["display_name"] == selected_wave]
        row = subset.iloc[0] if not subset.empty else snapshot_df.iloc[0]
    else:
        row = snapshot_df.iloc[0]

    _render_tactical_timing_explainer()
    _render_attribution_components(row, horizon_info)