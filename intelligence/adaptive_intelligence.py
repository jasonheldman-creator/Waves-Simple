# intelligence/adaptive_intelligence.py

import math
from typing import List

import pandas as pd
import streamlit as st


EM_DASH = "—"


def _resolve_attribution_horizon():
    """
    Resolve the alpha attribution horizon from session state.

    Accepts:
      - INTRADAY
      - 1D
      - 30D
      - 60D
      - 365D

    Returns a dict:
      {
        "label": one of "INTRADAY", "30D", "60D", "365D",
        "suffix": None for intraday/1D, or "30d"/"60d"/"365d" for others
      }

    Defaults safely to 30D if missing or invalid.
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

    # Fallback: 30D
    return {"label": "30D", "suffix": "30d"}


def _safe_get_value(row: pd.Series, col: str):
    """
    Safely get a numeric value from a row; return None if missing or non-finite.
    """
    if col not in row.index:
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
    """
    Render a label + progress bar pair, or an em dash if value is missing.

    The numeric value is linearly mapped from [min_val, max_val] to [0, 1]
    for the progress bar, preserving the existing visual style.
    """
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
    """
    Preserve the 'Tactical Timing' explanatory text block.
    """
    st.subheader("Tactical Timing")
    st.caption(
        "Tactical Timing decomposes realized alpha into distinct drivers—"
        "residual selection, momentum, volatility, beta, and allocation—"
        "over the selected attribution horizon. This view is designed to "
        "keep the focus on governance and explainability, not prediction."
    )


def _compute_residual_component(
    row: pd.Series,
    residual_col: str | None,
    alpha_col: str | None,
    other_component_cols: List[str],
):
    """
    Governance-compliant residual logic:

    - Never fabricate values.
    - If residual column exists and is valid, use it.
    - Otherwise, if alpha exists and at least one other component exists,
      compute residual = alpha − sum(known components).
    - If alpha is missing or no other components are known, return None.
    """
    # 1) Use explicit residual column if available
    if residual_col is not None:
        residual_val = _safe_get_value(row, residual_col)
        if residual_val is not None:
            return residual_val

    # 2) Derive residual from alpha minus known components
    if alpha_col is None:
        return None

    alpha_val = _safe_get_value(row, alpha_col)
    if alpha_val is None:
        return None

    known_components_sum = 0.0
    known_count = 0

    for col in other_component_cols:
        if col is None:
            continue
        comp_val = _safe_get_value(row, col)
        if comp_val is not None:
            known_components_sum += comp_val
            known_count += 1

    if known_count == 0:
        # No known components to subtract; cannot infer residual
        return None

    return alpha_val - known_components_sum


def _render_attribution_components(row: pd.Series, horizon_info: dict):
    """
    Render the alpha attribution components for the selected horizon.

    Intraday / 1D:
      - alpha_residual
      - alpha_momentum
      - alpha_volatility
      - alpha_beta
      - alpha_allocation
      - alpha_intraday (for residual derivation when needed)

    30D / 60D / 365D:
      - alpha_residual_{suffix}
      - alpha_momentum_{suffix}
      - alpha_volatility_{suffix}
      - alpha_beta_{suffix}
      - alpha_allocation_{suffix}
      - alpha_{suffix} (for residual derivation when needed)
    """
    suffix = horizon_info.get("suffix", None)

    if suffix is None:
        # Intraday / 1D: unsuffixed components, intraday alpha
        residual_col = "alpha_residual"
        momentum_col = "alpha_momentum"
        volatility_col = "alpha_volatility"
        beta_col = "alpha_beta"
        allocation_col = "alpha_allocation"
        alpha_col = "alpha_intraday"
    else:
        # 30D / 60D / 365D: suffixed columns
        residual_col = f"alpha_residual_{suffix}"
        momentum_col = f"alpha_momentum_{suffix}"
        volatility_col = f"alpha_volatility_{suffix}"
        beta_col = f"alpha_beta_{suffix}"
        allocation_col = f"alpha_allocation_{suffix}"
        alpha_col = f"alpha_{suffix}"

    # Non-residual components (governance: never infer if missing)
    momentum_val = _safe_get_value(row, momentum_col)
    volatility_val = _safe_get_value(row, volatility_col)
    beta_val = _safe_get_value(row, beta_col)
    allocation_val = _safe_get_value(row, allocation_col)

    # Residual: explicit column if present, otherwise alpha − sum(known components)
    residual_val = _compute_residual_component(
        row=row,
        residual_col=residual_col,
        alpha_col=alpha_col,
        other_component_cols=[momentum_col, volatility_col, beta_col, allocation_col],
    )

    st.markdown("### Alpha Attribution Drivers")

    col1, col2 = st.columns(2)

    with col1:
        _render_progress_with_label(
            label="Residual Selection",
            value=residual_val,
            help_text="Idiosyncratic stock selection beyond systematic factors.",
        )
        _render_progress_with_label(
            label="Momentum",
            value=momentum_val,
            help_text="Timing and trend-following effects captured in the portfolio.",
        )
        _render_progress_with_label(
            label="Volatility",
            value=volatility_val,
            help_text="Convexity and volatility harvesting contributions.",
        )

    with col2:
        _render_progress_with_label(
            label="Beta",
            value=beta_val,
            help_text="Exposure to broad market and factor betas.",
        )
        _render_progress_with_label(
            label="Allocation",
            value=allocation_val,
            help_text="Tactical tilts across sleeves, sectors, or themes.",
        )


def _render_header(selected_wave, horizon_info: dict):
    """
    Render the section header for the Alpha Attribution tab.
    """
    st.markdown("## Alpha Attribution")

    caption_parts = []
    if selected_wave is not None:
        caption_parts.append(f"Wave: **{selected_wave}**")

    label = horizon_info.get("label")
    if label:
        caption_parts.append(f"Horizon: **{label}**")

    if caption_parts:
        st.caption(" | ".join(caption_parts))


def _select_wave_row(snapshot_df: pd.DataFrame, selected_wave):
    """
    Safely select the row corresponding to the selected wave.

    FIX:
      - Match on snapshot_df['display_name'] == selected_wave
      - Only fall back to the first row if no match is found.
    """
    if snapshot_df is None or len(snapshot_df) == 0:
        return None

    if selected_wave is None:
        return snapshot_df.iloc[0]

    if "display_name" in snapshot_df.columns:
        subset = snapshot_df[snapshot_df["display_name"] == selected_wave]
        if not subset.empty:
            return subset.iloc[0]

    # Fallback: first row if the specific display_name is not found
    return snapshot_df.iloc[0]


def render_alpha_attribution_drivers(snapshot_df, selected_wave, RETURN_COLS, BENCHMARK_COLS):
    """
    Render the Alpha Attribution tab for the selected wave and horizon.

    Requirements:
      - Reads horizon from st.session_state["alpha_attribution_horizon"], defaulting to 30D.
      - Supports INTRADAY / 1D / 30D / 60D / 365D using existing CSV schema.
      - Intraday / 1D use unsuffixed alpha_* columns and alpha_intraday.
      - 30D / 60D / 365D use suffixed columns (_30d, _60d, _365d).
      - Uses progress bars, labels, and Tactical Timing explanatory text.
      - Missing values render as em dash (—) without fabrication or inference
        beyond governance-compliant residual derivation.
    """
    horizon_info = _resolve_attribution_horizon()
    _render_header(selected_wave, horizon_info)

    row = _select_wave_row(snapshot_df, selected_wave)
    if row is None:
        st.info("No data available for alpha attribution.")
        return

    _render_tactical_timing_explainer()
    _render_attribution_components(row, horizon_info)

    # Optional contextual view using RETURN_COLS / BENCHMARK_COLS
    if isinstance(RETURN_COLS, (list, tuple)) and isinstance(BENCHMARK_COLS, (list, tuple)):
        with st.expander("Context: Returns vs Benchmark (snapshot)", expanded=False):
            context_cols: List[str] = []
            for col in RETURN_COLS:
                if isinstance(col, str) and col in row.index:
                    context_cols.append(col)
            for col in BENCHMARK_COLS:
                if isinstance(col, str) and col in row.index and col not in context_cols:
                    context_cols.append(col)

            if not context_cols:
                st.write(EM_DASH)
            else:
                context_data = {col: row[col] for col in context_cols}
                st.json(context_data)
