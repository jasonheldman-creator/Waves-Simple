"""
portfolio_construction.py

Causal portfolio construction orchestration layer.

PURPOSE
-------
Make the following strategy components ECONOMICALLY CAUSAL by applying them
BEFORE portfolio returns and NAV are computed:

1) Stock selection
2) Momentum / trend overlays
3) Volatility / VIX regime overlays
4) Beta targeting / exposure scaling
5) Allocation / rotation

This file is a THIN ORCHESTRATION LAYER.
It does NOT introduce new math.
It delegates all logic to existing functions in `waves_engine.py`.

Once wired, portfolio returns will MOVE when:
- momentum flips
- VIX regime changes
- beta caps bind
- allocation rotates
"""

from typing import Any, Callable, Dict, Tuple

import pandas as pd
import numpy as np

import waves_engine


# ---------------------------------------------------------------------
# Type aliases (clarity only)
# ---------------------------------------------------------------------

PricesFrame = pd.DataFrame     # index: dates, columns: assets, values: prices
WeightsFrame = pd.DataFrame    # index: dates, columns: assets, values: weights
ReturnsSeries = pd.Series      # portfolio return series
NAVSeries = pd.Series          # portfolio NAV series


# ---------------------------------------------------------------------
# Core orchestration pipeline (CANONICAL)
# ---------------------------------------------------------------------

def construct_causal_portfolio(
    *,
    prices: PricesFrame,
    base_weights: WeightsFrame,
    config: Dict[str, Any],
    session_state: Dict[str, Any],

    stock_selection_fn: Callable[[WeightsFrame, Dict[str, Any], Dict[str, Any]], WeightsFrame],
    momentum_overlay_fn: Callable[[WeightsFrame, PricesFrame, Dict[str, Any], Dict[str, Any]], WeightsFrame],
    vix_overlay_fn: Callable[[WeightsFrame, Dict[str, Any], Dict[str, Any]], WeightsFrame],
    beta_targeting_fn: Callable[
        [WeightsFrame, PricesFrame, Dict[str, Any], Dict[str, Any]],
        Tuple[WeightsFrame, pd.Series]
    ],
    allocation_rotation_fn: Callable[[WeightsFrame, Dict[str, Any], Dict[str, Any]], WeightsFrame],
    returns_nav_fn: Callable[
        [PricesFrame, WeightsFrame, Dict[str, Any], Dict[str, Any]],
        Tuple[ReturnsSeries, NAVSeries]
    ],
) -> Dict[str, Any]:
    """
    Canonical causal construction pipeline.

    All strategy components are applied at the WEIGHT LEVEL
    BEFORE returns and NAV are computed.
    """

    # ---------------------------------------------------------------
    # 1. STOCK SELECTION (universe, ranking, cutoffs)
    # ---------------------------------------------------------------
    weights_after_selection = stock_selection_fn(
        base_weights,
        config,
        session_state,
    )

    # ---------------------------------------------------------------
    # 2. MOMENTUM / TREND OVERLAY
    # ---------------------------------------------------------------
    weights_after_momentum = momentum_overlay_fn(
        weights_after_selection,
        prices,
        config,
        session_state,
    )

    # ---------------------------------------------------------------
    # 3. VOLATILITY / VIX REGIME OVERLAY
    # ---------------------------------------------------------------
    weights_after_vix = vix_overlay_fn(
        weights_after_momentum,
        config,
        session_state,
    )

    # ---------------------------------------------------------------
    # 4. BETA TARGETING / EXPOSURE SCALING
    # ---------------------------------------------------------------
    weights_after_beta, exposure_series = beta_targeting_fn(
        weights_after_vix,
        prices,
        config,
        session_state,
    )

    # ---------------------------------------------------------------
    # 5. ALLOCATION / ROTATION
    # ---------------------------------------------------------------
    final_weights = allocation_rotation_fn(
        weights_after_beta,
        config,
        session_state,
    )

    # ---------------------------------------------------------------
    # 6. RETURNS + NAV (NOW CAUSAL)
    # ---------------------------------------------------------------
    return_series, nav_series = returns_nav_fn(
        prices,
        final_weights,
        config,
        session_state,
    )

    return {
        "final_weights": final_weights,
        "exposure_series": exposure_series,
        "return_series": return_series,
        "nav_series": nav_series,
    }


# ---------------------------------------------------------------------
# DEFAULT ADAPTERS (THIN, EXPLICIT, NON-FABRICATED)
# ---------------------------------------------------------------------

def _stock_selection_adapter(
    base_weights: WeightsFrame,
    config: Dict[str, Any],
    session_state: Dict[str, Any],
) -> WeightsFrame:
    """
    Adapter to stock selection logic in waves_engine.
    """
    return waves_engine.apply_stock_selection(
        base_weights=base_weights,
        config=config,
        session_state=session_state,
    )


def _momentum_overlay_adapter(
    weights: WeightsFrame,
    prices: PricesFrame,
    config: Dict[str, Any],
    session_state: Dict[str, Any],
) -> WeightsFrame:
    """
    Adapter to momentum / trend overlay logic.
    """
    momentum_signals = waves_engine.compute_momentum_signals(
        prices=prices,
        config=config,
        session_state=session_state,
    )

    return waves_engine.apply_momentum_overlay(
        weights=weights,
        momentum_signals=momentum_signals,
        config=config,
        session_state=session_state,
    )


def _vix_overlay_adapter(
    weights: WeightsFrame,
    config: Dict[str, Any],
    session_state: Dict[str, Any],
) -> WeightsFrame:
    """
    Adapter to VIX / volatility regime overlay logic.
    """
    regime_state = waves_engine.compute_volatility_regime(
        config=config,
        session_state=session_state,
    )

    return waves_engine.apply_vix_overlay(
        weights=weights,
        regime_state=regime_state,
        config=config,
        session_state=session_state,
    )


def _beta_targeting_adapter(
    weights: WeightsFrame,
    prices: PricesFrame,
    config: Dict[str, Any],
    session_state: Dict[str, Any],
) -> Tuple[WeightsFrame, pd.Series]:
    """
    Adapter to beta targeting / exposure scaling logic.
    """
    exposure_series = waves_engine.compute_portfolio_beta(
        weights=weights,
        prices=prices,
        config=config,
        session_state=session_state,
    )

    adjusted_weights = waves_engine.apply_beta_targeting(
        weights=weights,
        exposure_series=exposure_series,
        config=config,
        session_state=session_state,
    )

    return adjusted_weights, exposure_series


def _allocation_rotation_adapter(
    weights: WeightsFrame,
    config: Dict[str, Any],
    session_state: Dict[str, Any],
) -> WeightsFrame:
    """
    Adapter to allocation / rotation logic.
    """
    return waves_engine.apply_allocation_rotation(
        weights=weights,
        config=config,
        session_state=session_state,
    )


def _returns_nav_adapter(
    prices: PricesFrame,
    weights: WeightsFrame,
    config: Dict[str, Any],
    session_state: Dict[str, Any],
) -> Tuple[ReturnsSeries, NAVSeries]:
    """
    Adapter to canonical return + NAV computation.
    """
    return waves_engine.compute_portfolio_returns_nav(
        prices=prices,
        weights=weights,
        config=config,
        session_state=session_state,
    )


# ---------------------------------------------------------------------
# PUBLIC ENTRY POINT (USE THIS)
# ---------------------------------------------------------------------

def construct_causal_portfolio_with_defaults(
    *,
    prices: PricesFrame,
    base_weights: WeightsFrame,
    config: Dict[str, Any],
    session_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Canonical entry point.

    This is the ONLY function that should be used by:
    - snapshot generators
    - portfolio overview calculations
    - institutional reporting

    Once this is wired in, attribution and returns will finally agree.
    """
    return construct_causal_portfolio(
        prices=prices,
        base_weights=base_weights,
        config=config,
        session_state=session_state,
        stock_selection_fn=_stock_selection_adapter,
        momentum_overlay_fn=_momentum_overlay_adapter,
        vix_overlay_fn=_vix_overlay_adapter,
        beta_targeting_fn=_beta_targeting_adapter,
        allocation_rotation_fn=_allocation_rotation_adapter,
        returns_nav_fn=_returns_nav_adapter,
    )