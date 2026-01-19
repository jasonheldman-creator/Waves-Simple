"""
Strategy Overlays Module

This module implements individual strategy overlays for the unified strategy pipeline.
Each overlay modifies returns based on specific signals (momentum, trend, volatility, etc.).

Key Overlays:
- Momentum overlay: Gates returns based on relative strength/momentum detection
- Trend overlay: Risk-on/off switches for uptrend adherence  
- Volatility targeting overlay: Scaling mechanism for risk management
- VIX/SafeSmart overlay: Final risk control layer (already implemented)

All overlays follow a consistent interface:
    apply_X_overlay(returns, prices, **params) -> modified_returns
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def apply_momentum_overlay(
    returns: pd.Series,
    prices: pd.DataFrame,
    tickers: list,
    lookback_days: int = 60,
    threshold: float = 0.0
) -> Tuple[pd.Series, Dict]:
    """
    Apply momentum overlay to gate returns based on relative strength.
    
    This overlay checks if the portfolio has positive momentum over the lookback period.
    If momentum is negative, exposure is reduced (returns are dampened).
    
    Args:
        returns: Daily returns series (index=dates, values=returns)
        prices: Price dataframe (index=dates, columns=tickers)
        tickers: List of tickers in the portfolio
        lookback_days: Number of days for momentum calculation (default: 60)
        threshold: Momentum threshold for full exposure (default: 0.0 = positive momentum)
        
    Returns:
        Tuple of (modified_returns, diagnostics_dict)
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015], index=dates)
        >>> prices = pd.DataFrame({'SPY': [100, 101, 99]}, index=dates)
        >>> modified, diag = apply_momentum_overlay(returns, prices, ['SPY'])
    """
    diagnostics = {
        'overlay_name': 'momentum',
        'applied': False,
        'lookback_days': lookback_days,
        'threshold': threshold,
        'avg_momentum_signal': None,
        'days_gated': 0,
        'avg_exposure_adjustment': 1.0
    }
    
    try:
        # Compute portfolio momentum from prices
        available_tickers = [t for t in tickers if t in prices.columns]
        
        if not available_tickers:
            logger.warning("No available tickers for momentum overlay - returning original returns")
            return returns.copy(), diagnostics
        
        # Compute equal-weight portfolio prices
        portfolio_prices = prices[available_tickers].mean(axis=1)
        
        # Compute momentum: current price vs price N days ago
        momentum_signal = portfolio_prices / portfolio_prices.shift(lookback_days) - 1.0
        
        # Align momentum signal with returns
        momentum_signal = momentum_signal.reindex(returns.index)
        
        # Compute exposure based on momentum
        # If momentum > threshold: exposure = 1.0 (full)
        # If momentum <= threshold: exposure = 0.5 (reduced)
        exposure = pd.Series(1.0, index=returns.index)
        exposure[momentum_signal <= threshold] = 0.5
        
        # Apply exposure adjustment to returns
        modified_returns = returns * exposure
        
        # Diagnostics
        diagnostics['applied'] = True
        diagnostics['avg_momentum_signal'] = float(momentum_signal.mean()) if not momentum_signal.isna().all() else None
        diagnostics['days_gated'] = int((exposure < 1.0).sum())
        diagnostics['avg_exposure_adjustment'] = float(exposure.mean())
        
        logger.debug(f"Momentum overlay: {diagnostics['days_gated']} days gated, avg signal={diagnostics['avg_momentum_signal']:.4f}")
        
        return modified_returns, diagnostics
        
    except Exception as e:
        logger.error(f"Error applying momentum overlay: {e}", exc_info=True)
        return returns.copy(), diagnostics


def apply_trend_overlay(
    returns: pd.Series,
    prices: pd.DataFrame,
    tickers: list,
    short_ma: int = 20,
    long_ma: int = 60,
    risk_off_exposure: float = 0.25
) -> Tuple[pd.Series, Dict]:
    """
    Apply trend/regime filter overlay for risk-on/off switching.
    
    This overlay uses moving average crossover to detect uptrend/downtrend.
    In downtrend (risk-off), exposure is reduced significantly.
    
    Args:
        returns: Daily returns series
        prices: Price dataframe
        tickers: List of tickers in the portfolio
        short_ma: Short moving average period (default: 20 days)
        long_ma: Long moving average period (default: 60 days)
        risk_off_exposure: Exposure in risk-off regime (default: 0.25)
        
    Returns:
        Tuple of (modified_returns, diagnostics_dict)
    """
    diagnostics = {
        'overlay_name': 'trend',
        'applied': False,
        'short_ma': short_ma,
        'long_ma': long_ma,
        'risk_off_exposure': risk_off_exposure,
        'days_risk_off': 0,
        'days_risk_on': 0,
        'avg_exposure_adjustment': 1.0
    }
    
    try:
        # Compute portfolio prices
        available_tickers = [t for t in tickers if t in prices.columns]
        
        if not available_tickers:
            logger.warning("No available tickers for trend overlay - returning original returns")
            return returns.copy(), diagnostics
        
        # Equal-weight portfolio prices
        portfolio_prices = prices[available_tickers].mean(axis=1)
        
        # Compute moving averages
        short_sma = portfolio_prices.rolling(window=short_ma, min_periods=1).mean()
        long_sma = portfolio_prices.rolling(window=long_ma, min_periods=1).mean()
        
        # Align with returns
        short_sma = short_sma.reindex(returns.index)
        long_sma = long_sma.reindex(returns.index)
        
        # Determine regime: risk-on when short_ma > long_ma
        risk_on = short_sma > long_sma
        
        # Set exposure based on regime
        exposure = pd.Series(1.0, index=returns.index)  # Risk-on: full exposure
        exposure[~risk_on] = risk_off_exposure  # Risk-off: reduced exposure
        
        # Apply exposure adjustment
        modified_returns = returns * exposure
        
        # Diagnostics
        diagnostics['applied'] = True
        diagnostics['days_risk_on'] = int(risk_on.sum())
        diagnostics['days_risk_off'] = int((~risk_on).sum())
        diagnostics['avg_exposure_adjustment'] = float(exposure.mean())
        
        logger.debug(f"Trend overlay: {diagnostics['days_risk_on']} risk-on days, {diagnostics['days_risk_off']} risk-off days")
        
        return modified_returns, diagnostics
        
    except Exception as e:
        logger.error(f"Error applying trend overlay: {e}", exc_info=True)
        return returns.copy(), diagnostics


def apply_vol_targeting_overlay(
    returns: pd.Series,
    target_vol: float = 0.15,
    lookback_days: int = 60,
    min_exposure: float = 0.5,
    max_exposure: float = 1.5
) -> Tuple[pd.Series, Dict]:
    """
    Apply volatility targeting overlay as a scaling mechanism.
    
    This overlay adjusts exposure to maintain a target volatility level.
    When realized vol is high, exposure is reduced; when low, exposure can increase.
    
    Args:
        returns: Daily returns series
        target_vol: Target annualized volatility (default: 0.15 = 15%)
        lookback_days: Window for realized volatility calculation (default: 60)
        min_exposure: Minimum exposure cap (default: 0.5)
        max_exposure: Maximum exposure cap (default: 1.5)
        
    Returns:
        Tuple of (modified_returns, diagnostics_dict)
    """
    diagnostics = {
        'overlay_name': 'vol_targeting',
        'applied': False,
        'target_vol': target_vol,
        'lookback_days': lookback_days,
        'avg_realized_vol': None,
        'avg_exposure_adjustment': 1.0
    }
    
    try:
        # Compute rolling realized volatility (annualized)
        realized_vol = returns.rolling(window=lookback_days, min_periods=20).std() * np.sqrt(252)
        
        # Compute exposure: target_vol / realized_vol
        # If realized_vol is high, exposure is reduced
        # If realized_vol is low, exposure is increased
        exposure = target_vol / realized_vol
        
        # Cap exposure to min/max bounds
        exposure = exposure.clip(lower=min_exposure, upper=max_exposure)
        
        # Fill NaN with 1.0 (neutral exposure at start)
        exposure = exposure.fillna(1.0)
        
        # Apply exposure scaling
        modified_returns = returns * exposure
        
        # Diagnostics
        diagnostics['applied'] = True
        diagnostics['avg_realized_vol'] = float(realized_vol.mean()) if not realized_vol.isna().all() else None
        diagnostics['avg_exposure_adjustment'] = float(exposure.mean())
        
        logger.debug(f"Vol targeting overlay: avg realized vol={diagnostics['avg_realized_vol']:.4f}, avg exposure={diagnostics['avg_exposure_adjustment']:.4f}")
        
        return modified_returns, diagnostics
        
    except Exception as e:
        logger.error(f"Error applying vol targeting overlay: {e}", exc_info=True)
        return returns.copy(), diagnostics


def apply_vix_safesmart_overlay(
    returns: pd.Series,
    vix_prices: pd.Series,
    safe_returns: pd.Series,
    vix_low_threshold: float = 18.0,
    vix_high_threshold: float = 25.0,
    exposure_low: float = 1.0,
    exposure_moderate: float = 0.65,
    exposure_high: float = 0.25
) -> Tuple[pd.Series, Dict]:
    """
    Apply VIX/SafeSmart overlay for final risk control.
    
    This overlay adjusts exposure based on VIX regime and blends with safe asset returns.
    This is the final overlay in the stack and provides macro risk control.
    
    Note: This is a simplified interface. The full implementation already exists
    in wave_performance.py compute_portfolio_exposure_series(). This function
    provides a consistent interface for the strategy pipeline.
    
    Args:
        returns: Daily risk asset returns
        vix_prices: VIX price series
        safe_returns: Safe asset returns (e.g., BIL)
        vix_low_threshold: VIX level for low volatility regime (default: 18)
        vix_high_threshold: VIX level for high volatility regime (default: 25)
        exposure_low: Exposure in low vol regime (default: 1.0)
        exposure_moderate: Exposure in moderate vol regime (default: 0.65)
        exposure_high: Exposure in high vol regime (default: 0.25)
        
    Returns:
        Tuple of (modified_returns, diagnostics_dict)
    """
    diagnostics = {
        'overlay_name': 'vix_safesmart',
        'applied': False,
        'vix_low_threshold': vix_low_threshold,
        'vix_high_threshold': vix_high_threshold,
        'avg_vix': None,
        'avg_exposure': 1.0,
        'days_low_vol': 0,
        'days_moderate_vol': 0,
        'days_high_vol': 0
    }
    
    try:
        # Align VIX prices with returns
        vix_aligned = vix_prices.reindex(returns.index).ffill()
        safe_aligned = safe_returns.reindex(returns.index).fillna(0.0)
        
        # Determine exposure based on VIX regime
        exposure = pd.Series(exposure_low, index=returns.index)
        
        # Moderate volatility regime
        moderate_mask = (vix_aligned >= vix_low_threshold) & (vix_aligned < vix_high_threshold)
        exposure[moderate_mask] = exposure_moderate
        
        # High volatility regime
        high_mask = vix_aligned >= vix_high_threshold
        exposure[high_mask] = exposure_high
        
        # Compute blended returns: exposure * risk_returns + (1-exposure) * safe_returns
        modified_returns = exposure * returns + (1 - exposure) * safe_aligned
        
        # Diagnostics
        diagnostics['applied'] = True
        diagnostics['avg_vix'] = float(vix_aligned.mean()) if not vix_aligned.isna().all() else None
        diagnostics['avg_exposure'] = float(exposure.mean())
        diagnostics['days_low_vol'] = int((vix_aligned < vix_low_threshold).sum())
        diagnostics['days_moderate_vol'] = int(moderate_mask.sum())
        diagnostics['days_high_vol'] = int(high_mask.sum())
        
        logger.debug(f"VIX overlay: avg VIX={diagnostics['avg_vix']:.2f}, avg exposure={diagnostics['avg_exposure']:.4f}")
        
        return modified_returns, diagnostics
        
    except Exception as e:
        logger.error(f"Error applying VIX overlay: {e}", exc_info=True)
        return returns.copy(), diagnostics


def apply_strategy_stack(
    base_returns: pd.Series,
    prices: pd.DataFrame,
    tickers: list,
    strategy_stack: list,
    vix_prices: Optional[pd.Series] = None,
    safe_returns: Optional[pd.Series] = None,
    **overlay_params
) -> Tuple[pd.Series, Dict]:
    """
    Apply a stack of strategy overlays in sequence to base returns.
    
    This is the main entry point for the unified strategy pipeline.
    Overlays are applied in the order specified in strategy_stack.
    
    Args:
        base_returns: Base daily returns (selection returns before overlays)
        prices: Price dataframe for the portfolio
        tickers: List of tickers in the portfolio
        strategy_stack: Ordered list of strategy names to apply
        vix_prices: VIX price series (required if 'vix_safesmart' in stack)
        safe_returns: Safe asset returns (required if 'vix_safesmart' in stack)
        **overlay_params: Additional parameters for specific overlays
        
    Returns:
        Tuple of (final_returns, attribution_dict)
        
    Attribution dict structure:
        {
            'base_returns': pd.Series,
            'final_returns': pd.Series,
            'overlays_applied': List[str],
            'overlay_diagnostics': Dict[str, Dict],
            'total_alpha': float,
            'component_alphas': {
                'momentum_alpha': float,
                'trend_alpha': float,
                'vol_target_alpha': float,
                'vix_safesmart_alpha': float
            }
        }
    """
    attribution = {
        'base_returns': base_returns.copy(),
        'final_returns': None,
        'overlays_applied': [],
        'overlay_diagnostics': {},
        'total_alpha': None,
        'component_alphas': {}
    }
    
    # Start with base returns
    current_returns = base_returns.copy()
    previous_returns = base_returns.copy()
    
    # Apply each overlay in sequence
    for strategy_name in strategy_stack:
        try:
            if strategy_name == 'momentum':
                current_returns, diag = apply_momentum_overlay(
                    current_returns, prices, tickers,
                    **overlay_params.get('momentum', {})
                )
                attribution['overlays_applied'].append('momentum')
                attribution['overlay_diagnostics']['momentum'] = diag
                
                # Compute momentum alpha contribution
                momentum_alpha = (current_returns - previous_returns).sum()
                attribution['component_alphas']['momentum_alpha'] = momentum_alpha
                
            elif strategy_name == 'trend':
                current_returns, diag = apply_trend_overlay(
                    current_returns, prices, tickers,
                    **overlay_params.get('trend', {})
                )
                attribution['overlays_applied'].append('trend')
                attribution['overlay_diagnostics']['trend'] = diag
                
                # Compute trend alpha contribution
                trend_alpha = (current_returns - previous_returns).sum()
                attribution['component_alphas']['trend_alpha'] = trend_alpha
                
            elif strategy_name == 'vol_targeting':
                current_returns, diag = apply_vol_targeting_overlay(
                    current_returns,
                    **overlay_params.get('vol_targeting', {})
                )
                attribution['overlays_applied'].append('vol_targeting')
                attribution['overlay_diagnostics']['vol_targeting'] = diag
                
                # Compute vol targeting alpha contribution
                vol_alpha = (current_returns - previous_returns).sum()
                attribution['component_alphas']['vol_target_alpha'] = vol_alpha
                
            elif strategy_name == 'vix_safesmart':
                if vix_prices is None or safe_returns is None:
                    logger.warning("VIX/SafeSmart overlay requires vix_prices and safe_returns - skipping")
                    continue
                    
                current_returns, diag = apply_vix_safesmart_overlay(
                    current_returns, vix_prices, safe_returns,
                    **overlay_params.get('vix_safesmart', {})
                )
                attribution['overlays_applied'].append('vix_safesmart')
                attribution['overlay_diagnostics']['vix_safesmart'] = diag
                
                # Compute VIX overlay alpha contribution
                vix_alpha = (current_returns - previous_returns).sum()
                attribution['component_alphas']['overlay_alpha_vix_safesmart'] = vix_alpha
                
            else:
                logger.warning(f"Unknown strategy: {strategy_name} - skipping")
                continue
            
            # Update previous returns for next iteration
            previous_returns = current_returns.copy()
            
        except Exception as e:
            logger.error(f"Error applying strategy {strategy_name}: {e}", exc_info=True)
            continue
    
    # Final returns and total alpha
    attribution['final_returns'] = current_returns
    attribution['total_alpha'] = (current_returns - base_returns).sum()
    
    # Compute residual alpha
    component_sum = sum(attribution['component_alphas'].values())
    attribution['component_alphas']['residual_alpha'] = attribution['total_alpha'] - component_sum
    
    return current_returns, attribution
