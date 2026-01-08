"""
Crypto Overlay Diagnostics Helper

Provides functions to extract and display crypto-specific overlay diagnostics
for crypto waves in the UI.
"""

import pandas as pd
from typing import Dict, Optional, Any


def get_crypto_overlay_diagnostics(wave_name: str, wave_history: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Extract crypto overlay diagnostics from wave history DataFrame.
    
    Args:
        wave_name: Name of the wave
        wave_history: DataFrame from compute_history_nav with include_diagnostics=True
        
    Returns:
        Dictionary with crypto overlay fields or None if not a crypto wave
    """
    # Import here to avoid circular dependency
    try:
        import waves_engine as we
    except ImportError:
        return None
    
    # Check if this is a crypto wave
    if not we._is_crypto_wave(wave_name):
        return None
    
    # Check if wave_history has data and crypto fields
    if wave_history is None or wave_history.empty:
        return {
            'is_crypto': True,
            'overlay_active': False,
            'reason': 'no_data_available'
        }
    
    # Get latest row for current state
    latest = wave_history.iloc[-1] if len(wave_history) > 0 else None
    
    if latest is None:
        return {
            'is_crypto': True,
            'overlay_active': False,
            'reason': 'no_latest_data'
        }
    
    # Extract crypto overlay fields
    diagnostics = {
        'is_crypto': True,
        'is_crypto_growth': we._is_crypto_growth_wave(wave_name),
        'is_crypto_income': we._is_crypto_income_wave(wave_name),
        'overlay_active': False,
    }
    
    # Check for crypto-specific diagnostic columns
    crypto_columns = [
        'crypto_trend_regime',
        'crypto_vol_state',
        'crypto_liq_state'
    ]
    
    has_crypto_fields = any(col in wave_history.columns for col in crypto_columns)
    
    if has_crypto_fields:
        diagnostics['overlay_active'] = True
        
        # Extract regime information
        if 'crypto_trend_regime' in wave_history.columns:
            diagnostics['trend_regime'] = latest.get('crypto_trend_regime', 'unknown')
        
        # Extract volatility state
        if 'crypto_vol_state' in wave_history.columns:
            diagnostics['volatility_state'] = latest.get('crypto_vol_state', 'unknown')
        
        # Extract liquidity state
        if 'crypto_liq_state' in wave_history.columns:
            diagnostics['liquidity_state'] = latest.get('crypto_liq_state', 'unknown')
        
        # Extract exposure if available
        if 'exposure' in wave_history.columns:
            diagnostics['exposure'] = float(latest.get('exposure', 1.0))
        
        # Extract realized volatility if available
        if 'crypto_realized_vol' in wave_history.columns or 'realized_vol' in wave_history.columns:
            vol_col = 'crypto_realized_vol' if 'crypto_realized_vol' in wave_history.columns else 'realized_vol'
            diagnostics['realized_volatility'] = float(latest.get(vol_col, 0.0))
        
        # Extract max drawdown if available
        if 'max_drawdown' in wave_history.columns:
            diagnostics['max_drawdown'] = float(latest.get('max_drawdown', 0.0))
    else:
        diagnostics['overlay_active'] = False
        diagnostics['reason'] = 'no_crypto_diagnostic_fields'
    
    return diagnostics


def format_crypto_overlay_status(diagnostics: Optional[Dict[str, Any]]) -> str:
    """
    Format crypto overlay status for display in UI.
    
    Args:
        diagnostics: Crypto overlay diagnostics from get_crypto_overlay_diagnostics
        
    Returns:
        Formatted status string
    """
    if diagnostics is None:
        return "Not a crypto wave"
    
    if not diagnostics.get('overlay_active', False):
        reason = diagnostics.get('reason', 'unknown')
        return f"Overlay inactive ({reason})"
    
    return "✓ Overlay active"


def format_crypto_regime(diagnostics: Optional[Dict[str, Any]]) -> str:
    """
    Format crypto regime for display.
    
    Args:
        diagnostics: Crypto overlay diagnostics
        
    Returns:
        Formatted regime string with emoji
    """
    if diagnostics is None or not diagnostics.get('overlay_active', False):
        return "N/A"
    
    regime = diagnostics.get('trend_regime', 'unknown')
    
    # Map regime to display format with emoji
    regime_map = {
        'strong_uptrend': '📈 Strong Uptrend',
        'uptrend': '↗️ Uptrend',
        'neutral': '➖ Neutral',
        'downtrend': '↘️ Downtrend',
        'strong_downtrend': '📉 Strong Downtrend',
    }
    
    return regime_map.get(regime, f"⚠️ {regime.title()}")


def format_crypto_exposure(diagnostics: Optional[Dict[str, Any]]) -> str:
    """
    Format crypto exposure for display.
    
    Args:
        diagnostics: Crypto overlay diagnostics
        
    Returns:
        Formatted exposure string
    """
    if diagnostics is None or not diagnostics.get('overlay_active', False):
        return "N/A"
    
    exposure = diagnostics.get('exposure')
    
    if exposure is None:
        return "N/A"
    
    # Format as percentage
    return f"{exposure * 100:.0f}%"


def format_crypto_volatility(diagnostics: Optional[Dict[str, Any]]) -> str:
    """
    Format crypto volatility state for display.
    
    Args:
        diagnostics: Crypto overlay diagnostics
        
    Returns:
        Formatted volatility string
    """
    if diagnostics is None or not diagnostics.get('overlay_active', False):
        return "N/A"
    
    vol_state = diagnostics.get('volatility_state', 'unknown')
    realized_vol = diagnostics.get('realized_volatility')
    
    # Map volatility state to display format
    vol_map = {
        'extreme_compression': '🟢 Extremely Low',
        'compression': '🟢 Low',
        'normal': '🟡 Normal',
        'expansion': '🟠 High',
        'extreme_expansion': '🔴 Extremely High',
    }
    
    display = vol_map.get(vol_state, f"⚠️ {vol_state.title()}")
    
    # Add realized vol if available
    if realized_vol is not None and realized_vol > 0:
        display += f" ({realized_vol * 100:.1f}%)"
    
    return display


def get_crypto_overlay_minimum_exposure(wave_name: str) -> float:
    """
    Get minimum exposure floor for crypto wave.
    
    Args:
        wave_name: Name of the wave
        
    Returns:
        Minimum exposure (0.20 for growth, 0.40 for income)
    """
    try:
        import waves_engine as we
        
        if we._is_crypto_income_wave(wave_name):
            return 0.40
        elif we._is_crypto_growth_wave(wave_name):
            return 0.20
        else:
            return 0.0
    except ImportError:
        return 0.0
