"""
Crypto Volatility Overlay Module

This module provides crypto-specific volatility control for portfolio exposure management.
It operates independently from the VIX-based equity overlay system, using BTC/ETH price
data to classify market regimes and compute exposure scaling.

Key Features:
- Regime classification (calm, normal, elevated, stress) based on short/long-term volatility ratios
- Separate handling for crypto growth and income strategies
- Data safety: returns full default exposure when data is insufficient
- Completely isolated from equity wave logic

Regime Classification:
- calm: Low volatility, favorable conditions (ratio < 0.7)
- normal: Moderate volatility, typical market conditions (0.7 <= ratio < 1.3)
- elevated: High volatility, caution warranted (1.3 <= ratio < 2.0)
- stress: Extreme volatility, defensive positioning (ratio >= 2.0)

Exposure Scaling:
- Growth waves: More aggressive scaling (1.0 -> 0.3 in stress)
- Income waves: More conservative scaling (1.0 -> 0.5 in stress)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Configuration constants
CRYPTO_VOL_SHORT_WINDOW = 10  # Short-term volatility window (10 days)
CRYPTO_VOL_LONG_WINDOW = 30   # Long-term volatility window (30 days)
MIN_DATA_POINTS = 40          # Minimum data points required for regime computation

# BTC/ETH tickers for volatility calculation
BTC_TICKERS = ['BTC-USD']
ETH_TICKERS = ['ETH-USD']

# Regime thresholds based on short/long volatility ratio
REGIME_THRESHOLDS = {
    'calm': (0.0, 0.7),        # Ratio < 0.7
    'normal': (0.7, 1.3),      # 0.7 <= ratio < 1.3
    'elevated': (1.3, 2.0),    # 1.3 <= ratio < 2.0
    'stress': (2.0, float('inf'))  # ratio >= 2.0
}

# Exposure scaling by regime for growth waves
GROWTH_EXPOSURE_MAP = {
    'calm': 1.00,      # Full exposure in calm markets
    'normal': 0.85,    # Slight reduction in normal markets
    'elevated': 0.60,  # Moderate reduction in elevated volatility
    'stress': 0.30     # Significant reduction in stress
}

# Exposure scaling by regime for income waves (more conservative)
INCOME_EXPOSURE_MAP = {
    'calm': 1.00,      # Full exposure in calm markets
    'normal': 0.90,    # Minor reduction in normal markets
    'elevated': 0.70,  # Moderate reduction in elevated volatility
    'stress': 0.50     # Conservative reduction in stress
}

# Default exposure when data is insufficient or unavailable
DEFAULT_EXPOSURE = 1.00


def compute_volatility_ratio(
    price_series: pd.Series,
    short_window: int = CRYPTO_VOL_SHORT_WINDOW,
    long_window: int = CRYPTO_VOL_LONG_WINDOW
) -> Optional[float]:
    """
    Compute the ratio of short-term to long-term volatility.
    
    Args:
        price_series: Time series of prices (indexed by date)
        short_window: Window for short-term volatility (default: 10 days)
        long_window: Window for long-term volatility (default: 30 days)
        
    Returns:
        Volatility ratio (short/long), or None if insufficient data
    """
    if price_series is None or len(price_series) < long_window:
        logger.debug(f"Insufficient data for volatility computation: {len(price_series) if price_series is not None else 0} < {long_window}")
        return None
    
    # Compute returns
    returns = price_series.pct_change().dropna()
    
    if len(returns) < long_window:
        logger.debug(f"Insufficient returns for volatility computation: {len(returns)} < {long_window}")
        return None
    
    # Compute short-term volatility (recent window)
    short_vol = returns.iloc[-short_window:].std()
    
    # Compute long-term volatility (full window)
    long_vol = returns.iloc[-long_window:].std()
    
    # Handle edge cases
    if pd.isna(short_vol) or pd.isna(long_vol) or long_vol == 0:
        logger.debug("Invalid volatility values (NaN or zero long_vol)")
        return None
    
    # Compute ratio
    vol_ratio = short_vol / long_vol
    
    logger.debug(f"Volatility ratio: {vol_ratio:.3f} (short={short_vol:.4f}, long={long_vol:.4f})")
    
    return float(vol_ratio)


def classify_regime(vol_ratio: Optional[float]) -> str:
    """
    Classify market regime based on volatility ratio.
    
    Args:
        vol_ratio: Short-term / long-term volatility ratio
        
    Returns:
        Regime classification: 'calm', 'normal', 'elevated', or 'stress'
    """
    if vol_ratio is None or pd.isna(vol_ratio):
        logger.debug("No volatility ratio available, defaulting to 'normal'")
        return 'normal'
    
    for regime, (low, high) in REGIME_THRESHOLDS.items():
        if low <= vol_ratio < high:
            logger.debug(f"Classified regime as '{regime}' (vol_ratio={vol_ratio:.3f})")
            return regime
    
    # Should not reach here, but default to 'normal'
    logger.warning(f"Unexpected volatility ratio {vol_ratio}, defaulting to 'normal'")
    return 'normal'


def compute_exposure_scaling(
    regime: str,
    is_growth: bool = True
) -> float:
    """
    Compute exposure scaling factor based on regime and strategy type.
    
    Args:
        regime: Market regime ('calm', 'normal', 'elevated', 'stress')
        is_growth: True for growth waves, False for income waves
        
    Returns:
        Exposure scaling factor [0.0, 1.0]
    """
    exposure_map = GROWTH_EXPOSURE_MAP if is_growth else INCOME_EXPOSURE_MAP
    
    exposure = exposure_map.get(regime, DEFAULT_EXPOSURE)
    
    logger.debug(f"Exposure for regime '{regime}' ({'growth' if is_growth else 'income'}): {exposure:.2f}")
    
    return exposure


def get_crypto_signal_prices(
    price_book: pd.DataFrame,
    btc_tickers: List[str] = None,
    eth_tickers: List[str] = None
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Extract BTC and ETH price series from price book.
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        btc_tickers: List of BTC ticker symbols to try (default: ['BTC-USD'])
        eth_tickers: List of ETH ticker symbols to try (default: ['ETH-USD'])
        
    Returns:
        Tuple of (btc_series, eth_series), either may be None if not found
    """
    if btc_tickers is None:
        btc_tickers = BTC_TICKERS
    if eth_tickers is None:
        eth_tickers = ETH_TICKERS
    
    btc_series = None
    eth_series = None
    
    # Try to find BTC
    for ticker in btc_tickers:
        if ticker in price_book.columns:
            btc_series = price_book[ticker].dropna()
            logger.debug(f"Found BTC data using ticker {ticker}: {len(btc_series)} points")
            break
    
    # Try to find ETH
    for ticker in eth_tickers:
        if ticker in price_book.columns:
            eth_series = price_book[ticker].dropna()
            logger.debug(f"Found ETH data using ticker {ticker}: {len(eth_series)} points")
            break
    
    return btc_series, eth_series


def compute_crypto_volatility_regime(
    price_book: pd.DataFrame,
    short_window: int = CRYPTO_VOL_SHORT_WINDOW,
    long_window: int = CRYPTO_VOL_LONG_WINDOW,
    min_data_points: int = MIN_DATA_POINTS
) -> Dict[str, Any]:
    """
    Compute crypto market volatility regime and exposure recommendations.
    
    This function analyzes BTC and ETH price data to determine the current
    crypto market regime and compute appropriate exposure scaling factors
    for growth and income strategies.
    
    Data Safety:
    - Returns default exposure (1.0) when data is insufficient
    - Gracefully handles missing tickers
    - Requires minimum data points for reliable regime classification
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        short_window: Window for short-term volatility (default: 10 days)
        long_window: Window for long-term volatility (default: 30 days)
        min_data_points: Minimum data points required (default: 40 days)
        
    Returns:
        Dictionary with:
        - success: bool - Whether computation succeeded
        - available: bool - Whether crypto data is available
        - reason: Optional[str] - Reason for unavailability
        - regime: str - Current regime ('calm', 'normal', 'elevated', 'stress')
        - btc_vol_ratio: Optional[float] - BTC volatility ratio
        - eth_vol_ratio: Optional[float] - ETH volatility ratio
        - combined_vol_ratio: Optional[float] - Combined volatility ratio (average)
        - growth_exposure: float - Recommended exposure for growth waves [0.0, 1.0]
        - income_exposure: float - Recommended exposure for income waves [0.0, 1.0]
        - data_quality: str - Quality assessment ('good', 'partial', 'insufficient')
        - signals_used: List[str] - Which signals were used (BTC, ETH, or both)
    """
    result = {
        'success': False,
        'available': False,
        'reason': None,
        'regime': 'normal',
        'btc_vol_ratio': None,
        'eth_vol_ratio': None,
        'combined_vol_ratio': None,
        'growth_exposure': DEFAULT_EXPOSURE,
        'income_exposure': DEFAULT_EXPOSURE,
        'data_quality': 'insufficient',
        'signals_used': []
    }
    
    # Validate inputs
    if price_book is None or price_book.empty:
        result['reason'] = 'PRICE_BOOK is empty'
        logger.warning("compute_crypto_volatility_regime: PRICE_BOOK is empty")
        return result
    
    # Extract BTC and ETH price series
    btc_series, eth_series = get_crypto_signal_prices(price_book)
    
    # Check if we have any crypto data
    if btc_series is None and eth_series is None:
        result['available'] = False
        result['reason'] = 'No BTC or ETH data available'
        logger.warning("compute_crypto_volatility_regime: No BTC or ETH data found")
        # Return default exposure (data safety)
        result['success'] = True  # Successfully returned safe default
        return result
    
    # Compute volatility ratios
    btc_vol_ratio = None
    eth_vol_ratio = None
    signals_used = []
    
    if btc_series is not None and len(btc_series) >= min_data_points:
        btc_vol_ratio = compute_volatility_ratio(btc_series, short_window, long_window)
        if btc_vol_ratio is not None:
            signals_used.append('BTC')
            result['btc_vol_ratio'] = btc_vol_ratio
    
    if eth_series is not None and len(eth_series) >= min_data_points:
        eth_vol_ratio = compute_volatility_ratio(eth_series, short_window, long_window)
        if eth_vol_ratio is not None:
            signals_used.append('ETH')
            result['eth_vol_ratio'] = eth_vol_ratio
    
    result['signals_used'] = signals_used
    
    # If no valid volatility ratios, return default exposure (data safety)
    if not signals_used:
        result['available'] = False
        result['reason'] = 'Insufficient data for volatility computation'
        result['data_quality'] = 'insufficient'
        logger.info("compute_crypto_volatility_regime: Insufficient data, using default exposure")
        # Return default exposure (data safety)
        result['success'] = True
        return result
    
    # Compute combined volatility ratio (average of available signals)
    vol_ratios = [r for r in [btc_vol_ratio, eth_vol_ratio] if r is not None]
    combined_vol_ratio = np.mean(vol_ratios)
    result['combined_vol_ratio'] = float(combined_vol_ratio)
    
    # Assess data quality
    if len(signals_used) == 2:
        result['data_quality'] = 'good'
    else:
        result['data_quality'] = 'partial'
    
    # Classify regime
    regime = classify_regime(combined_vol_ratio)
    result['regime'] = regime
    
    # Compute exposure scaling
    growth_exposure = compute_exposure_scaling(regime, is_growth=True)
    income_exposure = compute_exposure_scaling(regime, is_growth=False)
    
    result['growth_exposure'] = growth_exposure
    result['income_exposure'] = income_exposure
    
    # Success!
    result['success'] = True
    result['available'] = True
    result['reason'] = None
    
    logger.info(f"Crypto volatility regime computed: regime={regime}, "
               f"growth_exposure={growth_exposure:.2f}, income_exposure={income_exposure:.2f}, "
               f"signals={signals_used}, quality={result['data_quality']}")
    
    return result


def get_crypto_wave_exposure(
    wave_name: str,
    price_book: pd.DataFrame,
    is_growth: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Get exposure recommendation for a specific crypto wave.
    
    This is a convenience function that computes the crypto volatility regime
    and returns the appropriate exposure for the specified wave type.
    
    Args:
        wave_name: Name of the crypto wave
        price_book: PRICE_BOOK DataFrame
        is_growth: True for growth waves, False for income waves
                   If None, will auto-detect from wave_name
        
    Returns:
        Dictionary with:
        - wave_name: str - Name of the wave
        - is_growth: bool - Whether this is a growth wave
        - exposure: float - Recommended exposure [0.0, 1.0]
        - regime: str - Current market regime
        - data_quality: str - Data quality assessment
        - overlay_status: str - Status message
    """
    # Auto-detect wave type if not specified
    if is_growth is None:
        is_growth = "Growth" in wave_name and "Income" not in wave_name
    
    # Compute regime
    regime_result = compute_crypto_volatility_regime(price_book)
    
    # Get appropriate exposure
    exposure = regime_result['growth_exposure'] if is_growth else regime_result['income_exposure']
    
    # Determine overlay status
    if not regime_result['available']:
        overlay_status = f"Default ({regime_result['reason']})"
    elif regime_result['data_quality'] == 'partial':
        overlay_status = f"Active (Partial: {', '.join(regime_result['signals_used'])})"
    else:
        overlay_status = f"Active ({', '.join(regime_result['signals_used'])})"
    
    return {
        'wave_name': wave_name,
        'is_growth': is_growth,
        'exposure': exposure,
        'regime': regime_result['regime'],
        'data_quality': regime_result['data_quality'],
        'overlay_status': overlay_status
    }
