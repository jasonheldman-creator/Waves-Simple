"""
Crypto Volatility Overlay (Phase 1B.2)

This module implements crypto-specific volatility-regime scaling for portfolio exposure.
It provides deterministic overlay logic based on volatility and drawdown thresholds.

Key Features:
- Volatility-scaling based on 30-day realized volatility
- Peak-to-trough drawdown monitoring over 60 days
- Regime classification: LOW/MED/HIGH/CRISIS
- Exposure scaling from 0.2 (crisis) to 1.0 (low volatility)
- Uses cached price data only (no live fetching)

Usage:
    from helpers.crypto_volatility_overlay import compute_crypto_overlay
    
    # Compute overlay for BTC/ETH benchmarks
    overlay = compute_crypto_overlay(
        benchmarks=['BTC-USD', 'ETH-USD'],
        price_data=price_book,
        vol_window=30,
        dd_window=60
    )
    
    print(f"Regime: {overlay['regime']}, Exposure: {overlay['exposure']}")
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Trading days per year constant for volatility annualization
TRADING_DAYS_PER_YEAR = 252

# Volatility regime thresholds (annualized volatility)
# Based on crypto-native volatility characteristics
VOL_THRESHOLDS = {
    "low": 0.40,      # < 40% annualized = LOW volatility regime
    "medium": 0.80,   # 40-80% = MEDIUM volatility regime  
    "high": 1.20,     # 80-120% = HIGH volatility regime
    # > 120% = implied CRISIS if drawdown also severe
}

# Drawdown thresholds (peak-to-trough)
DD_THRESHOLDS = {
    "minor": -0.15,      # -15% drawdown threshold
    "moderate": -0.30,   # -30% drawdown threshold
    "severe": -0.50,     # -50% drawdown threshold (crisis)
    "critical": -0.60,   # -60% drawdown = hard crisis threshold (dd_crit)
}

# Exposure multipliers for each regime
# Range: 0.2 (minimum crisis exposure, ≤20% val) to 1.0 (full exposure)
EXPOSURE_BY_REGIME = {
    "LOW": 1.00,      # Full exposure in low volatility
    "MED": 0.75,      # 75% exposure in medium volatility
    "HIGH": 0.50,     # 50% exposure in high volatility
    "CRISIS": 0.20,   # Minimum 20% exposure in crisis
}


def _compute_realized_volatility(prices: pd.Series, window: int = 30) -> float:
    """
    Compute annualized realized volatility from price series.
    
    Args:
        prices: Series of prices (typically daily)
        window: Lookback window in days (default: 30)
        
    Returns:
        Annualized realized volatility (float)
    """
    if prices is None or len(prices) < 2:
        return np.nan
    
    # Get recent window
    recent_prices = prices.tail(window + 1)
    
    if len(recent_prices) < 2:
        return np.nan
    
    # Compute daily returns
    returns = recent_prices.pct_change().dropna()
    
    if len(returns) == 0:
        return np.nan
    
    # Annualize volatility using trading days per year constant
    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return float(annualized_vol)


def _compute_max_drawdown(prices: pd.Series, window: int = 60) -> float:
    """
    Compute maximum peak-to-trough drawdown over specified window.
    
    Args:
        prices: Series of prices (typically daily)
        window: Lookback window in days (default: 60)
        
    Returns:
        Maximum drawdown as negative percentage (e.g., -0.30 for -30%)
    """
    if prices is None or len(prices) < 2:
        return 0.0
    
    # Get recent window
    recent_prices = prices.tail(window)
    
    if len(recent_prices) < 2:
        return 0.0
    
    # Compute running maximum (peak)
    running_max = recent_prices.expanding().max()
    
    # Compute drawdown from peak
    drawdown = (recent_prices - running_max) / running_max
    
    # Return the maximum (most negative) drawdown
    max_dd = float(drawdown.min())
    
    return max_dd


def _classify_volatility_regime(vol: float) -> str:
    """
    Classify volatility into regime category.
    
    Args:
        vol: Annualized realized volatility
        
    Returns:
        Regime string: "low", "medium", "high", or "extreme"
    """
    if np.isnan(vol) or vol < 0:
        return "medium"  # Default to medium if invalid
    
    if vol < VOL_THRESHOLDS["low"]:
        return "low"
    elif vol < VOL_THRESHOLDS["medium"]:
        return "medium"
    elif vol < VOL_THRESHOLDS["high"]:
        return "high"
    else:
        return "extreme"


def _classify_drawdown_severity(dd: float) -> str:
    """
    Classify drawdown severity.
    
    Args:
        dd: Maximum drawdown (negative percentage, e.g., -0.30)
        
    Returns:
        Severity string: "none", "minor", "moderate", "severe", or "critical"
    """
    if dd >= DD_THRESHOLDS["minor"]:
        return "none"
    elif dd >= DD_THRESHOLDS["moderate"]:
        return "minor"
    elif dd >= DD_THRESHOLDS["severe"]:
        return "moderate"
    elif dd >= DD_THRESHOLDS["critical"]:
        return "severe"
    else:
        return "critical"


def _determine_regime(vol_regime: str, dd_severity: str) -> str:
    """
    Determine overall regime based on volatility and drawdown.
    
    Crisis regime is triggered by either:
    1. Critical drawdown (hard threshold dd_crit >= -60%)
    2. Severe drawdown + extreme volatility
    
    Args:
        vol_regime: Volatility regime classification
        dd_severity: Drawdown severity classification
        
    Returns:
        Overall regime: "LOW", "MED", "HIGH", or "CRISIS"
    """
    # Critical drawdown triggers immediate crisis
    if dd_severity == "critical":
        return "CRISIS"
    
    # Severe drawdown + extreme volatility = crisis
    if dd_severity == "severe" and vol_regime == "extreme":
        return "CRISIS"
    
    # Moderate/severe drawdown = high risk regime
    if dd_severity in ["severe", "moderate"]:
        return "HIGH"
    
    # Otherwise classify by volatility
    if vol_regime == "low":
        return "LOW"
    elif vol_regime == "medium":
        return "MED"
    else:  # high or extreme volatility
        return "HIGH"


def compute_crypto_overlay(
    benchmarks: List[str],
    price_data: pd.DataFrame,
    vol_window: int = 30,
    dd_window: int = 60
) -> Dict[str, Any]:
    """
    Compute crypto-native volatility-regime overlay for portfolio exposure scaling.
    
    This function analyzes crypto benchmark volatility and drawdowns to determine
    appropriate portfolio exposure levels. It implements deterministic scaling logic
    that caps exposure during high volatility and drawdown periods.
    
    Args:
        benchmarks: List of benchmark tickers (e.g., ['BTC-USD', 'ETH-USD'])
        price_data: DataFrame with prices (index=dates, columns=tickers)
                   Must be cached data (no live fetching)
        vol_window: Volatility lookback window in days (default: 30)
        dd_window: Drawdown lookback window in days (default: 60)
        
    Returns:
        Dictionary containing:
        - overlay_label: str = "Crypto Vol"
        - regime: str = "LOW", "MED", "HIGH", or "CRISIS"
        - exposure: float = exposure multiplier (0.2 to 1.0)
        - volatility: float = annualized realized volatility
        - max_drawdown: float = maximum drawdown over period
        - vol_regime: str = volatility regime classification
        - dd_severity: str = drawdown severity classification
        
    Example:
        >>> from helpers.price_book import get_price_book
        >>> from helpers.crypto_volatility_overlay import compute_crypto_overlay
        >>> 
        >>> price_data = get_price_book(active_tickers=['BTC-USD', 'ETH-USD'])
        >>> overlay = compute_crypto_overlay(
        ...     benchmarks=['BTC-USD', 'ETH-USD'],
        ...     price_data=price_data
        ... )
        >>> print(f"Regime: {overlay['regime']}, Exposure: {overlay['exposure']:.2f}")
    """
    logger.info("=" * 70)
    logger.info("CRYPTO VOLATILITY OVERLAY - Computing regime and exposure")
    logger.info(f"Benchmarks: {benchmarks}")
    logger.info(f"Volatility window: {vol_window}D, Drawdown window: {dd_window}D")
    logger.info("=" * 70)
    
    # Validate inputs
    if not benchmarks:
        logger.warning("No benchmarks provided, using default regime")
        return {
            "overlay_label": "Crypto Vol",
            "regime": "MED",
            "exposure": EXPOSURE_BY_REGIME["MED"],
            "volatility": np.nan,
            "max_drawdown": 0.0,
            "vol_regime": "medium",
            "dd_severity": "none"
        }
    
    if price_data is None or price_data.empty:
        logger.warning("No price data provided, using default regime")
        return {
            "overlay_label": "Crypto Vol",
            "regime": "MED",
            "exposure": EXPOSURE_BY_REGIME["MED"],
            "volatility": np.nan,
            "max_drawdown": 0.0,
            "vol_regime": "medium",
            "dd_severity": "none"
        }
    
    # Aggregate volatility and drawdown across benchmarks
    # Use worst-case (highest vol, deepest drawdown) to be conservative
    max_vol = 0.0
    max_dd = 0.0
    
    for ticker in benchmarks:
        if ticker not in price_data.columns:
            logger.warning(f"Benchmark {ticker} not found in price data, skipping")
            continue
        
        prices = price_data[ticker].dropna()
        
        if len(prices) < 2:
            logger.warning(f"Insufficient price data for {ticker}, skipping")
            continue
        
        # Compute metrics
        vol = _compute_realized_volatility(prices, window=vol_window)
        dd = _compute_max_drawdown(prices, window=dd_window)
        
        logger.info(f"{ticker}: Vol={vol:.2%}, MaxDD={dd:.2%}")
        
        # Track worst-case metrics
        if not np.isnan(vol) and vol > max_vol:
            max_vol = vol
        
        if dd < max_dd:  # More negative = worse
            max_dd = dd
    
    # Classify regimes
    vol_regime = _classify_volatility_regime(max_vol)
    dd_severity = _classify_drawdown_severity(max_dd)
    
    logger.info(f"Volatility regime: {vol_regime} (vol={max_vol:.2%})")
    logger.info(f"Drawdown severity: {dd_severity} (dd={max_dd:.2%})")
    
    # Determine overall regime
    regime = _determine_regime(vol_regime, dd_severity)
    
    # Get exposure for regime
    exposure = EXPOSURE_BY_REGIME[regime]
    
    logger.info(f"Overall regime: {regime}, Exposure: {exposure:.2%}")
    logger.info("=" * 70)
    
    return {
        "overlay_label": "Crypto Vol",
        "regime": regime,
        "exposure": float(exposure),
        "volatility": float(max_vol) if not np.isnan(max_vol) else np.nan,
        "max_drawdown": float(max_dd),
        "vol_regime": vol_regime,
        "dd_severity": dd_severity
    }
