"""
adaptive_intelligence.py

ADAPTIVE INTELLIGENCE CENTER - Read-Only Monitoring and Diagnostics Module

IMPORTANT: This module is MONITORING-ONLY and does NOT modify any trading behavior,
strategies, parameters, weights, execution logic, or TruthFrame data.

This module provides pure functions that analyze wave behavior from the existing
TruthFrame to produce diagnostic signals for human understanding and oversight.

Key Features:
- Read-only access to TruthFrame data (never writes)
- Pure diagnostic functions with no side effects
- Structured data output for UI rendering
- Wave health monitoring (alpha trends, beta drift, exposure analysis)
- Volatility regime intelligence
- Learning signal detection (patterns and anomalies)

All diagnostics pull from TruthFrame and wave registry metadata only.
No legacy portfolio aggregation paths or duplicate strategy calculations.

Usage:
    from adaptive_intelligence import (
        analyze_wave_health,
        analyze_regime_intelligence,
        detect_learning_signals,
        get_wave_health_summary
    )
    
    # Get TruthFrame (read-only)
    truth_df = get_truth_frame(safe_mode=True)
    
    # Analyze wave health for a specific wave
    health = analyze_wave_health(truth_df, "sp500_wave")
    
    # Get portfolio-wide regime intelligence
    regime = analyze_regime_intelligence(truth_df)
    
    # Detect learning signals
    signals = detect_learning_signals(truth_df)
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Alpha thresholds
ALPHA_DECAY_THRESHOLD = -0.01  # Alpha worse than -1% is considered decaying
ALPHA_NEGATIVE_THRESHOLD = -0.02  # Alpha worse than -2% is problematic
ALPHA_POSITIVE_THRESHOLD = 0.02  # Alpha better than +2% is strong
ALPHA_STABILITY_THRESHOLD = 0.005  # Within 0.5% is considered stable

# Beta thresholds
BETA_DRIFT_WARNING_THRESHOLD = 0.15  # Drift > 0.15 is a warning
BETA_DRIFT_MODERATE_THRESHOLD = 0.10  # Drift > 0.10 is moderate concern

# Exposure thresholds
HIGH_EXPOSURE_THRESHOLD = 0.98  # Exposure > 98% is very high
LOW_EXPOSURE_THRESHOLD = 0.50  # Exposure < 50% is low
EXTREME_EXPOSURE_LOW_THRESHOLD = 0.30  # Exposure < 30% is extreme

# Drawdown thresholds
HIGH_DRAWDOWN_THRESHOLD = -0.20  # Drawdown worse than -20% is high

# Health score thresholds
HEALTH_SCORE_HEALTHY_THRESHOLD = 80
HEALTH_SCORE_WATCH_THRESHOLD = 60


# ============================================================================
# WAVE HEALTH MONITORING
# ============================================================================

def analyze_wave_health(truth_df: pd.DataFrame, wave_id: str) -> Dict[str, Any]:
    """
    Analyze health metrics for a specific wave (READ-ONLY).
    
    This function reads TruthFrame data and computes diagnostic metrics
    without modifying any data or trading behavior.
    
    Args:
        truth_df: TruthFrame DataFrame (not modified)
        wave_id: Wave identifier (e.g., "sp500_wave")
        
    Returns:
        Dictionary with health diagnostics:
        {
            'wave_id': str,
            'display_name': str,
            'alpha_1d': float or None,
            'alpha_30d': float or None,
            'alpha_60d': float or None,
            'alpha_direction': str ('improving', 'stable', 'decaying', 'N/A'),
            'beta_target': float or None,
            'beta_real': float or None,
            'beta_drift': float or None,
            'exposure_pct': float or None,
            'volatility_regime': str or None,
            'health_label': str ('healthy', 'watch', 'intervention_candidate', 'N/A'),
            'health_score': int (0-100 or None)
        }
    """
    # Filter to specific wave
    wave_data = truth_df[truth_df['wave_id'] == wave_id]
    
    if wave_data.empty:
        return {
            'wave_id': wave_id,
            'display_name': 'N/A',
            'alpha_1d': None,
            'alpha_30d': None,
            'alpha_60d': None,
            'alpha_direction': 'N/A',
            'beta_target': None,
            'beta_real': None,
            'beta_drift': None,
            'exposure_pct': None,
            'volatility_regime': None,
            'health_label': 'N/A',
            'health_score': None
        }
    
    # Extract wave data (read-only)
    row = wave_data.iloc[0]
    
    # Extract alpha metrics
    alpha_1d = row.get('alpha_1d', None)
    alpha_30d = row.get('alpha_30d', None)
    alpha_60d = row.get('alpha_60d', None)
    
    # Determine alpha direction (improving/stable/decaying)
    alpha_direction = _compute_alpha_direction(alpha_1d, alpha_30d, alpha_60d)
    
    # Extract beta metrics
    beta_target = row.get('beta_target', None)
    beta_real = row.get('beta_real', None)
    beta_drift = row.get('beta_drift', None)
    
    # Extract exposure
    exposure_pct = row.get('exposure_pct', None)
    
    # Extract volatility regime
    volatility_regime = row.get('data_regime_tag', None)
    
    # Compute health label and score
    health_label, health_score = _compute_health_label(
        alpha_direction=alpha_direction,
        beta_drift=beta_drift,
        alpha_30d=alpha_30d,
        exposure_pct=exposure_pct
    )
    
    return {
        'wave_id': wave_id,
        'display_name': row.get('display_name', 'N/A'),
        'alpha_1d': alpha_1d,
        'alpha_30d': alpha_30d,
        'alpha_60d': alpha_60d,
        'alpha_direction': alpha_direction,
        'beta_target': beta_target,
        'beta_real': beta_real,
        'beta_drift': beta_drift,
        'exposure_pct': exposure_pct,
        'volatility_regime': volatility_regime,
        'health_label': health_label,
        'health_score': health_score
    }


def get_wave_health_summary(truth_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Get health summary for all waves in TruthFrame (READ-ONLY).
    
    Args:
        truth_df: TruthFrame DataFrame (not modified)
        
    Returns:
        List of health dictionaries (one per wave)
    """
    wave_ids = truth_df['wave_id'].unique().tolist()
    
    health_summary = []
    for wave_id in wave_ids:
        health = analyze_wave_health(truth_df, wave_id)
        health_summary.append(health)
    
    return health_summary


# ============================================================================
# REGIME INTELLIGENCE
# ============================================================================

def analyze_regime_intelligence(truth_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze volatility regime and wave alignment (READ-ONLY).
    
    This function reads TruthFrame data to determine the current market regime
    and how many waves are aligned vs misaligned.
    
    Args:
        truth_df: TruthFrame DataFrame (not modified)
        
    Returns:
        Dictionary with regime intelligence:
        {
            'current_regime': str ('LIVE', 'SANDBOX', 'HYBRID', 'UNAVAILABLE'),
            'regime_description': str,
            'aligned_waves': int,
            'misaligned_waves': int,
            'total_waves': int,
            'alignment_pct': float,
            'regime_summary': str
        }
    """
    # Determine current regime (most common regime across waves)
    regime_counts = truth_df['data_regime_tag'].value_counts()
    
    if regime_counts.empty:
        current_regime = 'UNAVAILABLE'
    else:
        current_regime = regime_counts.index[0]
    
    # Count aligned vs misaligned waves
    aligned_waves = (truth_df['data_regime_tag'] == current_regime).sum()
    total_waves = len(truth_df)
    misaligned_waves = total_waves - aligned_waves
    alignment_pct = (aligned_waves / total_waves * 100) if total_waves > 0 else 0.0
    
    # Generate regime description
    regime_description = _get_regime_description(current_regime)
    
    # Generate regime summary
    regime_summary = f"{aligned_waves}/{total_waves} waves aligned with {current_regime} regime ({alignment_pct:.1f}%)"
    
    return {
        'current_regime': current_regime,
        'regime_description': regime_description,
        'aligned_waves': aligned_waves,
        'misaligned_waves': misaligned_waves,
        'total_waves': total_waves,
        'alignment_pct': alignment_pct,
        'regime_summary': regime_summary
    }


# ============================================================================
# LEARNING SIGNALS
# ============================================================================

def detect_learning_signals(truth_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect learning signals and patterns across all waves (READ-ONLY).
    
    This function analyzes TruthFrame data to identify patterns that may
    warrant human attention, such as sustained alpha decay, beta drift,
    or benchmark mismatches.
    
    Args:
        truth_df: TruthFrame DataFrame (not modified)
        
    Returns:
        List of signal dictionaries:
        [
            {
                'signal_type': str,
                'wave_id': str,
                'display_name': str,
                'severity': str ('info', 'warning', 'critical'),
                'description': str,
                'metric_value': float or None
            },
            ...
        ]
    """
    signals = []
    
    for _, row in truth_df.iterrows():
        wave_id = row['wave_id']
        display_name = row.get('display_name', 'N/A')
        
        # Signal 1: Sustained alpha decay (30d and 60d both negative)
        alpha_30d = row.get('alpha_30d', None)
        alpha_60d = row.get('alpha_60d', None)
        
        if alpha_30d is not None and alpha_60d is not None:
            if alpha_30d < ALPHA_DECAY_THRESHOLD and alpha_60d < ALPHA_DECAY_THRESHOLD:  # Both worse than -1%
                signals.append({
                    'signal_type': 'sustained_alpha_decay',
                    'wave_id': wave_id,
                    'display_name': display_name,
                    'severity': 'warning',
                    'description': f'Alpha negative in both 30d ({alpha_30d*100:.2f}%) and 60d ({alpha_60d*100:.2f}%)',
                    'metric_value': alpha_60d
                })
        
        # Signal 2: Significant beta drift
        beta_drift = row.get('beta_drift', None)
        
        if beta_drift is not None and beta_drift > BETA_DRIFT_WARNING_THRESHOLD:
            signals.append({
                'signal_type': 'beta_drift',
                'wave_id': wave_id,
                'display_name': display_name,
                'severity': 'warning',
                'description': f'Beta drift of {beta_drift:.3f} exceeds threshold ({BETA_DRIFT_WARNING_THRESHOLD})',
                'metric_value': beta_drift
            })
        
        # Signal 3: Extreme exposure (very high or very low)
        exposure_pct = row.get('exposure_pct', None)
        
        if exposure_pct is not None:
            if exposure_pct > HIGH_EXPOSURE_THRESHOLD:
                signals.append({
                    'signal_type': 'extreme_exposure_high',
                    'wave_id': wave_id,
                    'display_name': display_name,
                    'severity': 'info',
                    'description': f'Very high exposure ({exposure_pct*100:.1f}%) - minimal cash buffer',
                    'metric_value': exposure_pct
                })
            elif exposure_pct < LOW_EXPOSURE_THRESHOLD:
                signals.append({
                    'signal_type': 'extreme_exposure_low',
                    'wave_id': wave_id,
                    'display_name': display_name,
                    'severity': 'info',
                    'description': f'Low exposure ({exposure_pct*100:.1f}%) - high cash allocation',
                    'metric_value': exposure_pct
                })
        
        # Signal 4: Data regime mismatch (UNAVAILABLE or SANDBOX in production)
        data_regime = row.get('data_regime_tag', None)
        
        if data_regime in ['UNAVAILABLE', 'SANDBOX']:
            signals.append({
                'signal_type': 'data_regime_mismatch',
                'wave_id': wave_id,
                'display_name': display_name,
                'severity': 'critical' if data_regime == 'UNAVAILABLE' else 'warning',
                'description': f'Wave operating in {data_regime} regime',
                'metric_value': None
            })
        
        # Signal 5: High drawdown (if available)
        drawdown_60d = row.get('drawdown_60d', None)
        
        if drawdown_60d is not None and drawdown_60d < HIGH_DRAWDOWN_THRESHOLD:  # Drawdown worse than -20%
            signals.append({
                'signal_type': 'high_drawdown',
                'wave_id': wave_id,
                'display_name': display_name,
                'severity': 'warning',
                'description': f'60-day drawdown of {drawdown_60d*100:.2f}% exceeds {HIGH_DRAWDOWN_THRESHOLD*100:.0f}%',
                'metric_value': drawdown_60d
            })
    
    return signals


# ============================================================================
# HELPER FUNCTIONS (PRIVATE)
# ============================================================================

def _compute_alpha_direction(alpha_1d: Optional[float], 
                             alpha_30d: Optional[float], 
                             alpha_60d: Optional[float]) -> str:
    """
    Compute alpha direction trend (improving/stable/decaying).
    
    Logic:
    - Improving: Short-term alpha better than long-term
    - Decaying: Short-term alpha worse than long-term
    - Stable: Similar across timeframes
    - N/A: Insufficient data
    
    Args:
        alpha_1d: 1-day alpha (may be None)
        alpha_30d: 30-day alpha (may be None)
        alpha_60d: 60-day alpha (may be None)
        
    Returns:
        Direction string
    """
    # Need at least 30d and 60d for comparison
    if alpha_30d is None or alpha_60d is None:
        return 'N/A'
    
    # Compare 30d vs 60d
    alpha_diff = alpha_30d - alpha_60d
    
    # Threshold for "stable" (within 0.5%)
    if abs(alpha_diff) < ALPHA_STABILITY_THRESHOLD:
        return 'stable'
    elif alpha_diff > 0:
        return 'improving'
    else:
        return 'decaying'


def _compute_health_label(alpha_direction: str,
                          beta_drift: Optional[float],
                          alpha_30d: Optional[float],
                          exposure_pct: Optional[float]) -> Tuple[str, Optional[int]]:
    """
    Compute health label and score based on multiple factors.
    
    Health Label Logic:
    - healthy: Positive alpha, low beta drift, normal exposure
    - watch: Some concerning metrics but not critical
    - intervention_candidate: Multiple red flags
    - N/A: Insufficient data
    
    Health Score: 0-100 (higher is better), or None if insufficient data
    
    Args:
        alpha_direction: Direction of alpha trend
        beta_drift: Beta drift magnitude
        alpha_30d: 30-day alpha
        exposure_pct: Portfolio exposure percentage
        
    Returns:
        Tuple of (health_label, health_score)
    """
    # Start with base score of 100
    score = 100
    flags = []
    
    # Check alpha direction
    if alpha_direction == 'N/A':
        return ('N/A', None)
    elif alpha_direction == 'decaying':
        score -= 20
        flags.append('alpha_decay')
    elif alpha_direction == 'improving':
        score += 0  # No penalty or bonus for improving
    
    # Check alpha magnitude (30d)
    if alpha_30d is not None:
        if alpha_30d < ALPHA_NEGATIVE_THRESHOLD:  # Worse than -2%
            score -= 25
            flags.append('negative_alpha')
        elif alpha_30d > ALPHA_POSITIVE_THRESHOLD:  # Better than +2%
            score += 0  # No bonus, just neutral
    
    # Check beta drift
    if beta_drift is not None:
        if beta_drift > BETA_DRIFT_WARNING_THRESHOLD:
            score -= 20
            flags.append('high_beta_drift')
        elif beta_drift > BETA_DRIFT_MODERATE_THRESHOLD:
            score -= 10
            flags.append('moderate_beta_drift')
    
    # Check exposure (extreme values may indicate issues)
    if exposure_pct is not None:
        if exposure_pct < EXTREME_EXPOSURE_LOW_THRESHOLD or exposure_pct > HIGH_EXPOSURE_THRESHOLD:
            score -= 10
            flags.append('extreme_exposure')
    
    # Clamp score to 0-100
    score = max(0, min(100, score))
    
    # Determine label based on score and flags
    if score >= HEALTH_SCORE_HEALTHY_THRESHOLD and len(flags) == 0:
        label = 'healthy'
    elif score >= HEALTH_SCORE_WATCH_THRESHOLD and len(flags) <= 1:
        label = 'watch'
    else:
        label = 'intervention_candidate'
    
    return (label, score)


def _get_regime_description(regime: str) -> str:
    """
    Get human-readable description for a volatility regime.
    
    Args:
        regime: Regime tag (e.g., 'LIVE', 'SANDBOX')
        
    Returns:
        Description string
    """
    descriptions = {
        'LIVE': 'Live market data regime - real-time pricing and execution',
        'SANDBOX': 'Sandbox regime - testing environment with simulated data',
        'HYBRID': 'Hybrid regime - mix of live and sandbox data sources',
        'UNAVAILABLE': 'Data unavailable - wave is not currently operational'
    }
    
    return descriptions.get(regime, f'Unknown regime: {regime}')


# ============================================================================
# EXPORT FUNCTIONS FOR UI
# ============================================================================

def get_adaptive_intelligence_snapshot(truth_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get complete adaptive intelligence snapshot (READ-ONLY).
    
    This is a convenience function that bundles all adaptive intelligence
    diagnostics into a single dictionary for easy UI consumption.
    
    Args:
        truth_df: TruthFrame DataFrame (not modified)
        
    Returns:
        Dictionary with all adaptive intelligence data:
        {
            'wave_health': List[Dict],
            'regime_intelligence': Dict,
            'learning_signals': List[Dict],
            'timestamp': datetime
        }
    """
    return {
        'wave_health': get_wave_health_summary(truth_df),
        'regime_intelligence': analyze_regime_intelligence(truth_df),
        'learning_signals': detect_learning_signals(truth_df),
        'timestamp': datetime.now()
    }
