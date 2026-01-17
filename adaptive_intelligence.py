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

# Severity thresholds (Stage 2)
SEVERITY_LOW_THRESHOLD = 25
SEVERITY_MEDIUM_THRESHOLD = 50
SEVERITY_HIGH_THRESHOLD = 75
# Anything >= 75 is Critical

# Regime volatility multipliers (Stage 2)
REGIME_MULTIPLIER_NORMAL = 1.0  # LIVE regime
REGIME_MULTIPLIER_VOLATILE = 1.3  # HYBRID regime
REGIME_MULTIPLIER_RISK_OFF = 1.5  # SANDBOX or UNAVAILABLE regime


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
    
    Stage 2 Enhancement: Each signal now includes:
    - severity_score (0-100, deterministic)
    - severity_label ('Low', 'Medium', 'High', 'Critical')
    - confidence_score (0-100)
    - action_classification ('Info', 'Watch', 'Intervention')
    
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
                'severity': str ('info', 'warning', 'critical'),  # Legacy field
                'severity_score': int (0-100),
                'severity_label': str ('Low', 'Medium', 'High', 'Critical'),
                'confidence_score': int (0-100),
                'action_classification': str ('Info', 'Watch', 'Intervention'),
                'description': str,
                'metric_value': float or None
            },
            ...
        ]
    """
    signals = []
    
    # Get total portfolio weight for normalization
    total_waves = len(truth_df)
    default_weight = 1.0 / total_waves if total_waves > 0 else 0.33
    
    for _, row in truth_df.iterrows():
        wave_id = row['wave_id']
        display_name = row.get('display_name', 'N/A')
        data_regime = row.get('data_regime_tag', 'LIVE')
        
        # Estimate wave weight (use exposure as proxy if available)
        wave_weight = row.get('exposure_pct', default_weight)
        if wave_weight is None or wave_weight <= 0:
            wave_weight = default_weight
        
        # Signal 1: Sustained alpha decay (30d and 60d both negative)
        alpha_30d = row.get('alpha_30d', None)
        alpha_60d = row.get('alpha_60d', None)
        alpha_1d = row.get('alpha_1d', None)
        
        if alpha_30d is not None and alpha_60d is not None:
            if alpha_30d < ALPHA_DECAY_THRESHOLD and alpha_60d < ALPHA_DECAY_THRESHOLD:  # Both worse than -1%
                # Calculate magnitude: how negative is the alpha?
                magnitude = min(1.0, abs(alpha_60d) / 0.05)  # Normalize to 5% as extreme
                
                # Calculate persistence: both 30d and 60d negative indicates persistence
                persistence = 0.8  # High persistence since both periods are negative
                
                # Calculate data coverage
                data_coverage = 1.0 if alpha_1d is not None else 0.7
                
                # Calculate metric agreement (both 30d and 60d agree on negativity)
                metric_agreement = 1.0
                
                # Calculate recency (30d is more recent)
                recency = 0.9
                
                # Compute scores
                severity_score = _compute_severity_score(magnitude, persistence, data_regime, wave_weight)
                confidence_score = _compute_confidence_score(data_coverage, metric_agreement, recency)
                severity_label = _get_severity_label(severity_score)
                action_classification = _get_action_classification(severity_label)
                
                signals.append({
                    'signal_type': 'sustained_alpha_decay',
                    'wave_id': wave_id,
                    'display_name': display_name,
                    'severity': 'warning',  # Legacy field
                    'severity_score': severity_score,
                    'severity_label': severity_label,
                    'confidence_score': confidence_score,
                    'action_classification': action_classification,
                    'description': f'Alpha negative in both 30d ({alpha_30d*100:.2f}%) and 60d ({alpha_60d*100:.2f}%)',
                    'metric_value': alpha_60d
                })
        
        # Signal 2: Significant beta drift
        beta_drift = row.get('beta_drift', None)
        beta_target = row.get('beta_target', None)
        
        if beta_drift is not None and beta_drift > BETA_DRIFT_WARNING_THRESHOLD:
            # Calculate magnitude: how large is the drift?
            magnitude = min(1.0, beta_drift / 0.30)  # Normalize to 0.30 as extreme
            
            # Persistence: can't easily determine from single point, use moderate
            persistence = 0.5
            
            # Data coverage
            data_coverage = 1.0 if beta_target is not None else 0.6
            
            # Metric agreement: moderate (single metric)
            metric_agreement = 0.7
            
            # Recency
            recency = 0.8
            
            # Compute scores
            severity_score = _compute_severity_score(magnitude, persistence, data_regime, wave_weight)
            confidence_score = _compute_confidence_score(data_coverage, metric_agreement, recency)
            severity_label = _get_severity_label(severity_score)
            action_classification = _get_action_classification(severity_label)
            
            signals.append({
                'signal_type': 'beta_drift',
                'wave_id': wave_id,
                'display_name': display_name,
                'severity': 'warning',  # Legacy field
                'severity_score': severity_score,
                'severity_label': severity_label,
                'confidence_score': confidence_score,
                'action_classification': action_classification,
                'description': f'Beta drift of {beta_drift:.3f} exceeds threshold ({BETA_DRIFT_WARNING_THRESHOLD})',
                'metric_value': beta_drift
            })
        
        # Signal 3: Extreme exposure (very high or very low)
        exposure_pct = row.get('exposure_pct', None)
        
        if exposure_pct is not None:
            if exposure_pct > HIGH_EXPOSURE_THRESHOLD:
                # High exposure is informational, low severity
                magnitude = min(1.0, (exposure_pct - HIGH_EXPOSURE_THRESHOLD) / 0.02)
                persistence = 0.3
                data_coverage = 1.0
                metric_agreement = 0.9
                recency = 0.9
                
                severity_score = _compute_severity_score(magnitude, persistence, data_regime, wave_weight)
                confidence_score = _compute_confidence_score(data_coverage, metric_agreement, recency)
                severity_label = _get_severity_label(severity_score)
                action_classification = _get_action_classification(severity_label)
                
                signals.append({
                    'signal_type': 'extreme_exposure_high',
                    'wave_id': wave_id,
                    'display_name': display_name,
                    'severity': 'info',  # Legacy field
                    'severity_score': severity_score,
                    'severity_label': severity_label,
                    'confidence_score': confidence_score,
                    'action_classification': action_classification,
                    'description': f'Very high exposure ({exposure_pct*100:.1f}%) - minimal cash buffer',
                    'metric_value': exposure_pct
                })
            elif exposure_pct < LOW_EXPOSURE_THRESHOLD:
                # Low exposure may indicate issues
                magnitude = min(1.0, (LOW_EXPOSURE_THRESHOLD - exposure_pct) / 0.30)
                persistence = 0.4
                data_coverage = 1.0
                metric_agreement = 0.9
                recency = 0.9
                
                severity_score = _compute_severity_score(magnitude, persistence, data_regime, wave_weight)
                confidence_score = _compute_confidence_score(data_coverage, metric_agreement, recency)
                severity_label = _get_severity_label(severity_score)
                action_classification = _get_action_classification(severity_label)
                
                signals.append({
                    'signal_type': 'extreme_exposure_low',
                    'wave_id': wave_id,
                    'display_name': display_name,
                    'severity': 'info',  # Legacy field
                    'severity_score': severity_score,
                    'severity_label': severity_label,
                    'confidence_score': confidence_score,
                    'action_classification': action_classification,
                    'description': f'Low exposure ({exposure_pct*100:.1f}%) - high cash allocation',
                    'metric_value': exposure_pct
                })
        
        # Signal 4: Data regime mismatch (UNAVAILABLE or SANDBOX in production)
        if data_regime in ['UNAVAILABLE', 'SANDBOX']:
            # Regime mismatch is critical
            magnitude = 1.0 if data_regime == 'UNAVAILABLE' else 0.7
            persistence = 0.9  # Assume persistent until resolved
            data_coverage = 1.0
            metric_agreement = 1.0
            recency = 1.0
            
            severity_score = _compute_severity_score(magnitude, persistence, data_regime, wave_weight)
            confidence_score = _compute_confidence_score(data_coverage, metric_agreement, recency)
            severity_label = _get_severity_label(severity_score)
            action_classification = _get_action_classification(severity_label)
            
            signals.append({
                'signal_type': 'data_regime_mismatch',
                'wave_id': wave_id,
                'display_name': display_name,
                'severity': 'critical' if data_regime == 'UNAVAILABLE' else 'warning',  # Legacy field
                'severity_score': severity_score,
                'severity_label': severity_label,
                'confidence_score': confidence_score,
                'action_classification': action_classification,
                'description': f'Wave operating in {data_regime} regime',
                'metric_value': None
            })
        
        # Signal 5: High drawdown (if available)
        drawdown_60d = row.get('drawdown_60d', None)
        
        if drawdown_60d is not None and drawdown_60d < HIGH_DRAWDOWN_THRESHOLD:  # Drawdown worse than -20%
            # Calculate magnitude
            magnitude = min(1.0, abs(drawdown_60d) / 0.40)  # Normalize to -40% as extreme
            
            # Persistence: 60d drawdown indicates sustained issue
            persistence = 0.8
            
            # Data coverage
            data_coverage = 0.9
            
            # Metric agreement
            metric_agreement = 0.8
            
            # Recency
            recency = 0.7
            
            severity_score = _compute_severity_score(magnitude, persistence, data_regime, wave_weight)
            confidence_score = _compute_confidence_score(data_coverage, metric_agreement, recency)
            severity_label = _get_severity_label(severity_score)
            action_classification = _get_action_classification(severity_label)
            
            signals.append({
                'signal_type': 'high_drawdown',
                'wave_id': wave_id,
                'display_name': display_name,
                'severity': 'warning',  # Legacy field
                'severity_score': severity_score,
                'severity_label': severity_label,
                'confidence_score': confidence_score,
                'action_classification': action_classification,
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


def _get_regime_multiplier(regime: str) -> float:
    """
    Get volatility multiplier for regime-aware severity calculation (Stage 2).
    
    Args:
        regime: Regime tag (e.g., 'LIVE', 'SANDBOX')
        
    Returns:
        Multiplier value (1.0 for normal, up to 1.5 for risk-off)
    """
    multipliers = {
        'LIVE': REGIME_MULTIPLIER_NORMAL,
        'HYBRID': REGIME_MULTIPLIER_VOLATILE,
        'SANDBOX': REGIME_MULTIPLIER_RISK_OFF,
        'UNAVAILABLE': REGIME_MULTIPLIER_RISK_OFF
    }
    
    return multipliers.get(regime, REGIME_MULTIPLIER_NORMAL)


def _compute_severity_score(
    magnitude: float,
    persistence: float,
    regime: str,
    wave_weight: float
) -> int:
    """
    Compute deterministic severity score (0-100) for a signal (Stage 2).
    
    Severity is based on:
    - Magnitude of the issue (0-40 points)
    - Persistence of the issue (0-30 points)
    - Regime-aware multiplier (1.0-1.5x based on market volatility)
    - Wave role/importance (0-30 points based on weight)
    
    Args:
        magnitude: Magnitude of the issue (0.0-1.0, higher is worse)
        persistence: How long the issue has persisted (0.0-1.0, higher is worse)
        regime: Market regime ('LIVE', 'SANDBOX', etc.)
        wave_weight: Weight of the wave in portfolio (0.0-1.0)
        
    Returns:
        Severity score (0-100, capped at 100)
    """
    # Magnitude contribution (0-40 points)
    magnitude_score = min(40, magnitude * 40)
    
    # Persistence contribution (0-30 points)
    persistence_score = min(30, persistence * 30)
    
    # Wave importance contribution (0-30 points)
    wave_score = min(30, wave_weight * 30)
    
    # Base severity
    base_severity = magnitude_score + persistence_score + wave_score
    
    # Apply regime multiplier
    regime_multiplier = _get_regime_multiplier(regime)
    adjusted_severity = base_severity * regime_multiplier
    
    # Cap at 100
    return int(min(100, adjusted_severity))


def _compute_confidence_score(
    data_coverage: float,
    metric_agreement: float,
    recency: float
) -> int:
    """
    Compute confidence score (0-100) for a signal (Stage 2).
    
    Confidence is based on:
    - Data coverage: How complete the data is (0-40 points)
    - Metric agreement: Agreement between different metrics (0-40 points)
    - Recency: How recent the signal is (0-20 points)
    
    Args:
        data_coverage: Data completeness (0.0-1.0, 1.0 is full coverage)
        metric_agreement: Agreement between metrics (0.0-1.0, 1.0 is perfect agreement)
        recency: Recency of data (0.0-1.0, 1.0 is most recent)
        
    Returns:
        Confidence score (0-100)
    """
    coverage_score = min(40, data_coverage * 40)
    agreement_score = min(40, metric_agreement * 40)
    recency_score = min(20, recency * 20)
    
    return int(coverage_score + agreement_score + recency_score)


def _get_severity_label(severity_score: int) -> str:
    """
    Convert severity score to label (Stage 2).
    
    Args:
        severity_score: Severity score (0-100)
        
    Returns:
        Severity label ('Low', 'Medium', 'High', 'Critical')
    """
    if severity_score >= SEVERITY_HIGH_THRESHOLD:
        return 'Critical'
    elif severity_score >= SEVERITY_MEDIUM_THRESHOLD:
        return 'High'
    elif severity_score >= SEVERITY_LOW_THRESHOLD:
        return 'Medium'
    else:
        return 'Low'


def _get_action_classification(severity_label: str) -> str:
    """
    Get action classification based on severity label (Stage 2).
    
    Args:
        severity_label: Severity label ('Low', 'Medium', 'High', 'Critical')
        
    Returns:
        Action classification ('Info', 'Watch', 'Intervention')
    """
    action_map = {
        'Low': 'Info',
        'Medium': 'Watch',
        'High': 'Watch',
        'Critical': 'Intervention'
    }
    
    return action_map.get(severity_label, 'Info')


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
