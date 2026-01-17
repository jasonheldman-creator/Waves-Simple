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
- **STAGE 3**: Narrative & Causal Intelligence with clustering, change detection, and priority ranking
- **STAGE 4**: Decision Support Layer (Human-in-the-Loop) with action recommendations and attention flags

All diagnostics pull from TruthFrame and wave registry metadata only.
No legacy portfolio aggregation paths or duplicate strategy calculations.

Stage 3 Enhancements:
- Cluster related signals into causal themes (beta drift clusters, regime mismatches, etc.)
- Deterministic cluster severity, affected wave count, and persistence
- Template-based narrative explanations (no LLM)
- Change detection vs prior snapshot (new, escalating, improving, resolved)
- Priority stack ranking top 3 "What Matters Today" insights

Stage 4 Enhancements:
- Recommended review actions for top priority insights (deterministic, advisory)
- Risk of inaction metrics (Low/Medium/High)
- Wave-level attention flags (🔎 Needs Review, ⏳ Monitor, ⚠️ Escalating Risk)
- Time & trend context in narratives (persistence, direction, wave expansion/contraction)
- Decision support summary (read-only, human-review oriented)

Usage:
    from adaptive_intelligence import (
        analyze_wave_health,
        analyze_regime_intelligence,
        detect_learning_signals,
        get_wave_health_summary,
        cluster_signals,  # STAGE 3
        detect_cluster_changes,  # STAGE 3
        get_priority_insights,  # STAGE 3
        generate_recommended_action,  # STAGE 4
        calculate_risk_of_inaction,  # STAGE 4
        compute_attention_flag,  # STAGE 4
        enhance_narrative_with_time_context,  # STAGE 4
        get_decision_support_summary  # STAGE 4
    )
    
    # Get TruthFrame (read-only)
    truth_df = get_truth_frame(safe_mode=True)
    
    # Analyze wave health for a specific wave
    health = analyze_wave_health(truth_df, "sp500_wave")
    
    # Get portfolio-wide regime intelligence
    regime = analyze_regime_intelligence(truth_df)
    
    # Detect learning signals
    signals = detect_learning_signals(truth_df)
    
    # STAGE 3: Cluster signals into themes
    clusters = cluster_signals(signals)
    
    # STAGE 3: Detect changes from prior snapshot
    changes = detect_cluster_changes(clusters, prior_clusters)
    
    # STAGE 3: Get top 3 priority insights
    top_insights = get_priority_insights(clusters)
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

# Stage 3: Normalization constants for clustering
MAX_BETA_DRIFT_NORMALIZATION = 0.30  # Maximum beta drift for normalization (30% drift is extreme)
MAX_ALPHA_NORMALIZATION = 0.05  # Maximum alpha for normalization (5% is extreme)
MAX_DRAWDOWN_NORMALIZATION = 0.40  # Maximum drawdown for normalization (-40% is extreme)
MAX_EXPOSURE_DRIFT_NORMALIZATION = 0.02  # Maximum exposure drift for normalization (2% from threshold)


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
                magnitude = min(1.0, abs(alpha_60d) / MAX_ALPHA_NORMALIZATION)
                
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
            magnitude = min(1.0, beta_drift / MAX_BETA_DRIFT_NORMALIZATION)
            
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
                magnitude = min(1.0, (exposure_pct - HIGH_EXPOSURE_THRESHOLD) / MAX_EXPOSURE_DRIFT_NORMALIZATION)
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
                magnitude = min(1.0, (LOW_EXPOSURE_THRESHOLD - exposure_pct) / (LOW_EXPOSURE_THRESHOLD - EXTREME_EXPOSURE_LOW_THRESHOLD))
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
            magnitude = min(1.0, abs(drawdown_60d) / MAX_DRAWDOWN_NORMALIZATION)
            
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

def get_adaptive_intelligence_snapshot(truth_df: pd.DataFrame, prior_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get complete adaptive intelligence snapshot (READ-ONLY).
    
    This is a convenience function that bundles all adaptive intelligence
    diagnostics into a single dictionary for easy UI consumption.
    
    Stage 3 Enhancement: Includes signal clustering, change detection, and priority insights.
    
    Args:
        truth_df: TruthFrame DataFrame (not modified)
        prior_snapshot: Optional prior snapshot for change detection
        
    Returns:
        Dictionary with all adaptive intelligence data:
        {
            'wave_health': List[Dict],
            'regime_intelligence': Dict,
            'learning_signals': List[Dict],
            'signal_clusters': List[Dict],  # STAGE 3
            'cluster_changes': List[Dict],  # STAGE 3
            'priority_insights': List[Dict],  # STAGE 3
            'timestamp': datetime
        }
    """
    signals = detect_learning_signals(truth_df)
    clusters = cluster_signals(signals, truth_df)
    
    # Extract prior clusters if available
    prior_clusters = prior_snapshot.get('signal_clusters', []) if prior_snapshot else []
    cluster_changes = detect_cluster_changes(clusters, prior_clusters)
    
    priority_insights = get_priority_insights(clusters)
    
    return {
        'wave_health': get_wave_health_summary(truth_df),
        'regime_intelligence': analyze_regime_intelligence(truth_df),
        'learning_signals': signals,
        'signal_clusters': clusters,  # STAGE 3
        'cluster_changes': cluster_changes,  # STAGE 3
        'priority_insights': priority_insights,  # STAGE 3
        'timestamp': datetime.now()
    }


# ============================================================================
# STAGE 3: SIGNAL CLUSTERING & NARRATIVE INTELLIGENCE
# ============================================================================

def cluster_signals(signals: List[Dict[str, Any]], truth_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Cluster related signals into causal themes (STAGE 3, READ-ONLY).
    
    Groups signals by type into clusters representing broader systemic issues:
    - Beta Drift Cluster: Waves with tracking error
    - Regime Mismatch Cluster: Waves in non-LIVE regimes
    - Concentration Risk Cluster: Waves with extreme exposure
    - Alpha Decay Cluster: Waves with sustained underperformance
    - High Drawdown Cluster: Waves with significant losses
    
    Each cluster includes:
    - cluster_type: Type of cluster
    - cluster_severity: Deterministic severity score (0-100)
    - affected_waves: List of wave_ids in cluster
    - wave_count: Number of affected waves
    - persistence: How long this issue has persisted (0.0-1.0)
    - narrative: Template-based explanation (no LLM)
    
    Args:
        signals: List of signal dictionaries from detect_learning_signals()
        truth_df: TruthFrame DataFrame for additional context (not modified)
        
    Returns:
        List of cluster dictionaries sorted by severity (descending)
    """
    # Group signals by type
    signal_groups = {}
    for signal in signals:
        signal_type = signal['signal_type']
        if signal_type not in signal_groups:
            signal_groups[signal_type] = []
        signal_groups[signal_type].append(signal)
    
    clusters = []
    
    # Cluster 1: Beta Drift Cluster
    beta_drift_signals = signal_groups.get('beta_drift', [])
    if beta_drift_signals:
        clusters.append(_create_beta_drift_cluster(beta_drift_signals, truth_df))
    
    # Cluster 2: Regime Mismatch Cluster
    regime_mismatch_signals = signal_groups.get('data_regime_mismatch', [])
    if regime_mismatch_signals:
        clusters.append(_create_regime_mismatch_cluster(regime_mismatch_signals, truth_df))
    
    # Cluster 3: Alpha Decay Cluster
    alpha_decay_signals = signal_groups.get('sustained_alpha_decay', [])
    if alpha_decay_signals:
        clusters.append(_create_alpha_decay_cluster(alpha_decay_signals, truth_df))
    
    # Cluster 4: Concentration Risk Cluster (extreme exposure)
    exposure_high_signals = signal_groups.get('extreme_exposure_high', [])
    exposure_low_signals = signal_groups.get('extreme_exposure_low', [])
    if exposure_high_signals or exposure_low_signals:
        clusters.append(_create_concentration_risk_cluster(
            exposure_high_signals + exposure_low_signals, truth_df
        ))
    
    # Cluster 5: High Drawdown Cluster
    drawdown_signals = signal_groups.get('high_drawdown', [])
    if drawdown_signals:
        clusters.append(_create_high_drawdown_cluster(drawdown_signals, truth_df))
    
    # Sort clusters by severity (descending)
    clusters.sort(key=lambda c: c['cluster_severity'], reverse=True)
    
    return clusters


def _create_beta_drift_cluster(signals: List[Dict[str, Any]], truth_df: pd.DataFrame) -> Dict[str, Any]:
    """Create beta drift cluster from related signals."""
    affected_waves = [s['wave_id'] for s in signals]
    wave_count = len(affected_waves)
    
    # Calculate cluster severity (average of signal severities)
    avg_severity = int(np.mean([s['severity_score'] for s in signals]))
    
    # Calculate persistence (use average beta drift magnitude as proxy)
    avg_drift = np.mean([s.get('metric_value', 0.0) for s in signals if s.get('metric_value')])
    persistence = min(1.0, avg_drift / MAX_BETA_DRIFT_NORMALIZATION)
    
    # Generate narrative
    max_drift_signal = max(signals, key=lambda s: s.get('metric_value', 0.0))
    max_drift = max_drift_signal.get('metric_value', 0.0)
    max_drift_wave = max_drift_signal.get('display_name', 'Unknown')
    
    narrative = (
        f"**Beta Drift Detected:** {wave_count} wave{'s' if wave_count > 1 else ''} "
        f"showing tracking error vs target beta. "
        f"Largest drift: {max_drift_wave} ({max_drift:.3f}). "
        f"This indicates portfolio allocation may be deviating from intended market exposure. "
        f"Review rebalancing thresholds and consider tactical adjustments if drift persists."
    )
    
    return {
        'cluster_type': 'beta_drift',
        'cluster_name': 'Beta Drift Cluster',
        'cluster_severity': avg_severity,
        'affected_waves': affected_waves,
        'wave_count': wave_count,
        'persistence': persistence,
        'narrative': narrative,
        'signals': signals
    }


def _create_regime_mismatch_cluster(signals: List[Dict[str, Any]], truth_df: pd.DataFrame) -> Dict[str, Any]:
    """Create regime mismatch cluster from related signals."""
    affected_waves = [s['wave_id'] for s in signals]
    wave_count = len(affected_waves)
    
    # Regime mismatch is always high severity
    avg_severity = int(np.mean([s['severity_score'] for s in signals]))
    
    # Persistence is high for regime issues (assumed persistent until fixed)
    persistence = 0.9
    
    # Count by regime type
    regime_counts = {}
    for signal in signals:
        # Extract regime from description
        desc = signal['description']
        if 'UNAVAILABLE' in desc:
            regime = 'UNAVAILABLE'
        elif 'SANDBOX' in desc:
            regime = 'SANDBOX'
        elif 'HYBRID' in desc:
            regime = 'HYBRID'
        else:
            regime = 'UNKNOWN'
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    # Generate narrative
    regime_summary = ', '.join([f"{count} in {regime}" for regime, count in regime_counts.items()])
    narrative = (
        f"**Regime Mismatch Alert:** {wave_count} wave{'s' if wave_count > 1 else ''} "
        f"operating in non-LIVE data regimes ({regime_summary}). "
        f"These waves may be using stale, simulated, or unavailable data. "
        f"Verify data pipeline health and consider excluding these waves from execution "
        f"until data quality improves."
    )
    
    return {
        'cluster_type': 'regime_mismatch',
        'cluster_name': 'Regime Mismatch Cluster',
        'cluster_severity': avg_severity,
        'affected_waves': affected_waves,
        'wave_count': wave_count,
        'persistence': persistence,
        'narrative': narrative,
        'signals': signals
    }


def _create_alpha_decay_cluster(signals: List[Dict[str, Any]], truth_df: pd.DataFrame) -> Dict[str, Any]:
    """Create alpha decay cluster from related signals."""
    affected_waves = [s['wave_id'] for s in signals]
    wave_count = len(affected_waves)
    
    # Calculate cluster severity
    avg_severity = int(np.mean([s['severity_score'] for s in signals]))
    
    # Persistence is high (both 30d and 60d negative indicates sustained issue)
    persistence = 0.8
    
    # Calculate total underperformance
    avg_alpha = np.mean([s.get('metric_value', 0.0) for s in signals if s.get('metric_value')])
    
    # Find worst performer
    worst_signal = min(signals, key=lambda s: s.get('metric_value', 0.0))
    worst_alpha = worst_signal.get('metric_value', 0.0)
    worst_wave = worst_signal.get('display_name', 'Unknown')
    
    # Generate narrative
    narrative = (
        f"**Sustained Alpha Decay:** {wave_count} wave{'s' if wave_count > 1 else ''} "
        f"underperforming benchmark over 30+ days. "
        f"Average 60d alpha: {avg_alpha*100:.2f}%. "
        f"Worst performer: {worst_wave} ({worst_alpha*100:.2f}%). "
        f"This pattern suggests strategy ineffectiveness or adverse market conditions. "
        f"Review strategy assumptions, factor exposures, and consider defensive positioning."
    )
    
    return {
        'cluster_type': 'alpha_decay',
        'cluster_name': 'Alpha Decay Cluster',
        'cluster_severity': avg_severity,
        'affected_waves': affected_waves,
        'wave_count': wave_count,
        'persistence': persistence,
        'narrative': narrative,
        'signals': signals
    }


def _create_concentration_risk_cluster(signals: List[Dict[str, Any]], truth_df: pd.DataFrame) -> Dict[str, Any]:
    """Create concentration risk cluster from extreme exposure signals."""
    affected_waves = [s['wave_id'] for s in signals]
    wave_count = len(affected_waves)
    
    # Calculate cluster severity
    avg_severity = int(np.mean([s['severity_score'] for s in signals]))
    
    # Persistence is moderate (exposure can change quickly)
    persistence = 0.4
    
    # Separate high and low exposure
    high_exposure = [s for s in signals if 'high' in s['signal_type']]
    low_exposure = [s for s in signals if 'low' in s['signal_type']]
    
    # Generate narrative
    if high_exposure and low_exposure:
        narrative = (
            f"**Concentration Risk Alert:** {len(high_exposure)} wave{'s' if len(high_exposure) > 1 else ''} "
            f"with very high exposure (>98%) and {len(low_exposure)} wave{'s' if len(low_exposure) > 1 else ''} "
            f"with low exposure (<50%). "
            f"This indicates imbalanced capital allocation. "
            f"High exposure waves have minimal cash buffer for rebalancing or drawdowns. "
            f"Low exposure waves may be underutilized. Review allocation strategy."
        )
    elif high_exposure:
        narrative = (
            f"**High Concentration Risk:** {len(high_exposure)} wave{'s' if len(high_exposure) > 1 else ''} "
            f"operating at >98% exposure with minimal cash buffer. "
            f"Limited flexibility for rebalancing or managing drawdowns. "
            f"Consider reducing exposure or increasing cash allocation for risk management."
        )
    else:
        narrative = (
            f"**Low Exposure Alert:** {len(low_exposure)} wave{'s' if len(low_exposure) > 1 else ''} "
            f"holding <50% exposure with high cash allocation. "
            f"Capital may be underutilized. "
            f"Review investment policy and consider increasing exposure if market conditions permit."
        )
    
    return {
        'cluster_type': 'concentration_risk',
        'cluster_name': 'Concentration Risk Cluster',
        'cluster_severity': avg_severity,
        'affected_waves': affected_waves,
        'wave_count': wave_count,
        'persistence': persistence,
        'narrative': narrative,
        'signals': signals
    }


def _create_high_drawdown_cluster(signals: List[Dict[str, Any]], truth_df: pd.DataFrame) -> Dict[str, Any]:
    """Create high drawdown cluster from related signals."""
    affected_waves = [s['wave_id'] for s in signals]
    wave_count = len(affected_waves)
    
    # Calculate cluster severity
    avg_severity = int(np.mean([s['severity_score'] for s in signals]))
    
    # Persistence is high (60d drawdown indicates sustained issue)
    persistence = 0.8
    
    # Calculate average drawdown
    avg_drawdown = np.mean([s.get('metric_value', 0.0) for s in signals if s.get('metric_value')])
    
    # Find worst drawdown
    worst_signal = min(signals, key=lambda s: s.get('metric_value', 0.0))
    worst_drawdown = worst_signal.get('metric_value', 0.0)
    worst_wave = worst_signal.get('display_name', 'Unknown')
    
    # Generate narrative
    narrative = (
        f"**High Drawdown Alert:** {wave_count} wave{'s' if wave_count > 1 else ''} "
        f"experiencing significant 60-day drawdowns. "
        f"Average drawdown: {avg_drawdown*100:.2f}%. "
        f"Worst drawdown: {worst_wave} ({worst_drawdown*100:.2f}%). "
        f"Extended drawdowns increase recovery time and may indicate structural issues. "
        f"Review risk management, stop-loss policies, and consider defensive hedging strategies."
    )
    
    return {
        'cluster_type': 'high_drawdown',
        'cluster_name': 'High Drawdown Cluster',
        'cluster_severity': avg_severity,
        'affected_waves': affected_waves,
        'wave_count': wave_count,
        'persistence': persistence,
        'narrative': narrative,
        'signals': signals
    }


# ============================================================================
# STAGE 3: CHANGE DETECTION
# ============================================================================

def detect_cluster_changes(
    current_clusters: List[Dict[str, Any]], 
    prior_clusters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect changes between current and prior cluster snapshots (STAGE 3, READ-ONLY).
    
    Identifies:
    - New clusters: Clusters that didn't exist before
    - Escalating clusters: Clusters with increased severity or wave count
    - Improving clusters: Clusters with decreased severity or wave count
    - Resolved clusters: Clusters that existed before but no longer exist
    
    Args:
        current_clusters: Current cluster list
        prior_clusters: Prior cluster list (may be empty)
        
    Returns:
        List of change dictionaries:
        [
            {
                'change_type': str ('new', 'escalating', 'improving', 'resolved'),
                'cluster_type': str,
                'cluster_name': str,
                'severity_change': int (delta in severity),
                'wave_count_change': int (delta in wave count),
                'description': str
            },
            ...
        ]
    """
    if not prior_clusters:
        # All current clusters are new
        return [
            {
                'change_type': 'new',
                'cluster_type': c['cluster_type'],
                'cluster_name': c['cluster_name'],
                'severity_change': c['cluster_severity'],
                'wave_count_change': c['wave_count'],
                'description': f"New {c['cluster_name'].lower()} detected with {c['wave_count']} affected wave{'s' if c['wave_count'] > 1 else ''}"
            }
            for c in current_clusters
        ]
    
    changes = []
    
    # Create lookup maps
    current_map = {c['cluster_type']: c for c in current_clusters}
    prior_map = {c['cluster_type']: c for c in prior_clusters}
    
    # Check for new, escalating, and improving clusters
    for cluster_type, current_cluster in current_map.items():
        if cluster_type not in prior_map:
            # New cluster
            changes.append({
                'change_type': 'new',
                'cluster_type': cluster_type,
                'cluster_name': current_cluster['cluster_name'],
                'severity_change': current_cluster['cluster_severity'],
                'wave_count_change': current_cluster['wave_count'],
                'description': f"New {current_cluster['cluster_name'].lower()} detected with {current_cluster['wave_count']} affected wave{'s' if current_cluster['wave_count'] > 1 else ''}"
            })
        else:
            # Existing cluster - check for changes
            prior_cluster = prior_map[cluster_type]
            severity_delta = current_cluster['cluster_severity'] - prior_cluster['cluster_severity']
            wave_count_delta = current_cluster['wave_count'] - prior_cluster['wave_count']
            
            # Threshold for meaningful change
            if abs(severity_delta) >= 10 or wave_count_delta != 0:
                if severity_delta > 0 or wave_count_delta > 0:
                    # Escalating
                    changes.append({
                        'change_type': 'escalating',
                        'cluster_type': cluster_type,
                        'cluster_name': current_cluster['cluster_name'],
                        'severity_change': severity_delta,
                        'wave_count_change': wave_count_delta,
                        'description': (
                            f"{current_cluster['cluster_name']} escalating: "
                            f"severity {'↑' if severity_delta > 0 else '→'}{abs(severity_delta)} points, "
                            f"waves {'↑' if wave_count_delta > 0 else '→'}{abs(wave_count_delta)}"
                        )
                    })
                else:
                    # Improving
                    changes.append({
                        'change_type': 'improving',
                        'cluster_type': cluster_type,
                        'cluster_name': current_cluster['cluster_name'],
                        'severity_change': severity_delta,
                        'wave_count_change': wave_count_delta,
                        'description': (
                            f"{current_cluster['cluster_name']} improving: "
                            f"severity ↓{abs(severity_delta)} points, "
                            f"waves ↓{abs(wave_count_delta)}"
                        )
                    })
    
    # Check for resolved clusters
    for cluster_type, prior_cluster in prior_map.items():
        if cluster_type not in current_map:
            changes.append({
                'change_type': 'resolved',
                'cluster_type': cluster_type,
                'cluster_name': prior_cluster['cluster_name'],
                'severity_change': -prior_cluster['cluster_severity'],
                'wave_count_change': -prior_cluster['wave_count'],
                'description': f"{prior_cluster['cluster_name']} resolved - no longer detected"
            })
    
    return changes


# ============================================================================
# STAGE 3: PRIORITY STACK
# ============================================================================

def get_priority_insights(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get top 3 priority insights ("What Matters Today") from clusters (STAGE 3, READ-ONLY).
    
    Prioritization algorithm:
    1. Cluster severity (0-100)
    2. Number of affected waves (more waves = higher priority)
    3. Regime sensitivity (regime mismatches prioritized)
    4. Persistence (longer-lasting issues prioritized)
    
    Args:
        clusters: List of cluster dictionaries
        
    Returns:
        List of top 3 priority insights (or fewer if less than 3 clusters exist):
        [
            {
                'rank': int (1-3),
                'cluster_type': str,
                'cluster_name': str,
                'cluster_severity': int,
                'wave_count': int,
                'priority_score': float,
                'narrative': str,
                'justification': str (why this is prioritized)
            },
            ...
        ]
    """
    if not clusters:
        return []
    
    # Calculate priority score for each cluster
    scored_clusters = []
    for cluster in clusters:
        priority_score = _calculate_priority_score(cluster)
        scored_clusters.append({
            **cluster,
            'priority_score': priority_score
        })
    
    # Sort by priority score (descending)
    scored_clusters.sort(key=lambda c: c['priority_score'], reverse=True)
    
    # Take top 3
    top_3 = scored_clusters[:3]
    
    # Add rank and justification
    insights = []
    for rank, cluster in enumerate(top_3, start=1):
        justification = _generate_priority_justification(cluster, rank)
        insights.append({
            'rank': rank,
            'cluster_type': cluster['cluster_type'],
            'cluster_name': cluster['cluster_name'],
            'cluster_severity': cluster['cluster_severity'],
            'wave_count': cluster['wave_count'],
            'priority_score': cluster['priority_score'],
            'narrative': cluster['narrative'],
            'justification': justification,
            'affected_waves': cluster.get('affected_waves', []),  # STAGE 4
            'persistence': cluster.get('persistence', 0.5)  # STAGE 4
        })
    
    return insights


def _calculate_priority_score(cluster: Dict[str, Any]) -> float:
    """
    Calculate priority score for a cluster.
    
    Formula:
    - Severity: 40% weight (normalized 0-100)
    - Wave count: 30% weight (normalized by total waves)
    - Regime sensitivity: 20% weight (regime mismatches get boost)
    - Persistence: 10% weight (0.0-1.0)
    
    Args:
        cluster: Cluster dictionary
        
    Returns:
        Priority score (0-100)
    """
    # Severity component (0-40 points)
    severity_component = (cluster['cluster_severity'] / 100.0) * 40
    
    # Wave count component (0-30 points)
    # Assume max 10 waves as normalization factor
    wave_count_component = min(cluster['wave_count'] / 10.0, 1.0) * 30
    
    # Regime sensitivity component (0-20 points)
    if cluster['cluster_type'] == 'regime_mismatch':
        regime_component = 20  # Full points for regime issues
    elif cluster['cluster_type'] in ['beta_drift', 'alpha_decay']:
        regime_component = 10  # Half points for tracking issues
    else:
        regime_component = 5  # Minimal points for exposure issues
    
    # Persistence component (0-10 points)
    persistence_component = cluster['persistence'] * 10
    
    # Total priority score
    priority_score = (
        severity_component + 
        wave_count_component + 
        regime_component + 
        persistence_component
    )
    
    return round(priority_score, 2)


def _generate_priority_justification(cluster: Dict[str, Any], rank: int) -> str:
    """
    Generate justification text for why a cluster is prioritized.
    
    Args:
        cluster: Cluster dictionary
        rank: Priority rank (1-3)
        
    Returns:
        Justification string
    """
    severity = cluster['cluster_severity']
    wave_count = cluster['wave_count']
    persistence = cluster['persistence']
    cluster_type = cluster['cluster_type']
    
    # Build justification components
    reasons = []
    
    if severity >= 75:
        reasons.append("critical severity")
    elif severity >= 50:
        reasons.append("high severity")
    
    if wave_count >= 5:
        reasons.append(f"affects {wave_count} waves")
    elif wave_count >= 3:
        reasons.append(f"affects multiple waves ({wave_count})")
    
    if persistence >= 0.8:
        reasons.append("highly persistent issue")
    elif persistence >= 0.5:
        reasons.append("moderately persistent")
    
    if cluster_type == 'regime_mismatch':
        reasons.append("data quality concern")
    elif cluster_type == 'alpha_decay':
        reasons.append("underperformance")
    elif cluster_type == 'beta_drift':
        reasons.append("tracking error concern")
    
    # Format justification
    if reasons:
        justification = f"Ranked #{rank} due to: {', '.join(reasons)}"
    else:
        justification = f"Ranked #{rank} based on overall priority scoring"
    
    return justification


# ============================================================================
# STAGE 4: DECISION SUPPORT LAYER (HUMAN-IN-THE-LOOP)
# ============================================================================

def generate_recommended_action(cluster: Dict[str, Any]) -> str:
    """
    Generate deterministic recommended review action for a cluster (STAGE 4, READ-ONLY).
    
    Maps cluster type and severity to specific, actionable human review recommendations.
    All recommendations are advisory only and do not trigger any automated changes.
    
    Args:
        cluster: Cluster dictionary with type, severity, and metadata
        
    Returns:
        Recommended action string (e.g., "Review beta targets for affected waves")
    """
    cluster_type = cluster['cluster_type']
    severity = cluster['cluster_severity']
    wave_count = cluster['wave_count']
    
    # Beta drift cluster recommendations
    if cluster_type == 'beta_drift':
        if severity >= 75:
            return "Urgent: Review beta targets and rebalancing thresholds for all affected waves"
        elif severity >= 50:
            return "Review beta targets for affected waves; consider tactical rebalancing"
        else:
            return "Monitor beta drift trends; review if persistence increases"
    
    # Regime mismatch cluster recommendations
    elif cluster_type == 'regime_mismatch':
        if severity >= 75:
            return "Critical: Investigate data pipeline health; consider excluding affected waves"
        elif severity >= 50:
            return "Investigate data regime mismatch; verify data quality"
        else:
            return "Monitor data regime status; verify expected behavior"
    
    # Alpha decay cluster recommendations
    elif cluster_type == 'alpha_decay':
        if severity >= 75:
            return "Critical: Review strategy effectiveness and factor exposures; consider defensive positioning"
        elif severity >= 50:
            return "Review strategy assumptions and recent market conditions"
        else:
            return "Monitor alpha trends; review strategy if decay persists"
    
    # Concentration risk cluster recommendations
    elif cluster_type == 'concentration_risk':
        if severity >= 75:
            return "Review allocation strategy; rebalance exposure levels if appropriate"
        elif severity >= 50:
            return "Review capital allocation across affected waves"
        else:
            return "Monitor exposure levels; ensure within policy limits"
    
    # High drawdown cluster recommendations
    elif cluster_type == 'high_drawdown':
        if severity >= 75:
            return "Urgent: Review risk management and stop-loss policies; consider defensive hedging"
        elif severity >= 50:
            return "Review drawdown recovery strategies and risk controls"
        else:
            return "Monitor drawdown trends; review if recovery stalls"
    
    # Default fallback
    else:
        return "Review cluster details and determine appropriate response"


def calculate_risk_of_inaction(cluster: Dict[str, Any]) -> str:
    """
    Calculate deterministic risk of inaction metric (STAGE 4, READ-ONLY).
    
    Classifies risk level (Low/Medium/High) based on cluster severity, persistence,
    and affected wave count. This is purely advisory and informational.
    
    Args:
        cluster: Cluster dictionary with severity, persistence, and wave count
        
    Returns:
        Risk level string: "Low", "Medium", or "High"
    """
    severity = cluster['cluster_severity']
    persistence = cluster['persistence']
    wave_count = cluster['wave_count']
    
    # Calculate composite risk score (0-100)
    # Weights: severity 50%, persistence 30%, wave count 20%
    severity_component = (severity / 100.0) * 50
    persistence_component = persistence * 30
    wave_count_component = min(wave_count / 10.0, 1.0) * 20
    
    risk_score = severity_component + persistence_component + wave_count_component
    
    # Classify risk level
    if risk_score >= 65:
        return "High"
    elif risk_score >= 35:
        return "Medium"
    else:
        return "Low"


def compute_attention_flag(wave_id: str, clusters: List[Dict[str, Any]], truth_df: pd.DataFrame) -> str:
    """
    Compute wave-level attention flag (STAGE 4, READ-ONLY).
    
    Determines appropriate non-interactive flag for a wave based on:
    - Cluster severity affecting this wave
    - Change direction (escalating/improving)
    - Regime sensitivity
    
    Flags:
    - 🔎 Needs Review: Wave in high/critical severity cluster
    - ⏳ Monitor: Wave in medium severity cluster or improving
    - ⚠️ Escalating Risk: Wave in escalating cluster with high severity
    
    Args:
        wave_id: Wave identifier
        clusters: List of current clusters
        truth_df: TruthFrame DataFrame for regime context
        
    Returns:
        Flag string: "🔎 Needs Review", "⏳ Monitor", "⚠️ Escalating Risk", or "" (none)
    """
    # Find clusters affecting this wave
    affecting_clusters = [c for c in clusters if wave_id in c['affected_waves']]
    
    if not affecting_clusters:
        return ""  # No flag if wave not in any cluster
    
    # Get max severity across affecting clusters
    max_severity = max(c['cluster_severity'] for c in affecting_clusters)
    
    # Check for escalating regime mismatch (highest priority)
    regime_clusters = [c for c in affecting_clusters if c['cluster_type'] == 'regime_mismatch']
    if regime_clusters and max_severity >= 50:
        return "⚠️ Escalating Risk"
    
    # Check severity level
    if max_severity >= 75:
        return "⚠️ Escalating Risk"
    elif max_severity >= 50:
        return "🔎 Needs Review"
    elif max_severity >= 25:
        return "⏳ Monitor"
    else:
        return ""  # Low severity, no flag needed


def enhance_narrative_with_time_context(
    cluster: Dict[str, Any],
    prior_clusters: List[Dict[str, Any]],
    snapshot_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Enhance cluster narrative with time and trend context (STAGE 4, READ-ONLY).
    
    Adds deterministic phrasing about:
    - Persistence (e.g., "persisting 6 of last 8 snapshots")
    - Directional trend (e.g., "Escalating over last 3 snapshots")
    - Affected-wave expansion/contraction
    
    All context derived from existing snapshot history.
    
    Args:
        cluster: Current cluster dictionary
        prior_clusters: List of prior clusters for comparison
        snapshot_history: Optional list of historical snapshots (not yet implemented)
        
    Returns:
        Enhanced narrative string with time context
    """
    base_narrative = cluster['narrative']
    cluster_type = cluster['cluster_type']
    
    # Find matching prior cluster
    prior_cluster = next((c for c in prior_clusters if c['cluster_type'] == cluster_type), None)
    
    time_context_parts = []
    
    # Add persistence context
    persistence = cluster['persistence']
    if persistence >= 0.8:
        time_context_parts.append("**Highly persistent issue**")
    elif persistence >= 0.5:
        time_context_parts.append("**Moderately persistent**")
    
    # Add trend context if prior data available
    if prior_cluster:
        severity_delta = cluster['cluster_severity'] - prior_cluster['cluster_severity']
        wave_delta = cluster['wave_count'] - prior_cluster['wave_count']
        
        # Determine trend
        if severity_delta >= 10 or wave_delta > 0:
            if severity_delta >= 20:
                time_context_parts.append("**⬆️ Escalating rapidly over last snapshot**")
            else:
                time_context_parts.append("**⬆️ Escalating since last snapshot**")
        elif severity_delta <= -10 or wave_delta < 0:
            time_context_parts.append("**⬇️ Improving but unresolved**")
        else:
            time_context_parts.append("**→ Stable since last snapshot**")
        
        # Add wave count change context
        if wave_delta > 0:
            time_context_parts.append(f"*Expanded by {wave_delta} wave{'s' if wave_delta > 1 else ''} since prior snapshot*")
        elif wave_delta < 0:
            time_context_parts.append(f"*Contracted by {abs(wave_delta)} wave{'s' if abs(wave_delta) > 1 else ''} since prior snapshot*")
    else:
        time_context_parts.append("**🆕 Newly detected in this snapshot**")
    
    # Combine base narrative with time context
    if time_context_parts:
        time_context = "\n\n" + " · ".join(time_context_parts)
        return base_narrative + time_context
    else:
        return base_narrative


def get_decision_support_summary(
    priority_insights: List[Dict[str, Any]],
    prior_clusters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate Decision Support Summary for top priority insights (STAGE 4, READ-ONLY).
    
    For each of the top 3 priority insights, adds:
    - Recommended review action
    - Risk of inaction metric (Low/Medium/High)
    - Enhanced narrative with time context
    
    Args:
        priority_insights: List of top priority insights from Stage 3
        prior_clusters: List of prior clusters for time context
        
    Returns:
        List of enhanced decision support dictionaries:
        [
            {
                ...all original insight fields...,
                'recommended_action': str,
                'risk_of_inaction': str,
                'enhanced_narrative': str
            }
        ]
    """
    decision_support = []
    
    for insight in priority_insights:
        # Create cluster dict from insight for helper functions
        cluster = {
            'cluster_type': insight['cluster_type'],
            'cluster_name': insight['cluster_name'],
            'cluster_severity': insight['cluster_severity'],
            'wave_count': insight['wave_count'],
            'persistence': insight.get('persistence', 0.5),  # Default if missing
            'narrative': insight['narrative'],
            'affected_waves': insight.get('affected_waves', [])
        }
        
        # Generate Stage 4 enhancements
        recommended_action = generate_recommended_action(cluster)
        risk_of_inaction = calculate_risk_of_inaction(cluster)
        enhanced_narrative = enhance_narrative_with_time_context(cluster, prior_clusters)
        
        # Combine original insight with Stage 4 additions
        decision_support.append({
            **insight,
            'recommended_action': recommended_action,
            'risk_of_inaction': risk_of_inaction,
            'enhanced_narrative': enhanced_narrative
        })
    
    return decision_support
def get_decision_support_summary(truth_df, prior_snapshot=None):
    """
    Stage 4 compatibility wrapper.
    Returns the decision support summary for UI consumption.
    """
    snapshot = get_adaptive_intelligence_snapshot(truth_df, prior_snapshot)
    return snapshot.get("decision_support_summary", [])