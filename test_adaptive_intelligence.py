"""
test_adaptive_intelligence.py

Unit tests for adaptive_intelligence.py module

This test suite validates that the adaptive intelligence module:
1. Correctly analyzes wave health from TruthFrame data
2. Correctly analyzes regime intelligence
3. Correctly detects learning signals
4. Never modifies TruthFrame data (read-only behavior)
5. (Stage 2) Correctly computes severity scores, confidence scores, and action classifications
"""

import pandas as pd
import numpy as np
from adaptive_intelligence import (
    analyze_wave_health,
    get_wave_health_summary,
    analyze_regime_intelligence,
    detect_learning_signals,
    get_adaptive_intelligence_snapshot,
    _compute_alpha_direction,
    _compute_health_label,
    _get_regime_description,
    _get_regime_multiplier,
    _compute_severity_score,
    _compute_confidence_score,
    _get_severity_label,
    _get_action_classification
)


def create_sample_truth_frame():
    """
    Create a sample TruthFrame for testing.
    
    Returns:
        DataFrame mimicking TruthFrame structure
    """
    data = {
        'wave_id': ['sp500_wave', 'income_wave', 'crypto_l1_growth_wave'],
        'display_name': ['S&P 500 Wave', 'Income Wave', 'Crypto L1 Growth'],
        'alpha_1d': [0.002, -0.001, 0.005],
        'alpha_30d': [0.020, -0.012, 0.025],
        'alpha_60d': [0.012, -0.015, 0.020],
        'beta_target': [1.0, 0.5, 1.5],
        'beta_real': [1.05, 0.48, 1.68],
        'beta_drift': [0.05, 0.02, 0.18],
        'exposure_pct': [0.95, 0.85, 0.99],
        'data_regime_tag': ['LIVE', 'LIVE', 'SANDBOX'],
        'drawdown_60d': [-0.08, -0.05, -0.25]
    }
    
    return pd.DataFrame(data)


def test_analyze_wave_health():
    """Test wave health analysis for a specific wave"""
    truth_df = create_sample_truth_frame()
    
    # Test SP500 wave (healthy)
    health = analyze_wave_health(truth_df, 'sp500_wave')
    
    print(f"✓ SP500 Wave Health: {health['health_label']} (score: {health['health_score']})")
    assert health['wave_id'] == 'sp500_wave'
    assert health['display_name'] == 'S&P 500 Wave'
    assert health['alpha_30d'] == 0.020
    assert health['alpha_direction'] == 'improving'  # 30d (0.020) > 60d (0.012)
    assert health['beta_drift'] == 0.05
    assert health['health_label'] in ['healthy', 'watch', 'intervention_candidate', 'N/A']
    
    # Test Income wave (negative alpha)
    health = analyze_wave_health(truth_df, 'income_wave')
    
    print(f"✓ Income Wave Health: {health['health_label']} (score: {health['health_score']})")
    assert health['alpha_direction'] in ['stable', 'improving', 'decaying']  # Valid direction
    assert health['health_score'] is not None
    
    # Test non-existent wave
    health = analyze_wave_health(truth_df, 'fake_wave')
    
    print(f"✓ Non-existent Wave Health: {health['health_label']}")
    assert health['health_label'] == 'N/A'
    assert health['health_score'] is None
    
    print("✓ analyze_wave_health() tests passed")


def test_get_wave_health_summary():
    """Test getting health summary for all waves"""
    truth_df = create_sample_truth_frame()
    
    summary = get_wave_health_summary(truth_df)
    
    print(f"✓ Wave Health Summary: {len(summary)} waves analyzed")
    assert len(summary) == 3  # 3 waves in sample data
    assert all('wave_id' in h for h in summary)
    assert all('health_label' in h for h in summary)
    assert all('health_score' in h for h in summary)
    
    print("✓ get_wave_health_summary() tests passed")


def test_analyze_regime_intelligence():
    """Test regime intelligence analysis"""
    truth_df = create_sample_truth_frame()
    
    regime = analyze_regime_intelligence(truth_df)
    
    print(f"✓ Current Regime: {regime['current_regime']}")
    print(f"✓ Alignment: {regime['aligned_waves']}/{regime['total_waves']} ({regime['alignment_pct']:.1f}%)")
    
    assert regime['current_regime'] == 'LIVE'  # Most common in sample
    assert regime['total_waves'] == 3
    assert regime['aligned_waves'] == 2  # SP500 and Income are LIVE
    assert regime['misaligned_waves'] == 1  # Crypto is SANDBOX
    assert 60 < regime['alignment_pct'] < 70  # ~66.7%
    assert 'regime_description' in regime
    assert 'regime_summary' in regime
    
    print("✓ analyze_regime_intelligence() tests passed")


def test_detect_learning_signals():
    """Test learning signal detection"""
    truth_df = create_sample_truth_frame()
    
    signals = detect_learning_signals(truth_df)
    
    print(f"✓ Detected {len(signals)} learning signals")
    
    # Should detect multiple signals:
    # 1. Income wave: sustained alpha decay (both 30d and 60d negative)
    # 2. Crypto wave: high beta drift (0.18 > 0.15)
    # 3. Crypto wave: data regime mismatch (SANDBOX)
    # 4. Crypto wave: high drawdown (-0.25 < -0.20)
    # 5. Crypto wave: extreme exposure (0.99 > 0.98)
    
    assert len(signals) >= 4  # At least these signals should be detected
    
    # Check signal structure
    for signal in signals:
        assert 'signal_type' in signal
        assert 'wave_id' in signal
        assert 'display_name' in signal
        assert 'severity' in signal
        assert 'description' in signal
        assert signal['severity'] in ['info', 'warning', 'critical']
    
    # Check for specific expected signals
    signal_types = [s['signal_type'] for s in signals]
    assert 'sustained_alpha_decay' in signal_types  # Income wave
    assert 'beta_drift' in signal_types  # Crypto wave
    assert 'data_regime_mismatch' in signal_types  # Crypto wave
    
    print(f"✓ Signal types detected: {set(signal_types)}")
    print("✓ detect_learning_signals() tests passed")


def test_compute_alpha_direction():
    """Test alpha direction computation"""
    
    # Improving: 30d > 60d (difference > 0.5%)
    direction = _compute_alpha_direction(0.005, 0.020, 0.010)
    assert direction == 'improving'
    
    # Decaying: 30d < 60d (difference > 0.5%)
    direction = _compute_alpha_direction(-0.002, -0.020, -0.010)
    assert direction == 'decaying'
    
    # Stable: 30d ≈ 60d (within 0.5%)
    direction = _compute_alpha_direction(0.003, 0.012, 0.012)
    assert direction == 'stable'
    
    # N/A: Missing data
    direction = _compute_alpha_direction(None, 0.012, None)
    assert direction == 'N/A'
    
    print("✓ _compute_alpha_direction() tests passed")


def test_compute_health_label():
    """Test health label and score computation"""
    
    # Healthy: improving alpha, low drift
    label, score = _compute_health_label('improving', 0.05, 0.025, 0.90)
    assert label in ['healthy', 'watch']
    assert score >= 70
    print(f"✓ Healthy case: {label} (score: {score})")
    
    # Watch: decaying alpha, moderate drift
    label, score = _compute_health_label('decaying', 0.12, 0.015, 0.85)
    assert label in ['watch', 'intervention_candidate']
    assert score >= 50
    print(f"✓ Watch case: {label} (score: {score})")
    
    # Intervention: negative alpha, high drift
    label, score = _compute_health_label('decaying', 0.18, -0.025, 0.99)
    assert label == 'intervention_candidate'
    assert score < 60
    print(f"✓ Intervention case: {label} (score: {score})")
    
    # N/A: insufficient data
    label, score = _compute_health_label('N/A', None, None, None)
    assert label == 'N/A'
    assert score is None
    print(f"✓ N/A case: {label}")
    
    print("✓ _compute_health_label() tests passed")


def test_get_regime_description():
    """Test regime description generation"""
    
    desc = _get_regime_description('LIVE')
    assert 'live' in desc.lower() or 'Live' in desc
    
    desc = _get_regime_description('SANDBOX')
    assert 'sandbox' in desc.lower() or 'Sandbox' in desc
    
    desc = _get_regime_description('HYBRID')
    assert 'hybrid' in desc.lower() or 'Hybrid' in desc
    
    desc = _get_regime_description('UNAVAILABLE')
    assert 'unavailable' in desc.lower() or 'Unavailable' in desc
    
    print("✓ _get_regime_description() tests passed")


def test_get_adaptive_intelligence_snapshot():
    """Test complete adaptive intelligence snapshot"""
    truth_df = create_sample_truth_frame()
    
    snapshot = get_adaptive_intelligence_snapshot(truth_df)
    
    print(f"✓ Snapshot generated at {snapshot['timestamp']}")
    
    assert 'wave_health' in snapshot
    assert 'regime_intelligence' in snapshot
    assert 'learning_signals' in snapshot
    assert 'timestamp' in snapshot
    
    assert len(snapshot['wave_health']) == 3
    assert snapshot['regime_intelligence']['total_waves'] == 3
    assert len(snapshot['learning_signals']) >= 4
    
    print("✓ get_adaptive_intelligence_snapshot() tests passed")


def test_read_only_behavior():
    """Test that adaptive intelligence module never modifies TruthFrame"""
    truth_df = create_sample_truth_frame()
    
    # Create a copy for comparison
    truth_df_original = truth_df.copy()
    
    # Run all analysis functions
    analyze_wave_health(truth_df, 'sp500_wave')
    get_wave_health_summary(truth_df)
    analyze_regime_intelligence(truth_df)
    detect_learning_signals(truth_df)
    get_adaptive_intelligence_snapshot(truth_df)
    
    # Verify TruthFrame was not modified
    pd.testing.assert_frame_equal(truth_df, truth_df_original)
    
    print("✓ READ-ONLY BEHAVIOR VERIFIED - TruthFrame not modified")


# ============================================================================
# STAGE 2 TESTS
# ============================================================================

def test_regime_multiplier():
    """Test regime multiplier calculation (Stage 2)"""
    
    # LIVE regime should have normal multiplier
    multiplier = _get_regime_multiplier('LIVE')
    assert multiplier == 1.0
    print(f"✓ LIVE regime multiplier: {multiplier}")
    
    # HYBRID regime should have volatile multiplier
    multiplier = _get_regime_multiplier('HYBRID')
    assert multiplier == 1.3
    print(f"✓ HYBRID regime multiplier: {multiplier}")
    
    # SANDBOX regime should have risk-off multiplier
    multiplier = _get_regime_multiplier('SANDBOX')
    assert multiplier == 1.5
    print(f"✓ SANDBOX regime multiplier: {multiplier}")
    
    # UNAVAILABLE regime should have risk-off multiplier
    multiplier = _get_regime_multiplier('UNAVAILABLE')
    assert multiplier == 1.5
    print(f"✓ UNAVAILABLE regime multiplier: {multiplier}")
    
    print("✓ _get_regime_multiplier() tests passed")


def test_compute_severity_score():
    """Test severity score computation (Stage 2)"""
    
    # Low severity case
    score = _compute_severity_score(
        magnitude=0.2,
        persistence=0.1,
        regime='LIVE',
        wave_weight=0.3
    )
    assert 0 <= score <= 100
    assert score < 50  # Should be low severity
    print(f"✓ Low severity case: {score}/100")
    
    # High severity case
    score = _compute_severity_score(
        magnitude=0.8,
        persistence=0.9,
        regime='SANDBOX',
        wave_weight=0.5
    )
    assert 0 <= score <= 100
    assert score >= 50  # Should be high severity
    print(f"✓ High severity case: {score}/100")
    
    # Critical severity case
    score = _compute_severity_score(
        magnitude=1.0,
        persistence=1.0,
        regime='UNAVAILABLE',
        wave_weight=1.0
    )
    assert score == 100  # Capped at 100
    print(f"✓ Critical severity case: {score}/100 (capped)")
    
    print("✓ _compute_severity_score() tests passed")


def test_compute_confidence_score():
    """Test confidence score computation (Stage 2)"""
    
    # High confidence case
    score = _compute_confidence_score(
        data_coverage=1.0,
        metric_agreement=1.0,
        recency=1.0
    )
    assert score == 100
    print(f"✓ High confidence case: {score}%")
    
    # Medium confidence case
    score = _compute_confidence_score(
        data_coverage=0.7,
        metric_agreement=0.6,
        recency=0.8
    )
    assert 50 <= score < 90
    print(f"✓ Medium confidence case: {score}%")
    
    # Low confidence case
    score = _compute_confidence_score(
        data_coverage=0.3,
        metric_agreement=0.4,
        recency=0.2
    )
    assert score < 50
    print(f"✓ Low confidence case: {score}%")
    
    print("✓ _compute_confidence_score() tests passed")


def test_get_severity_label():
    """Test severity label classification (Stage 2)"""
    
    assert _get_severity_label(10) == 'Low'
    assert _get_severity_label(24) == 'Low'
    assert _get_severity_label(25) == 'Medium'
    assert _get_severity_label(49) == 'Medium'
    assert _get_severity_label(50) == 'High'
    assert _get_severity_label(74) == 'High'
    assert _get_severity_label(75) == 'Critical'
    assert _get_severity_label(100) == 'Critical'
    
    print("✓ _get_severity_label() tests passed")


def test_get_action_classification():
    """Test action classification (Stage 2)"""
    
    assert _get_action_classification('Low') == 'Info'
    assert _get_action_classification('Medium') == 'Watch'
    assert _get_action_classification('High') == 'Watch'
    assert _get_action_classification('Critical') == 'Intervention'
    
    print("✓ _get_action_classification() tests passed")


def test_detect_learning_signals_stage2():
    """Test learning signal detection with Stage 2 enhancements"""
    truth_df = create_sample_truth_frame()
    
    signals = detect_learning_signals(truth_df)
    
    print(f"✓ Detected {len(signals)} learning signals (Stage 2)")
    
    # Verify all signals have Stage 2 fields
    for signal in signals:
        assert 'signal_type' in signal
        assert 'wave_id' in signal
        assert 'display_name' in signal
        assert 'severity' in signal  # Legacy field
        assert 'severity_score' in signal  # Stage 2
        assert 'severity_label' in signal  # Stage 2
        assert 'confidence_score' in signal  # Stage 2
        assert 'action_classification' in signal  # Stage 2
        assert 'description' in signal
        
        # Validate severity score
        assert 0 <= signal['severity_score'] <= 100
        
        # Validate severity label
        assert signal['severity_label'] in ['Low', 'Medium', 'High', 'Critical']
        
        # Validate confidence score
        assert 0 <= signal['confidence_score'] <= 100
        
        # Validate action classification
        assert signal['action_classification'] in ['Info', 'Watch', 'Intervention']
        
        print(f"  - {signal['signal_type']}: Severity={signal['severity_label']} ({signal['severity_score']}), "
              f"Confidence={signal['confidence_score']}%, Action={signal['action_classification']}")
    
    print("✓ detect_learning_signals() Stage 2 enhancements verified")


def test_deterministic_severity():
    """Test that severity scoring is deterministic (Stage 2)"""
    truth_df = create_sample_truth_frame()
    
    # Run signal detection multiple times
    signals1 = detect_learning_signals(truth_df)
    signals2 = detect_learning_signals(truth_df)
    signals3 = detect_learning_signals(truth_df)
    
    # All runs should produce identical results
    assert len(signals1) == len(signals2) == len(signals3)
    
    for s1, s2, s3 in zip(signals1, signals2, signals3):
        assert s1['severity_score'] == s2['severity_score'] == s3['severity_score']
        assert s1['severity_label'] == s2['severity_label'] == s3['severity_label']
        assert s1['confidence_score'] == s2['confidence_score'] == s3['confidence_score']
        assert s1['action_classification'] == s2['action_classification'] == s3['action_classification']
    
    print("✓ DETERMINISTIC BEHAVIOR VERIFIED - Severity scoring is reproducible")


def test_regime_aware_severity():
    """Test that severity is regime-aware (Stage 2)"""
    
    # Same issue in different regimes should have different severity
    base_params = {
        'magnitude': 0.5,
        'persistence': 0.5,
        'wave_weight': 0.33
    }
    
    live_severity = _compute_severity_score(**base_params, regime='LIVE')
    hybrid_severity = _compute_severity_score(**base_params, regime='HYBRID')
    sandbox_severity = _compute_severity_score(**base_params, regime='SANDBOX')
    
    # Severity should increase in more volatile regimes
    assert live_severity < hybrid_severity < sandbox_severity
    
    print(f"✓ Regime-aware severity: LIVE={live_severity}, HYBRID={hybrid_severity}, SANDBOX={sandbox_severity}")
    print("✓ Regime-aware severity calculation verified")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ADAPTIVE INTELLIGENCE MODULE - UNIT TESTS (STAGE 2)")
    print("="*70)
    print()
    
    test_analyze_wave_health()
    print()
    
    test_get_wave_health_summary()
    print()
    
    test_analyze_regime_intelligence()
    print()
    
    test_detect_learning_signals()
    print()
    
    test_compute_alpha_direction()
    print()
    
    test_compute_health_label()
    print()
    
    test_get_regime_description()
    print()
    
    test_get_adaptive_intelligence_snapshot()
    print()
    
    test_read_only_behavior()
    print()
    
    # Stage 2 tests
    print("="*70)
    print("STAGE 2 ENHANCEMENTS - ADDITIONAL TESTS")
    print("="*70)
    print()
    
    test_regime_multiplier()
    print()
    
    test_compute_severity_score()
    print()
    
    test_compute_confidence_score()
    print()
    
    test_get_severity_label()
    print()
    
    test_get_action_classification()
    print()
    
    test_detect_learning_signals_stage2()
    print()
    
    test_deterministic_severity()
    print()
    
    test_regime_aware_severity()
    print()
    
    print("="*70)
    print("✓ ALL TESTS PASSED (INCLUDING STAGE 2 ENHANCEMENTS)")
    print("="*70)
