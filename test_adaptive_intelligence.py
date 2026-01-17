"""
test_adaptive_intelligence.py

Unit tests for adaptive_intelligence.py module

This test suite validates that the adaptive intelligence module:
1. Correctly analyzes wave health from TruthFrame data
2. Correctly analyzes regime intelligence
3. Correctly detects learning signals
4. Never modifies TruthFrame data (read-only behavior)
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
    _get_regime_description
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
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ADAPTIVE INTELLIGENCE MODULE - UNIT TESTS")
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
    
    print("="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
