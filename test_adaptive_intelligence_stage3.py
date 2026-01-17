"""
test_adaptive_intelligence_stage3.py

Unit tests for Stage 3 Adaptive Intelligence features:
- Signal clustering
- Change detection
- Priority insights
- Narrative generation

This test suite validates that Stage 3 features:
1. Correctly cluster signals by causal theme
2. Calculate deterministic cluster severity, wave count, and persistence
3. Generate template-based narratives (no LLM)
4. Detect changes between snapshots (new, escalating, improving, resolved)
5. Rank clusters by priority to surface top 3 insights
6. Maintain read-only behavior (no TruthFrame modifications)
7. Produce deterministic, reproducible results
"""

import pandas as pd
import numpy as np
from adaptive_intelligence import (
    detect_learning_signals,
    cluster_signals,
    detect_cluster_changes,
    get_priority_insights,
    get_adaptive_intelligence_snapshot,
    _calculate_priority_score,
    _generate_priority_justification
)


def create_sample_truth_frame():
    """
    Create a sample TruthFrame for testing Stage 3 features.
    
    Returns:
        DataFrame mimicking TruthFrame structure with diverse scenarios
    """
    data = {
        'wave_id': [
            'sp500_wave', 
            'income_wave', 
            'crypto_l1_growth_wave',
            'tech_wave',
            'value_wave'
        ],
        'display_name': [
            'S&P 500 Wave', 
            'Income Wave', 
            'Crypto L1 Growth',
            'Tech Wave',
            'Value Wave'
        ],
        'alpha_1d': [0.002, -0.001, 0.005, -0.003, -0.002],
        'alpha_30d': [0.020, -0.012, 0.025, -0.015, -0.018],
        'alpha_60d': [0.012, -0.015, 0.020, -0.020, -0.022],
        'beta_target': [1.0, 0.5, 1.5, 1.2, 0.8],
        'beta_real': [1.05, 0.48, 1.68, 1.02, 0.78],
        'beta_drift': [0.05, 0.02, 0.18, 0.02, 0.02],
        'exposure_pct': [0.95, 0.85, 0.99, 0.40, 0.92],
        'data_regime_tag': ['LIVE', 'LIVE', 'SANDBOX', 'LIVE', 'UNAVAILABLE'],
        'drawdown_60d': [-0.08, -0.05, -0.25, -0.12, -0.15]
    }
    
    return pd.DataFrame(data)


def test_cluster_signals():
    """Test signal clustering by causal theme"""
    print("\n=== Testing cluster_signals() ===")
    
    truth_df = create_sample_truth_frame()
    signals = detect_learning_signals(truth_df)
    
    print(f"✓ Generated {len(signals)} signals for clustering")
    
    # Cluster the signals
    clusters = cluster_signals(signals, truth_df)
    
    print(f"✓ Created {len(clusters)} clusters")
    
    # Validate cluster structure
    assert len(clusters) > 0, "Should create at least one cluster"
    
    for cluster in clusters:
        # Check required fields
        assert 'cluster_type' in cluster
        assert 'cluster_name' in cluster
        assert 'cluster_severity' in cluster
        assert 'affected_waves' in cluster
        assert 'wave_count' in cluster
        assert 'persistence' in cluster
        assert 'narrative' in cluster
        assert 'signals' in cluster
        
        # Validate data types
        assert isinstance(cluster['cluster_severity'], int)
        assert 0 <= cluster['cluster_severity'] <= 100
        assert isinstance(cluster['wave_count'], int)
        assert cluster['wave_count'] > 0
        assert isinstance(cluster['persistence'], float)
        assert 0.0 <= cluster['persistence'] <= 1.0
        assert isinstance(cluster['narrative'], str)
        assert len(cluster['narrative']) > 0
        
        print(f"✓ {cluster['cluster_name']}: {cluster['wave_count']} waves, severity {cluster['cluster_severity']}, persistence {cluster['persistence']:.2f}")
    
    # Check that clusters are sorted by severity
    severities = [c['cluster_severity'] for c in clusters]
    assert severities == sorted(severities, reverse=True), "Clusters should be sorted by severity (descending)"
    
    print("✓ cluster_signals() tests passed")


def test_cluster_types():
    """Test that different cluster types are created correctly"""
    print("\n=== Testing cluster types ===")
    
    truth_df = create_sample_truth_frame()
    signals = detect_learning_signals(truth_df)
    clusters = cluster_signals(signals, truth_df)
    
    cluster_types = [c['cluster_type'] for c in clusters]
    
    # Expected cluster types based on our sample data
    expected_types = {
        'beta_drift',  # crypto_l1_growth_wave has high drift
        'regime_mismatch',  # crypto_l1_growth_wave in SANDBOX, value_wave UNAVAILABLE
        'alpha_decay',  # income_wave, tech_wave, value_wave have negative alpha
        'concentration_risk',  # crypto_l1_growth_wave has 99% exposure, tech_wave has 40%
        'high_drawdown'  # crypto_l1_growth_wave has -25% drawdown
    }
    
    for cluster_type in cluster_types:
        assert cluster_type in expected_types, f"Unexpected cluster type: {cluster_type}"
        print(f"✓ Found expected cluster type: {cluster_type}")
    
    print("✓ Cluster type tests passed")


def test_cluster_narratives():
    """Test that cluster narratives are properly generated"""
    print("\n=== Testing cluster narratives ===")
    
    truth_df = create_sample_truth_frame()
    signals = detect_learning_signals(truth_df)
    clusters = cluster_signals(signals, truth_df)
    
    for cluster in clusters:
        narrative = cluster['narrative']
        cluster_name = cluster['cluster_name']
        wave_count = cluster['wave_count']
        
        # Validate narrative contains key information
        assert len(narrative) > 50, f"Narrative for {cluster_name} is too short"
        
        # For concentration risk, wave count may be split into high/low exposure
        # For other clusters, check wave count appears
        if cluster['cluster_type'] != 'concentration_risk':
            wave_count_mentioned = (
                str(wave_count) in narrative or 
                f"{wave_count} wave" in narrative
            )
            assert wave_count_mentioned, \
                f"Narrative should mention wave count for {cluster_name}"
        else:
            # For concentration risk, just check that wave numbers appear
            assert any(str(i) in narrative for i in range(1, 10)), \
                f"Narrative should mention wave counts for {cluster_name}"
        
        # Check for template markers (no LLM artifacts)
        assert "```" not in narrative, "Narrative should not contain code blocks"
        assert "AI:" not in narrative.upper(), "Narrative should not contain AI markers"
        
        print(f"✓ {cluster_name} narrative: {len(narrative)} chars")
    
    print("✓ Cluster narrative tests passed")


def test_detect_cluster_changes_new():
    """Test detection of new clusters"""
    print("\n=== Testing new cluster detection ===")
    
    truth_df = create_sample_truth_frame()
    signals = detect_learning_signals(truth_df)
    current_clusters = cluster_signals(signals, truth_df)
    
    # No prior clusters - all should be new
    changes = detect_cluster_changes(current_clusters, [])
    
    print(f"✓ Detected {len(changes)} changes")
    
    assert len(changes) == len(current_clusters), "All clusters should be marked as new"
    
    for change in changes:
        assert change['change_type'] == 'new'
        assert 'cluster_type' in change
        assert 'cluster_name' in change
        assert 'description' in change
        assert change['severity_change'] > 0  # New clusters have positive severity
        
        print(f"✓ New cluster: {change['cluster_name']}")
    
    print("✓ New cluster detection tests passed")


def test_detect_cluster_changes_escalating():
    """Test detection of escalating clusters"""
    print("\n=== Testing escalating cluster detection ===")
    
    truth_df = create_sample_truth_frame()
    signals = detect_learning_signals(truth_df)
    current_clusters = cluster_signals(signals, truth_df)
    
    # Create prior clusters with lower severity
    prior_clusters = []
    for cluster in current_clusters:
        prior_cluster = cluster.copy()
        prior_cluster['cluster_severity'] = max(0, cluster['cluster_severity'] - 20)  # Reduce severity
        prior_cluster['wave_count'] = max(1, cluster['wave_count'] - 1)  # Reduce wave count
        prior_clusters.append(prior_cluster)
    
    changes = detect_cluster_changes(current_clusters, prior_clusters)
    
    # Should detect escalating changes
    escalating_changes = [c for c in changes if c['change_type'] == 'escalating']
    
    print(f"✓ Detected {len(escalating_changes)} escalating clusters")
    
    for change in escalating_changes:
        assert change['severity_change'] > 0 or change['wave_count_change'] > 0
        print(f"✓ Escalating: {change['cluster_name']} (severity +{change['severity_change']}, waves +{change['wave_count_change']})")
    
    print("✓ Escalating cluster detection tests passed")


def test_detect_cluster_changes_improving():
    """Test detection of improving clusters"""
    print("\n=== Testing improving cluster detection ===")
    
    truth_df = create_sample_truth_frame()
    signals = detect_learning_signals(truth_df)
    current_clusters = cluster_signals(signals, truth_df)
    
    # Create prior clusters with higher severity
    prior_clusters = []
    for cluster in current_clusters:
        prior_cluster = cluster.copy()
        prior_cluster['cluster_severity'] = min(100, cluster['cluster_severity'] + 20)  # Increase severity
        prior_cluster['wave_count'] = cluster['wave_count'] + 1  # Increase wave count
        prior_clusters.append(prior_cluster)
    
    changes = detect_cluster_changes(current_clusters, prior_clusters)
    
    # Should detect improving changes
    improving_changes = [c for c in changes if c['change_type'] == 'improving']
    
    print(f"✓ Detected {len(improving_changes)} improving clusters")
    
    for change in improving_changes:
        assert change['severity_change'] < 0 or change['wave_count_change'] < 0
        print(f"✓ Improving: {change['cluster_name']} (severity {change['severity_change']}, waves {change['wave_count_change']})")
    
    print("✓ Improving cluster detection tests passed")


def test_detect_cluster_changes_resolved():
    """Test detection of resolved clusters"""
    print("\n=== Testing resolved cluster detection ===")
    
    truth_df = create_sample_truth_frame()
    signals = detect_learning_signals(truth_df)
    current_clusters = cluster_signals(signals, truth_df)
    
    # Create prior clusters with extra cluster that doesn't exist anymore
    prior_clusters = current_clusters.copy()
    prior_clusters.append({
        'cluster_type': 'fake_cluster',
        'cluster_name': 'Fake Cluster',
        'cluster_severity': 50,
        'wave_count': 2
    })
    
    changes = detect_cluster_changes(current_clusters, prior_clusters)
    
    # Should detect resolved cluster
    resolved_changes = [c for c in changes if c['change_type'] == 'resolved']
    
    print(f"✓ Detected {len(resolved_changes)} resolved clusters")
    
    assert len(resolved_changes) >= 1, "Should detect at least one resolved cluster"
    
    for change in resolved_changes:
        assert change['severity_change'] < 0
        assert change['wave_count_change'] < 0
        print(f"✓ Resolved: {change['cluster_name']}")
    
    print("✓ Resolved cluster detection tests passed")


def test_get_priority_insights():
    """Test priority insights generation"""
    print("\n=== Testing get_priority_insights() ===")
    
    truth_df = create_sample_truth_frame()
    signals = detect_learning_signals(truth_df)
    clusters = cluster_signals(signals, truth_df)
    
    # Get priority insights
    insights = get_priority_insights(clusters)
    
    print(f"✓ Generated {len(insights)} priority insights")
    
    # Should have up to 3 insights
    assert len(insights) <= 3, "Should return at most 3 insights"
    assert len(insights) > 0, "Should return at least 1 insight if clusters exist"
    
    # Validate insight structure
    for i, insight in enumerate(insights):
        assert insight['rank'] == i + 1, f"Insight rank should be {i+1}"
        assert 'cluster_type' in insight
        assert 'cluster_name' in insight
        assert 'cluster_severity' in insight
        assert 'wave_count' in insight
        assert 'priority_score' in insight
        assert 'narrative' in insight
        assert 'justification' in insight
        
        print(f"✓ Insight #{insight['rank']}: {insight['cluster_name']} (priority score: {insight['priority_score']:.2f})")
        print(f"  Justification: {insight['justification']}")
    
    # Check that insights are sorted by priority score
    priority_scores = [i['priority_score'] for i in insights]
    assert priority_scores == sorted(priority_scores, reverse=True), \
        "Insights should be sorted by priority score (descending)"
    
    print("✓ get_priority_insights() tests passed")


def test_priority_scoring():
    """Test priority score calculation"""
    print("\n=== Testing priority score calculation ===")
    
    # Test different cluster scenarios
    test_cases = [
        {
            'name': 'High severity, high wave count',
            'cluster': {
                'cluster_type': 'regime_mismatch',
                'cluster_severity': 90,
                'wave_count': 8,
                'persistence': 0.9
            },
            'expected_min': 70  # Should have high priority
        },
        {
            'name': 'Low severity, low wave count',
            'cluster': {
                'cluster_type': 'concentration_risk',
                'cluster_severity': 20,
                'wave_count': 1,
                'persistence': 0.3
            },
            'expected_max': 30  # Should have low priority
        },
        {
            'name': 'Regime mismatch boost',
            'cluster': {
                'cluster_type': 'regime_mismatch',
                'cluster_severity': 50,
                'wave_count': 3,
                'persistence': 0.5
            },
            'expected_min': 30  # Regime mismatch gets bonus
        }
    ]
    
    for case in test_cases:
        score = _calculate_priority_score(case['cluster'])
        print(f"✓ {case['name']}: priority score = {score:.2f}")
        
        if 'expected_min' in case:
            assert score >= case['expected_min'], \
                f"{case['name']} should have priority >= {case['expected_min']}"
        if 'expected_max' in case:
            assert score <= case['expected_max'], \
                f"{case['name']} should have priority <= {case['expected_max']}"
    
    print("✓ Priority scoring tests passed")


def test_deterministic_behavior():
    """Test that Stage 3 produces deterministic, reproducible results"""
    print("\n=== Testing deterministic behavior ===")
    
    truth_df = create_sample_truth_frame()
    
    # Run clustering twice
    signals1 = detect_learning_signals(truth_df)
    clusters1 = cluster_signals(signals1, truth_df)
    insights1 = get_priority_insights(clusters1)
    
    signals2 = detect_learning_signals(truth_df)
    clusters2 = cluster_signals(signals2, truth_df)
    insights2 = get_priority_insights(clusters2)
    
    # Results should be identical
    assert len(clusters1) == len(clusters2), "Cluster count should be deterministic"
    assert len(insights1) == len(insights2), "Insight count should be deterministic"
    
    for c1, c2 in zip(clusters1, clusters2):
        assert c1['cluster_type'] == c2['cluster_type']
        assert c1['cluster_severity'] == c2['cluster_severity']
        assert c1['wave_count'] == c2['wave_count']
        assert c1['narrative'] == c2['narrative']
    
    for i1, i2 in zip(insights1, insights2):
        assert i1['rank'] == i2['rank']
        assert i1['priority_score'] == i2['priority_score']
    
    print("✓ Deterministic behavior verified - results are reproducible")
    print("✓ Deterministic behavior tests passed")


def test_read_only_compliance():
    """Test that Stage 3 functions never modify TruthFrame"""
    print("\n=== Testing read-only compliance ===")
    
    truth_df = create_sample_truth_frame()
    
    # Create a copy to compare
    truth_df_copy = truth_df.copy()
    
    # Run all Stage 3 functions
    signals = detect_learning_signals(truth_df)
    clusters = cluster_signals(signals, truth_df)
    insights = get_priority_insights(clusters)
    snapshot = get_adaptive_intelligence_snapshot(truth_df)
    
    # Verify TruthFrame is unchanged
    assert truth_df.equals(truth_df_copy), "TruthFrame should not be modified"
    
    print("✓ TruthFrame unmodified - read-only compliance verified")
    print("✓ Read-only compliance tests passed")


def test_snapshot_integration():
    """Test integrated snapshot with all Stage 3 features"""
    print("\n=== Testing integrated snapshot ===")
    
    truth_df = create_sample_truth_frame()
    
    # Generate snapshot
    snapshot = get_adaptive_intelligence_snapshot(truth_df)
    
    # Validate snapshot structure
    assert 'wave_health' in snapshot
    assert 'regime_intelligence' in snapshot
    assert 'learning_signals' in snapshot
    assert 'signal_clusters' in snapshot  # STAGE 3
    assert 'cluster_changes' in snapshot  # STAGE 3
    assert 'priority_insights' in snapshot  # STAGE 3
    assert 'timestamp' in snapshot
    
    print(f"✓ Snapshot contains all Stage 3 features")
    print(f"  - {len(snapshot['signal_clusters'])} clusters")
    print(f"  - {len(snapshot['cluster_changes'])} changes")
    print(f"  - {len(snapshot['priority_insights'])} priority insights")
    
    # Test snapshot comparison
    prior_snapshot = snapshot.copy()
    current_snapshot = get_adaptive_intelligence_snapshot(truth_df, prior_snapshot)
    
    # Changes should detect stable state
    changes = current_snapshot['cluster_changes']
    print(f"✓ Change detection working: {len(changes)} changes detected")
    
    print("✓ Integrated snapshot tests passed")


def run_all_tests():
    """Run all Stage 3 tests"""
    print("\n" + "="*70)
    print("STAGE 3 ADAPTIVE INTELLIGENCE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    test_functions = [
        test_cluster_signals,
        test_cluster_types,
        test_cluster_narratives,
        test_detect_cluster_changes_new,
        test_detect_cluster_changes_escalating,
        test_detect_cluster_changes_improving,
        test_detect_cluster_changes_resolved,
        test_get_priority_insights,
        test_priority_scoring,
        test_deterministic_behavior,
        test_read_only_compliance,
        test_snapshot_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n✅ ALL STAGE 3 TESTS PASSED!")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
