#!/usr/bin/env python3
"""
Test suite for alpha attribution integration.

Tests the new alpha attribution functionality added to vector_truth.py and app.py.
"""

import pandas as pd
import numpy as np
from vector_truth import (
    build_vector_truth_report,
    extract_alpha_attribution_breakdown,
    VectorTruthReport,
    VectorAlphaSources,
    VectorAlphaReconciliation,
    VectorRegimeAttribution,
    VectorDurabilityScan
)


def test_extract_alpha_attribution_breakdown():
    """Test extract_alpha_attribution_breakdown function."""
    print("Testing extract_alpha_attribution_breakdown...")
    
    # Create a mock report
    sources = VectorAlphaSources(
        total_excess_return=0.15,
        security_selection_alpha=0.08,
        exposure_management_alpha=0.03,
        capital_preservation_effect=0.02,
        benchmark_construction_effect=0.02,
        assessment='Test assessment'
    )
    
    reconciliation = VectorAlphaReconciliation(
        capital_weighted_alpha=0.15,
        exposure_adjusted_alpha=0.16,
        explanation='Test explanation',
        conclusion='Test conclusion',
        inflation_risk='LOW'
    )
    
    regime = VectorRegimeAttribution(
        alpha_risk_on=0.10,
        alpha_risk_off=0.05,
        volatility_sensitivity='Balanced',
        flag='Test flag'
    )
    
    durability = VectorDurabilityScan(
        alpha_type='Test',
        fragility_score=0.3,
        primary_risk='Test risk',
        verdict='Test verdict'
    )
    
    report = VectorTruthReport(
        wave_name='TestWave',
        timeframe_label='365D',
        sources=sources,
        reconciliation=reconciliation,
        regime=regime,
        durability=durability
    )
    
    breakdown = extract_alpha_attribution_breakdown(report)
    
    # Verify all required keys are present
    required_keys = [
        'exposure_timing', 'vix_regime_overlays', 'asset_selection',
        'total_excess', 'residual_strategy', 'benchmark_construction',
        'risk_on_alpha', 'risk_off_alpha', 'wave_name', 'timeframe',
        'assessment', 'regime_sensitivity'
    ]
    
    for key in required_keys:
        assert key in breakdown, f"Missing key: {key}"
    
    # Verify values match expected
    assert breakdown['exposure_timing'] == 0.03
    assert breakdown['vix_regime_overlays'] == 0.02
    assert breakdown['asset_selection'] == 0.08
    assert breakdown['total_excess'] == 0.15
    assert abs(breakdown['residual_strategy'] - 0.11) < 0.001  # 0.15 - 0.02 - 0.02
    assert breakdown['risk_on_alpha'] == 0.10
    assert breakdown['risk_off_alpha'] == 0.05
    assert breakdown['wave_name'] == 'TestWave'
    
    print("✓ extract_alpha_attribution_breakdown test passed!")


def test_build_vector_truth_report_with_overlays():
    """Test build_vector_truth_report with overlay contributions."""
    print("\nTesting build_vector_truth_report with overlay contributions...")
    
    # Simulate a simple wave history
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    wave_ret = pd.Series(np.random.normal(0.0005, 0.01, 365), index=dates)
    bm_ret = pd.Series(np.random.normal(0.0003, 0.01, 365), index=dates)
    
    # Calculate alpha series
    alpha_series = (wave_ret - bm_ret).values.tolist()
    
    # Create regime series (RISK_ON when bm_ret >= 0)
    regime_series = ['RISK_ON' if r >= 0 else 'RISK_OFF' for r in bm_ret.values]
    
    # Build report with overlay contributions
    cap_alpha = (wave_ret.sum() - bm_ret.sum())
    exp_adj_alpha = cap_alpha / 0.85  # Assuming 85% average exposure
    
    # Simulated overlay contributions
    overlay_contribution = 0.015
    vix_contribution = 0.008
    smartsafe_contribution = 0.007
    
    report = build_vector_truth_report(
        wave_name="Test Wave",
        timeframe_label="365D window",
        total_excess_return=cap_alpha,
        capital_weighted_alpha=cap_alpha,
        exposure_adjusted_alpha=exp_adj_alpha,
        alpha_series=alpha_series,
        regime_series=regime_series,
        overlay_contribution=overlay_contribution,
        vix_contribution=vix_contribution,
        smartsafe_contribution=smartsafe_contribution,
        benchmark_snapshot_id="BM-TEST-001",
        benchmark_drift_status="stable"
    )
    
    # Verify report was created
    assert report is not None
    assert report.wave_name == "Test Wave"
    assert report.sources.total_excess_return is not None
    assert report.sources.capital_preservation_effect is not None
    
    # Verify overlay contributions were processed
    # SmartSafe contribution should be used for capital preservation
    assert report.sources.capital_preservation_effect == smartsafe_contribution
    
    print("✓ build_vector_truth_report with overlays test passed!")


def test_attribution_breakdown_completeness():
    """Test that attribution breakdown provides all required sources."""
    print("\nTesting attribution breakdown completeness...")
    
    # Create a comprehensive report
    sources = VectorAlphaSources(
        total_excess_return=0.25,
        security_selection_alpha=0.12,
        exposure_management_alpha=0.05,
        capital_preservation_effect=0.04,
        benchmark_construction_effect=0.04,
        assessment='Comprehensive test assessment'
    )
    
    reconciliation = VectorAlphaReconciliation(
        capital_weighted_alpha=0.25,
        exposure_adjusted_alpha=0.28,
        explanation='Test explanation',
        conclusion='Test conclusion',
        inflation_risk='MODERATE'
    )
    
    regime = VectorRegimeAttribution(
        alpha_risk_on=0.18,
        alpha_risk_off=0.07,
        volatility_sensitivity='Risk-On biased',
        flag='Monitor regime shifts'
    )
    
    durability = VectorDurabilityScan(
        alpha_type='Residual Strategy',
        fragility_score=0.45,
        primary_risk='Regime concentration',
        verdict='Monitor durability'
    )
    
    report = VectorTruthReport(
        wave_name='Comprehensive Test Wave',
        timeframe_label='365D window',
        sources=sources,
        reconciliation=reconciliation,
        regime=regime,
        durability=durability
    )
    
    breakdown = extract_alpha_attribution_breakdown(report)
    
    # Verify all attribution sources are surfaced as per requirements:
    # 1. Exposure & Timing attribution
    assert 'exposure_timing' in breakdown
    assert breakdown['exposure_timing'] == 0.05
    
    # 2. VIX/Regime overlays
    assert 'vix_regime_overlays' in breakdown
    assert breakdown['vix_regime_overlays'] == 0.04
    
    # 3. Asset selection alpha
    assert 'asset_selection' in breakdown
    assert breakdown['asset_selection'] == 0.12
    
    # 4. Total excess and residual strategy
    assert 'total_excess' in breakdown
    assert 'residual_strategy' in breakdown
    
    # 5. Regime attribution (risk-on/off)
    assert 'risk_on_alpha' in breakdown
    assert 'risk_off_alpha' in breakdown
    assert breakdown['risk_on_alpha'] == 0.18
    assert breakdown['risk_off_alpha'] == 0.07
    
    # Verify assessment and metadata
    assert 'assessment' in breakdown
    assert 'regime_sensitivity' in breakdown
    assert 'wave_name' in breakdown
    assert 'timeframe' in breakdown
    
    print("✓ Attribution breakdown completeness test passed!")
    print(f"  All {len(breakdown)} attribution components verified:")
    for key in sorted(breakdown.keys()):
        print(f"    - {key}")


if __name__ == '__main__':
    print("=" * 80)
    print("Alpha Attribution Integration Test Suite")
    print("=" * 80)
    
    try:
        test_extract_alpha_attribution_breakdown()
        test_build_vector_truth_report_with_overlays()
        test_attribution_breakdown_completeness()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ extract_alpha_attribution_breakdown works correctly")
        print("  ✓ build_vector_truth_report handles overlay contributions")
        print("  ✓ All required attribution sources are surfaced:")
        print("    - Exposure & Timing attribution")
        print("    - VIX/Regime overlay contributions")
        print("    - Asset selection alpha")
        print("    - Risk control impacts (via capital preservation)")
        print("    - Momentum (integrated into exposure management)")
        print("\n  Alpha attribution integration complete! ✨")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise
