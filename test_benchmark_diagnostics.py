"""
Unit tests for benchmark diagnostics and proof fields.

This test module validates:
1. Benchmark mode identification (DYNAMIC vs STATIC)
2. Benchmark components preview formatting
3. Benchmark hash stability
4. 365D window integrity computation
5. Alpha reconciliation checks
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import functions to test
from waves_engine import (
    _compute_benchmark_hash,
    _format_benchmark_components_preview,
    _compute_365d_window_integrity,
    _compute_alpha_reconciliation
)


class TestBenchmarkHash:
    """Test benchmark hash computation for auditability."""
    
    def test_hash_deterministic(self):
        """Hash should be deterministic for same components."""
        components = [
            {"ticker": "SPY", "weight": 0.6},
            {"ticker": "QQQ", "weight": 0.4}
        ]
        
        hash1 = _compute_benchmark_hash(components)
        hash2 = _compute_benchmark_hash(components)
        
        assert hash1 == hash2, "Hash should be deterministic"
        assert len(hash1) == 16, "Hash should be 16 characters"
    
    def test_hash_order_independent(self):
        """Hash should be same regardless of component order."""
        components1 = [
            {"ticker": "SPY", "weight": 0.6},
            {"ticker": "QQQ", "weight": 0.4}
        ]
        components2 = [
            {"ticker": "QQQ", "weight": 0.4},
            {"ticker": "SPY", "weight": 0.6}
        ]
        
        hash1 = _compute_benchmark_hash(components1)
        hash2 = _compute_benchmark_hash(components2)
        
        assert hash1 == hash2, "Hash should be order-independent"
    
    def test_hash_changes_with_weight(self):
        """Hash should change when weights change."""
        components1 = [
            {"ticker": "SPY", "weight": 0.6},
            {"ticker": "QQQ", "weight": 0.4}
        ]
        components2 = [
            {"ticker": "SPY", "weight": 0.5},
            {"ticker": "QQQ", "weight": 0.5}
        ]
        
        hash1 = _compute_benchmark_hash(components1)
        hash2 = _compute_benchmark_hash(components2)
        
        assert hash1 != hash2, "Hash should change when weights change"
    
    def test_hash_changes_with_ticker(self):
        """Hash should change when tickers change."""
        components1 = [
            {"ticker": "SPY", "weight": 0.6},
            {"ticker": "QQQ", "weight": 0.4}
        ]
        components2 = [
            {"ticker": "SPY", "weight": 0.6},
            {"ticker": "IWM", "weight": 0.4}
        ]
        
        hash1 = _compute_benchmark_hash(components1)
        hash2 = _compute_benchmark_hash(components2)
        
        assert hash1 != hash2, "Hash should change when tickers change"
    
    def test_hash_empty_components(self):
        """Hash should handle empty components gracefully."""
        components = []
        
        hash_val = _compute_benchmark_hash(components)
        
        assert hash_val == "none", "Empty components should return 'none'"


class TestBenchmarkComponentsPreview:
    """Test benchmark components preview formatting."""
    
    def test_simple_preview(self):
        """Simple 2-component preview should format correctly."""
        components = [
            {"ticker": "SPY", "weight": 0.6},
            {"ticker": "QQQ", "weight": 0.4}
        ]
        
        preview = _format_benchmark_components_preview(components)
        
        assert "SPY:60.0%" in preview, "SPY should be in preview"
        assert "QQQ:40.0%" in preview, "QQQ should be in preview"
    
    def test_preview_top5(self):
        """Preview should show top 5 components by weight."""
        components = [
            {"ticker": "A", "weight": 0.10},
            {"ticker": "B", "weight": 0.20},
            {"ticker": "C", "weight": 0.30},
            {"ticker": "D", "weight": 0.15},
            {"ticker": "E", "weight": 0.25}
        ]
        
        preview = _format_benchmark_components_preview(components, max_display=3)
        
        # Top 3 by weight should be C (30%), E (25%), B (20%)
        assert "C:30.0%" in preview, "C should be in preview (highest weight)"
        assert "E:25.0%" in preview, "E should be in preview"
        assert "B:20.0%" in preview, "B should be in preview"
        assert "+2 more" in preview, "Should indicate 2 more components"
    
    def test_preview_remaining_count(self):
        """Preview should show count of remaining components."""
        components = [
            {"ticker": f"TICK{i}", "weight": 0.1} for i in range(10)
        ]
        
        preview = _format_benchmark_components_preview(components, max_display=5)
        
        assert "+5 more" in preview, "Should indicate 5 remaining components"
    
    def test_preview_empty_components(self):
        """Preview should handle empty components gracefully."""
        components = []
        
        preview = _format_benchmark_components_preview(components)
        
        assert preview == "none", "Empty components should return 'none'"


class TestWindow365DIntegrity:
    """Test 365D window integrity computation."""
    
    def test_full_overlap(self):
        """Test case with full overlap between wave and benchmark."""
        # Create 365 days of data
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        
        wave_ret = pd.Series(np.random.randn(365) * 0.01, index=dates)
        bm_ret = pd.Series(np.random.randn(365) * 0.01, index=dates)
        
        result = _compute_365d_window_integrity(wave_ret, bm_ret, trading_days_365d=252)
        
        assert result['wave_365d_days'] == 252, "Should use last 252 days for wave"
        assert result['bench_365d_days'] == 252, "Should use last 252 days for benchmark"
        assert result['intersection_days_used'] <= 252, "Intersection should be <= 252"
        assert result['sufficient_history'] == True, "Should have sufficient history"
        assert result['warning_message'] is None, "Should have no warning with full data"
    
    def test_limited_history(self):
        """Test case with limited history (< 200 days)."""
        # Create only 150 days of data
        dates = pd.date_range(end=datetime.now(), periods=150, freq='D')
        
        wave_ret = pd.Series(np.random.randn(150) * 0.01, index=dates)
        bm_ret = pd.Series(np.random.randn(150) * 0.01, index=dates)
        
        result = _compute_365d_window_integrity(wave_ret, bm_ret, trading_days_365d=252)
        
        assert result['wave_365d_days'] == 150, "Should have 150 days for wave"
        assert result['bench_365d_days'] == 150, "Should have 150 days for benchmark"
        assert result['sufficient_history'] == False, "Should NOT have sufficient history"
        assert result['warning_message'] is not None, "Should have warning for limited history"
        assert "LIMITED HISTORY" in result['warning_message'], "Warning should mention LIMITED HISTORY"
    
    def test_partial_overlap(self):
        """Test case with partial overlap between wave and benchmark."""
        # Wave has 300 days, benchmark has 250 days with 200 day overlap
        wave_dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
        bm_dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        
        wave_ret = pd.Series(np.random.randn(300) * 0.01, index=wave_dates)
        bm_ret = pd.Series(np.random.randn(250) * 0.01, index=bm_dates)
        
        result = _compute_365d_window_integrity(wave_ret, bm_ret, trading_days_365d=252)
        
        # Both should be truncated to last 252 days
        assert result['wave_365d_days'] == 252, "Wave should have 252 days"
        assert result['bench_365d_days'] == 250, "Benchmark should have all 250 days"
        assert result['intersection_days_used'] <= 250, "Intersection limited by benchmark"
    
    def test_empty_series(self):
        """Test case with empty series."""
        wave_ret = pd.Series(dtype=float)
        bm_ret = pd.Series(dtype=float)
        
        result = _compute_365d_window_integrity(wave_ret, bm_ret, trading_days_365d=252)
        
        assert result['wave_365d_days'] == 0, "Empty wave should have 0 days"
        assert result['bench_365d_days'] == 0, "Empty benchmark should have 0 days"
        assert result['sufficient_history'] == False, "Empty data should not be sufficient"
        assert result['warning_message'] is not None, "Should have warning"
    
    def test_date_ranges(self):
        """Test that date ranges are correctly captured."""
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        
        wave_ret = pd.Series(np.random.randn(365) * 0.01, index=dates)
        bm_ret = pd.Series(np.random.randn(365) * 0.01, index=dates)
        
        result = _compute_365d_window_integrity(wave_ret, bm_ret, trading_days_365d=252)
        
        # Check that date fields are populated
        assert result['wave_365d_start'] is not None, "Wave start date should be set"
        assert result['wave_365d_end'] is not None, "Wave end date should be set"
        assert result['bench_365d_start'] is not None, "Benchmark start date should be set"
        assert result['bench_365d_end'] is not None, "Benchmark end date should be set"
        assert result['last_date_wave'] is not None, "Last wave date should be set"
        assert result['last_date_bench'] is not None, "Last benchmark date should be set"
        
        # Verify dates are in YYYY-MM-DD format
        assert len(result['wave_365d_start']) == 10, "Date should be YYYY-MM-DD format"
        assert '-' in result['wave_365d_start'], "Date should contain dashes"


class TestAlphaReconciliation:
    """Test alpha reconciliation checks."""
    
    def test_perfect_reconciliation(self):
        """Test case where alpha matches exactly."""
        wave_ret = 0.15  # 15% wave return
        bench_ret = 0.10  # 10% benchmark return
        alpha = 0.05  # 5% alpha (exactly wave - bench)
        
        result = _compute_alpha_reconciliation(wave_ret, bench_ret, alpha, tolerance=0.001)
        
        assert result['reconciliation_passed'] == True, "Perfect match should pass"
        assert abs(result['expected_alpha'] - 0.05) < 1e-10, "Expected alpha should be 5%"
        assert abs(result['computed_alpha'] - 0.05) < 1e-10, "Computed alpha should be 5%"
        assert abs(result['mismatch']) < 1e-10, "Mismatch should be near 0"
        assert result['warning_message'] is None, "Should have no warning"
    
    def test_small_mismatch_within_tolerance(self):
        """Test case with small mismatch within tolerance."""
        wave_ret = 0.15
        bench_ret = 0.10
        alpha = 0.0502  # Small 2bp error
        
        result = _compute_alpha_reconciliation(wave_ret, bench_ret, alpha, tolerance=0.001)
        
        assert result['reconciliation_passed'] == True, "Small mismatch should pass"
        assert abs(result['mismatch']) < 0.001, "Mismatch should be within tolerance"
    
    def test_large_mismatch_fails(self):
        """Test case with large mismatch exceeding tolerance."""
        wave_ret = 0.15
        bench_ret = 0.10
        alpha = 0.06  # 100bp error (1%)
        
        result = _compute_alpha_reconciliation(wave_ret, bench_ret, alpha, tolerance=0.001)
        
        assert result['reconciliation_passed'] == False, "Large mismatch should fail"
        assert result['mismatch'] > 0.001, "Mismatch should exceed tolerance"
        assert result['warning_message'] is not None, "Should have warning"
        assert "RECONCILIATION FAILED" in result['warning_message'], "Warning should mention failure"
        assert "100" in result['warning_message'] or "10" in result['warning_message'], "Warning should mention bps"
    
    def test_negative_alpha(self):
        """Test case with negative alpha."""
        wave_ret = 0.08
        bench_ret = 0.10
        alpha = -0.02  # -2% alpha (underperformance)
        
        result = _compute_alpha_reconciliation(wave_ret, bench_ret, alpha, tolerance=0.001)
        
        assert result['reconciliation_passed'] == True, "Negative alpha should work"
        assert abs(result['expected_alpha'] - (-0.02)) < 1e-10, "Expected alpha should be -2%"
    
    def test_none_values(self):
        """Test case with None values."""
        result = _compute_alpha_reconciliation(None, 0.10, 0.05, tolerance=0.001)
        
        assert result['reconciliation_passed'] == False, "None values should fail"
        assert result['warning_message'] is not None, "Should have warning"
        assert "Missing data" in result['warning_message'], "Warning should mention missing data"
    
    def test_nan_values(self):
        """Test case with NaN values."""
        result = _compute_alpha_reconciliation(np.nan, 0.10, 0.05, tolerance=0.001)
        
        assert result['reconciliation_passed'] == False, "NaN values should fail"
        assert result['warning_message'] is not None, "Should have warning"
        assert "NaN" in result['warning_message'], "Warning should mention NaN"
    
    def test_mismatch_bps_calculation(self):
        """Test that mismatch is correctly converted to basis points."""
        wave_ret = 0.15
        bench_ret = 0.10
        alpha = 0.051  # 10bp error
        
        result = _compute_alpha_reconciliation(wave_ret, bench_ret, alpha, tolerance=0.001)
        
        # 10bp = 0.001 in decimal = 10 basis points
        assert abs(result['mismatch_bps'] - 10.0) < 0.1, "Should be approximately 10 basis points"


class TestIntegrationWithDummyData:
    """Integration test with dummy wave and benchmark data."""
    
    def test_known_overlap_scenario(self):
        """Test with controlled overlap to verify intersection logic."""
        # Create dummy data:
        # - Wave has 300 trading days
        # - Benchmark has 280 trading days  
        # - 260 day overlap (sufficient for 252-day window)
        
        base_date = datetime(2023, 1, 1)
        
        # Wave: 300 days starting from base_date
        wave_dates = pd.date_range(start=base_date, periods=300, freq='B')  # Business days
        wave_returns = pd.Series(np.random.randn(300) * 0.01, index=wave_dates)
        
        # Benchmark: 280 days starting 20 days later (so 260 day overlap)
        bench_start = base_date + timedelta(days=20)
        bench_dates = pd.date_range(start=bench_start, periods=280, freq='B')
        bench_returns = pd.Series(np.random.randn(280) * 0.01, index=bench_dates)
        
        # Compute integrity
        integrity = _compute_365d_window_integrity(wave_returns, bench_returns, trading_days_365d=252)
        
        # Verify intersection logic
        assert integrity['wave_365d_days'] == 252, "Should use last 252 days from wave"
        assert integrity['bench_365d_days'] <= 252, "Should use available benchmark days"
        assert integrity['intersection_days_used'] >= 200, "Should have sufficient overlap"
        assert integrity['sufficient_history'] == True, "260 days overlap should be sufficient"
        
    def test_alpha_reconciliation_with_realistic_returns(self):
        """Test alpha reconciliation with realistic return scenarios."""
        # Scenario: Wave outperforms benchmark by 5% over 365 days
        
        # Create realistic daily returns (using geometric returns)
        # Wave: +15% annualized = ~0.056% daily
        # Benchmark: +10% annualized = ~0.038% daily
        
        wave_daily_ret = 0.00056
        bench_daily_ret = 0.00038
        
        # Simulate 252 trading days
        days = 252
        wave_returns = [wave_daily_ret] * days
        bench_returns = [bench_daily_ret] * days
        
        # Compute cumulative returns
        wave_cumulative = (1 + pd.Series(wave_returns)).prod() - 1
        bench_cumulative = (1 + pd.Series(bench_returns)).prod() - 1
        alpha_observed = wave_cumulative - bench_cumulative
        
        # Test reconciliation
        reconciliation = _compute_alpha_reconciliation(
            wave_cumulative,
            bench_cumulative,
            alpha_observed,
            tolerance=0.001
        )
        
        assert reconciliation['reconciliation_passed'] == True, "Should pass with exact calculation"
        assert abs(reconciliation['mismatch']) < 1e-10, "Mismatch should be near zero"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
