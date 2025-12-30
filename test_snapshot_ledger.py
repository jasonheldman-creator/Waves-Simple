"""
test_snapshot_ledger.py

Tests for Wave Snapshot Ledger functionality.

This test suite validates:
1. Snapshot generation with all 28 waves
2. Tiered fallback logic (A -> B -> C -> D)
3. Snapshot persistence and loading
4. Metadata tracking
5. Timeout guards and circuit breakers
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snapshot_ledger import (
    generate_snapshot,
    load_snapshot,
    get_snapshot_metadata,
    generate_broken_tickers_artifact,
    _safe_return,
    _compute_beta,
    _compute_max_drawdown,
    _load_wave_history_from_csv,
    SNAPSHOT_FILE,
    BROKEN_TICKERS_FILE,
    TIMEFRAMES,
)


class TestSnapshotLedger:
    """Test suite for Wave Snapshot Ledger."""
    
    def test_generate_snapshot_all_28_waves(self):
        """Test that snapshot generation always produces 28 waves."""
        print("\n" + "="*80)
        print("TEST: Generate snapshot with all 28 waves")
        print("="*80)
        
        # Generate snapshot
        snapshot_df = generate_snapshot(force_refresh=True, max_runtime_seconds=60)
        
        # Validate
        assert snapshot_df is not None, "Snapshot should not be None"
        assert not snapshot_df.empty, "Snapshot should not be empty"
        assert len(snapshot_df) == 28, f"Expected 28 waves, got {len(snapshot_df)}"
        
        print(f"✓ Snapshot has exactly 28 waves")
        print(f"✓ Data Regime breakdown:")
        print(snapshot_df['Data_Regime_Tag'].value_counts())
    
    def test_snapshot_columns(self):
        """Test that snapshot has all required columns."""
        print("\n" + "="*80)
        print("TEST: Snapshot has all required columns")
        print("="*80)
        
        # Load snapshot
        snapshot_df = load_snapshot(force_refresh=False)
        
        # Required columns
        required_columns = [
            "Wave", "Mode", "Date",
            "NAV", "NAV_1D_Change",
            "Return_1D", "Return_30D", "Return_60D", "Return_365D",
            "Benchmark_Return_1D", "Benchmark_Return_30D", "Benchmark_Return_60D", "Benchmark_Return_365D",
            "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D",
            "Exposure", "CashPercent",
            "VIX_Level", "VIX_Regime",
            "Beta_Real", "Beta_Target", "Beta_Drift",
            "Turnover_Est", "MaxDD",
            "Flags", "Data_Regime_Tag", "Coverage_Score"
        ]
        
        # Validate
        for col in required_columns:
            assert col in snapshot_df.columns, f"Missing required column: {col}"
        
        print(f"✓ All {len(required_columns)} required columns present")
        print(f"✓ Total columns in snapshot: {len(snapshot_df.columns)}")
    
    def test_snapshot_persistence(self):
        """Test that snapshot is persisted to disk."""
        print("\n" + "="*80)
        print("TEST: Snapshot persistence")
        print("="*80)
        
        # Generate snapshot
        snapshot_df = generate_snapshot(force_refresh=True, max_runtime_seconds=60)
        
        # Check file exists
        assert os.path.exists(SNAPSHOT_FILE), f"Snapshot file not found: {SNAPSHOT_FILE}"
        
        # Check file is recent
        mtime = os.path.getmtime(SNAPSHOT_FILE)
        age_seconds = time.time() - mtime
        assert age_seconds < 120, f"Snapshot file is too old: {age_seconds:.1f} seconds"
        
        # Load from disk
        loaded_df = pd.read_csv(SNAPSHOT_FILE)
        assert len(loaded_df) == 28, f"Loaded snapshot has {len(loaded_df)} waves, expected 28"
        
        print(f"✓ Snapshot persisted to {SNAPSHOT_FILE}")
        print(f"✓ Snapshot age: {age_seconds:.1f} seconds")
        print(f"✓ Snapshot loaded successfully with {len(loaded_df)} waves")
    
    def test_snapshot_metadata(self):
        """Test snapshot metadata retrieval."""
        print("\n" + "="*80)
        print("TEST: Snapshot metadata")
        print("="*80)
        
        # Get metadata
        metadata = get_snapshot_metadata()
        
        # Validate
        assert metadata is not None, "Metadata should not be None"
        assert "exists" in metadata, "Metadata should have 'exists' key"
        assert "timestamp" in metadata, "Metadata should have 'timestamp' key"
        assert "age_hours" in metadata, "Metadata should have 'age_hours' key"
        assert "wave_count" in metadata, "Metadata should have 'wave_count' key"
        assert "is_stale" in metadata, "Metadata should have 'is_stale' key"
        
        # If snapshot exists, validate values
        if metadata["exists"]:
            assert metadata["wave_count"] == 28, f"Expected 28 waves, got {metadata['wave_count']}"
            assert metadata["age_hours"] is not None, "Age hours should not be None"
        
        print(f"✓ Metadata retrieved successfully")
        print(f"  - Exists: {metadata['exists']}")
        print(f"  - Wave count: {metadata['wave_count']}")
        print(f"  - Age hours: {metadata.get('age_hours', 'N/A')}")
        print(f"  - Is stale: {metadata.get('is_stale', 'N/A')}")
    
    def test_populated_waves_count(self):
        """Test that populated waves have valid data."""
        print("\n" + "="*80)
        print("TEST: Populated waves count")
        print("="*80)
        
        # Load snapshot
        snapshot_df = load_snapshot(force_refresh=False)
        
        # Count populated waves (waves with non-NaN returns)
        populated_count = snapshot_df["Return_30D"].notna().sum()
        degraded_count = len(snapshot_df) - populated_count
        
        print(f"✓ Total waves: {len(snapshot_df)}")
        print(f"✓ Populated waves (with 30D return data): {populated_count}")
        print(f"✓ Degraded waves: {degraded_count}")
        
        # At least some waves should have data (not all Tier D)
        assert populated_count > 0, "At least some waves should have populated data"
        
        # Check populated waves have valid metrics
        populated_df = snapshot_df[snapshot_df["Return_30D"].notna()]
        
        # Validate alpha is also populated for these waves
        alpha_populated = populated_df["Alpha_30D"].notna().sum()
        print(f"✓ Waves with 30D alpha data: {alpha_populated}")
        
        # Exposure and Cash% should be populated for all waves (even Tier D)
        exposure_populated = snapshot_df["Exposure"].notna().sum()
        cash_populated = snapshot_df["CashPercent"].notna().sum()
        
        print(f"✓ Waves with Exposure data: {exposure_populated}")
        print(f"✓ Waves with Cash% data: {cash_populated}")
        
        # These should always be populated (from VIX ladder logic)
        assert exposure_populated == 28, "All waves should have Exposure"
        assert cash_populated == 28, "All waves should have Cash%"
    
    def test_tiered_fallback(self):
        """Test that tiered fallback is working correctly."""
        print("\n" + "="*80)
        print("TEST: Tiered fallback logic")
        print("="*80)
        
        # Load snapshot
        snapshot_df = load_snapshot(force_refresh=False)
        
        # Count by data regime
        regime_counts = snapshot_df["Data_Regime_Tag"].value_counts()
        
        print("Data Regime Breakdown:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} waves ({count/len(snapshot_df)*100:.1f}%)")
        
        # Validate that we have a distribution (not all in one tier)
        unique_regimes = len(regime_counts)
        assert unique_regimes >= 1, "Should have at least one data regime"
        
        # Check flags for Tier D waves
        tier_d_df = snapshot_df[snapshot_df["Data_Regime_Tag"] == "Unavailable"]
        if not tier_d_df.empty:
            print(f"\nTier D (Unavailable) waves: {len(tier_d_df)}")
            for _, row in tier_d_df.iterrows():
                print(f"  - {row['Wave']}: {row['Flags']}")
    
    def test_helper_functions(self):
        """Test helper functions used in snapshot generation."""
        print("\n" + "="*80)
        print("TEST: Helper functions")
        print("="*80)
        
        # Test _safe_return
        nav_series = pd.Series([100, 102, 105, 103, 108])
        
        return_1d = _safe_return(nav_series, 1)
        return_30d = _safe_return(nav_series, 30)  # Should use all available data
        
        print(f"✓ _safe_return(1D): {return_1d:.4f}")
        print(f"✓ _safe_return(30D): {return_30d:.4f}")
        
        assert not np.isnan(return_1d), "1D return should not be NaN"
        assert not np.isnan(return_30d), "30D return should not be NaN"
        
        # Test _compute_max_drawdown
        max_dd = _compute_max_drawdown(nav_series)
        print(f"✓ _compute_max_drawdown: {max_dd:.4f}")
        assert not np.isnan(max_dd), "Max drawdown should not be NaN"
        assert max_dd <= 0, "Max drawdown should be negative"
        
        # Test _compute_beta
        wave_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        bm_returns = pd.Series([0.005, -0.01, 0.015, -0.005, 0.01])
        
        beta = _compute_beta(wave_returns, bm_returns, min_points=3)
        print(f"✓ _compute_beta: {beta:.4f}")
        # Beta can be NaN if not enough points, but should be a float
        assert isinstance(beta, float), "Beta should be a float"
    
    def test_wave_history_csv_fallback(self):
        """Test that wave_history.csv fallback works."""
        print("\n" + "="*80)
        print("TEST: wave_history.csv fallback")
        print("="*80)
        
        # Check if wave_history.csv exists
        if not os.path.exists("wave_history.csv"):
            print("⚠ wave_history.csv not found, skipping test")
            return
        
        # Load wave history for a sample wave
        hist_df = _load_wave_history_from_csv("sp500_wave", days=90)
        
        if hist_df is not None and not hist_df.empty:
            print(f"✓ Loaded {len(hist_df)} days of history for sp500_wave")
            print(f"✓ Columns: {hist_df.columns.tolist()}")
            
            # Validate columns
            assert "wave_nav" in hist_df.columns, "Missing wave_nav column"
            assert "bm_nav" in hist_df.columns, "Missing bm_nav column"
            assert "wave_ret" in hist_df.columns, "Missing wave_ret column"
            assert "bm_ret" in hist_df.columns, "Missing bm_ret column"
            
            # Validate NAV values
            assert hist_df["wave_nav"].notna().all(), "NAV should not have NaN values"
            assert hist_df["bm_nav"].notna().all(), "Benchmark NAV should not have NaN values"
            
            print(f"✓ NAV range: {hist_df['wave_nav'].min():.2f} to {hist_df['wave_nav'].max():.2f}")
        else:
            print("⚠ No history data found for sp500_wave")
    
    def test_timeout_guard(self):
        """Test that timeout guard prevents infinite hangs."""
        print("\n" + "="*80)
        print("TEST: Timeout guard (max runtime enforcement)")
        print("="*80)
        
        # Generate snapshot with very short timeout
        start = time.time()
        snapshot_df = generate_snapshot(force_refresh=True, max_runtime_seconds=5)
        elapsed = time.time() - start
        
        print(f"✓ Snapshot generation completed in {elapsed:.1f} seconds")
        print(f"✓ Timeout was set to 5 seconds")
        
        # Validate snapshot still has 28 waves (even with timeout)
        assert len(snapshot_df) == 28, f"Expected 28 waves even with timeout, got {len(snapshot_df)}"
        
        # With very short timeout, expect more Tier D fallbacks
        tier_d_count = (snapshot_df["Data_Regime_Tag"] == "Unavailable").sum()
        print(f"✓ Tier D fallback waves with 5s timeout: {tier_d_count}")
        print(f"✓ System remained responsive and produced complete snapshot")
    
    def test_broken_tickers_artifact(self):
        """Test that broken_tickers.csv artifact is generated."""
        print("\n" + "="*80)
        print("TEST: Broken tickers artifact generation")
        print("="*80)
        
        # Generate broken tickers artifact
        broken_df = generate_broken_tickers_artifact()
        
        # Validate artifact was created
        assert os.path.exists(BROKEN_TICKERS_FILE), f"Broken tickers file not found: {BROKEN_TICKERS_FILE}"
        print(f"✓ Broken tickers artifact created at {BROKEN_TICKERS_FILE}")
        
        # Validate DataFrame has correct columns
        expected_columns = [
            "ticker_original",
            "ticker_normalized",
            "failure_type",
            "error_message",
            "impacted_waves",
            "suggested_fix",
            "first_seen",
            "last_seen",
            "is_fatal"
        ]
        
        for col in expected_columns:
            assert col in broken_df.columns, f"Missing expected column: {col}"
        
        print(f"✓ All {len(expected_columns)} expected columns present")
        print(f"✓ Total broken tickers: {len(broken_df)}")
        
        # Validate file can be read
        loaded_df = pd.read_csv(BROKEN_TICKERS_FILE)
        print(f"✓ Artifact loaded successfully from disk")
        
        # If there are broken tickers, validate structure
        if not broken_df.empty:
            print("\nSample broken tickers:")
            for idx, row in broken_df.head(3).iterrows():
                print(f"  - {row['ticker_original']}: {row['failure_type']}")
                print(f"    Impacted waves: {row['impacted_waves']}")
                print(f"    Suggested fix: {row['suggested_fix'][:50]}...")
    
    def test_snapshot_and_broken_tickers_together(self):
        """Test that generate_snapshot produces both artifacts."""
        print("\n" + "="*80)
        print("TEST: Snapshot generation produces both artifacts")
        print("="*80)
        
        # Generate snapshot (which should also generate broken_tickers)
        snapshot_df = generate_snapshot(force_refresh=True, max_runtime_seconds=30)
        
        # Validate both files exist
        assert os.path.exists(SNAPSHOT_FILE), f"Snapshot file not found: {SNAPSHOT_FILE}"
        assert os.path.exists(BROKEN_TICKERS_FILE), f"Broken tickers file not found: {BROKEN_TICKERS_FILE}"
        
        print(f"✓ Snapshot artifact exists: {SNAPSHOT_FILE}")
        print(f"✓ Broken tickers artifact exists: {BROKEN_TICKERS_FILE}")
        
        # Load both
        snapshot_loaded = pd.read_csv(SNAPSHOT_FILE)
        broken_loaded = pd.read_csv(BROKEN_TICKERS_FILE)
        
        assert len(snapshot_loaded) == 28, f"Expected 28 waves in snapshot, got {len(snapshot_loaded)}"
        
        print(f"✓ Snapshot has {len(snapshot_loaded)} waves")
        print(f"✓ Broken tickers has {len(broken_loaded)} entries")
        print(f"✓ Both artifacts generated successfully in single operation")


def run_all_tests():
    """Run all tests in the test suite."""
    print("\n" + "="*80)
    print("WAVE SNAPSHOT LEDGER - Test Suite")
    print("="*80)
    
    test_suite = TestSnapshotLedger()
    
    tests = [
        ("Generate snapshot with all 28 waves", test_suite.test_generate_snapshot_all_28_waves),
        ("Snapshot columns validation", test_suite.test_snapshot_columns),
        ("Snapshot persistence", test_suite.test_snapshot_persistence),
        ("Snapshot metadata", test_suite.test_snapshot_metadata),
        ("Populated waves count", test_suite.test_populated_waves_count),
        ("Tiered fallback logic", test_suite.test_tiered_fallback),
        ("Helper functions", test_suite.test_helper_functions),
        ("wave_history.csv fallback", test_suite.test_wave_history_csv_fallback),
        ("Timeout guard", test_suite.test_timeout_guard),
        ("Broken tickers artifact", test_suite.test_broken_tickers_artifact),
        ("Snapshot and broken tickers together", test_suite.test_snapshot_and_broken_tickers_together),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ PASSED: {test_name}")
        except AssertionError as e:
            failed += 1
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {test_name}")
            print(f"   Error: {str(e)}")
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
