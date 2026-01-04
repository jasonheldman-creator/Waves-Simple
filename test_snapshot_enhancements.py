"""
test_snapshot_enhancements.py

Unit tests for enhanced snapshot generation with status and missing_tickers columns.
"""

import pandas as pd
import numpy as np
from snapshot_ledger import (
    _load_canonical_waves_from_weights,
    _get_wave_tickers,
    _build_snapshot_row_tier_d,
)


def test_load_canonical_waves():
    """Test loading canonical waves from wave_weights.csv"""
    print("\n=== Testing Canonical Wave Loading ===")
    
    canonical_waves = _load_canonical_waves_from_weights()
    
    assert canonical_waves is not None, "Failed to load canonical waves"
    assert len(canonical_waves) == 28, f"Expected 28 waves, got {len(canonical_waves)}"
    
    # Check structure: should be list of (wave_name, wave_id) tuples
    assert all(len(w) == 2 for w in canonical_waves), "Each wave should be a (name, id) tuple"
    
    # Check some known waves
    wave_names = [w[0] for w in canonical_waves]
    assert "S&P 500 Wave" in wave_names, "S&P 500 Wave should be in canonical list"
    assert "Income Wave" in wave_names, "Income Wave should be in canonical list"
    
    print(f"✓ Loaded {len(canonical_waves)} canonical waves")
    print(f"  Sample waves: {[w[0] for w in canonical_waves[:3]]}")
    

def test_get_wave_tickers():
    """Test getting tickers for a wave"""
    print("\n=== Testing Wave Ticker Retrieval ===")
    
    # Test with a known wave
    tickers = _get_wave_tickers("S&P 500 Wave")
    
    assert tickers is not None, "Failed to get tickers"
    assert len(tickers) > 0, "Should have at least one ticker"
    assert "SPY" in tickers, "S&P 500 Wave should include SPY"
    
    print(f"✓ S&P 500 Wave has {len(tickers)} tickers: {tickers}")


def test_tier_d_row_structure():
    """Test Tier D fallback row has required columns"""
    print("\n=== Testing Tier D Row Structure ===")
    
    wave_id = "test_wave"
    wave_name = "Test Wave"
    mode = "Standard"
    
    row = _build_snapshot_row_tier_d(wave_id, wave_name, mode, None)
    
    # Check required columns exist
    required_columns = [
        "Wave_ID", "Wave", "status", "missing_tickers",
        "Return_1D", "Return_30D", "Return_60D", "Return_365D",
        "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D",
        "Benchmark_Return_1D", "Benchmark_Return_30D", 
        "Benchmark_Return_60D", "Benchmark_Return_365D",
        "NAV", "Exposure", "CashPercent", "Flags",
        "Data_Regime_Tag", "Coverage_Score"
    ]
    
    for col in required_columns:
        assert col in row, f"Required column '{col}' missing from Tier D row"
    
    # Check status is NO DATA for Tier D
    assert row["status"] == "NO DATA", f"Tier D status should be 'NO DATA', got '{row['status']}'"
    
    # Check that metrics are NaN
    assert np.isnan(row["Return_1D"]), "Return_1D should be NaN in Tier D"
    assert np.isnan(row["Alpha_1D"]), "Alpha_1D should be NaN in Tier D"
    assert np.isnan(row["NAV"]), "NAV should be NaN in Tier D"
    
    # Check missing_tickers is populated
    assert isinstance(row["missing_tickers"], str), "missing_tickers should be a string"
    
    print("✓ Tier D row has all required columns")
    print(f"  status: {row['status']}")
    print(f"  missing_tickers: {row['missing_tickers'][:50]}...")
    print(f"  Data_Regime_Tag: {row['Data_Regime_Tag']}")


def test_snapshot_column_requirements():
    """Test that snapshot has required new columns"""
    print("\n=== Testing Snapshot Column Requirements ===")
    
    # Load existing snapshot if available
    try:
        snapshot_df = pd.read_csv("data/live_snapshot.csv")
        
        # Check for new columns
        assert "status" in snapshot_df.columns, "Snapshot missing 'status' column"
        assert "missing_tickers" in snapshot_df.columns, "Snapshot missing 'missing_tickers' column"
        
        # Check status values
        valid_statuses = {"OK", "NO DATA"}
        actual_statuses = set(snapshot_df["status"].unique())
        assert actual_statuses.issubset(valid_statuses), f"Invalid status values: {actual_statuses - valid_statuses}"
        
        # Check row count
        assert len(snapshot_df) == 28, f"Snapshot should have 28 rows, got {len(snapshot_df)}"
        
        print(f"✓ Snapshot has {len(snapshot_df)} rows")
        print(f"  status column: ✓ ({snapshot_df['status'].value_counts().to_dict()})")
        print(f"  missing_tickers column: ✓")
        
        # Show examples of waves with missing tickers
        with_missing = snapshot_df[snapshot_df["missing_tickers"].str.len() > 0]
        if not with_missing.empty:
            print(f"\n  Waves with missing tickers: {len(with_missing)}")
            for idx, row in with_missing.head(3).iterrows():
                print(f"    - {row['Wave']}: {row['missing_tickers'][:50]}...")
        
    except FileNotFoundError:
        print("⚠ data/live_snapshot.csv not found - skipping snapshot validation")
        print("  (Generate snapshot first with snapshot_ledger.generate_snapshot())")


def run_all_tests():
    """Run all snapshot enhancement tests"""
    print("=" * 80)
    print("SNAPSHOT ENHANCEMENT TESTS")
    print("=" * 80)
    
    tests = [
        ("Load Canonical Waves", test_load_canonical_waves),
        ("Get Wave Tickers", test_get_wave_tickers),
        ("Tier D Row Structure", test_tier_d_row_structure),
        ("Snapshot Column Requirements", test_snapshot_column_requirements),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"{'='*80}")
        try:
            test_func()
            print(f"✓ PASSED: {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {test_name}")
            print(f"  Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
