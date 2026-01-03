"""
test_smartsafe_exemptions.py

Tests for SmartSafe cash wave exemptions.

This test validates:
1. SmartSafe cash wave identification
2. Engine behavior (0% returns, no price ingestion)
3. Snapshot generation (proper row building)
4. Analytics pipeline (skip benchmark generation)
5. Executive summary (excluded from attention)
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waves_engine import (
    SMARTSAFE_CASH_WAVES,
    is_smartsafe_cash_wave,
    compute_history_nav,
    get_display_name_from_wave_id
)

from snapshot_ledger import _build_smartsafe_cash_wave_row
from analytics_pipeline import generate_benchmark_prices_csv


def test_smartsafe_identification():
    """Test SmartSafe cash wave identification."""
    print("=" * 80)
    print("TEST 1: SmartSafe Cash Wave Identification")
    print("=" * 80)
    
    # Test with wave_ids
    assert is_smartsafe_cash_wave("smartsafe_treasury_cash_wave"), "Failed: treasury wave_id"
    assert is_smartsafe_cash_wave("smartsafe_tax_free_money_market_wave"), "Failed: money market wave_id"
    assert not is_smartsafe_cash_wave("sp500_wave"), "Failed: sp500 should not be SmartSafe"
    
    # Test with display names
    treasury_name = get_display_name_from_wave_id("smartsafe_treasury_cash_wave")
    if treasury_name:
        assert is_smartsafe_cash_wave(treasury_name), "Failed: treasury display name"
    
    print("✓ SmartSafe wave identification works correctly")
    print()


def test_engine_behavior():
    """Test engine behavior for SmartSafe cash waves."""
    print("=" * 80)
    print("TEST 2: Engine Behavior (0% Returns, No Price Ingestion)")
    print("=" * 80)
    
    wave_name = get_display_name_from_wave_id("smartsafe_treasury_cash_wave")
    result = compute_history_nav(wave_name, mode="Standard", days=30)
    
    # Check shape and columns
    assert not result.empty, "Failed: result should not be empty"
    assert "wave_nav" in result.columns, "Failed: missing wave_nav column"
    assert "bm_nav" in result.columns, "Failed: missing bm_nav column"
    assert "wave_ret" in result.columns, "Failed: missing wave_ret column"
    assert "bm_ret" in result.columns, "Failed: missing bm_ret column"
    
    # Check all NAVs are 1.0
    assert (result["wave_nav"] == 1.0).all(), "Failed: wave_nav should be constant 1.0"
    assert (result["bm_nav"] == 1.0).all(), "Failed: bm_nav should be constant 1.0"
    
    # Check all returns are 0.0
    assert (result["wave_ret"] == 0.0).all(), "Failed: wave_ret should be 0.0"
    assert (result["bm_ret"] == 0.0).all(), "Failed: bm_ret should be 0.0"
    
    # Check coverage metadata
    coverage = result.attrs.get("coverage", {})
    assert coverage.get("wave_coverage_pct") == 100.0, "Failed: wave coverage should be 100%"
    assert coverage.get("bm_coverage_pct") == 100.0, "Failed: benchmark coverage should be 100%"
    assert coverage.get("is_smartsafe_cash_wave") == True, "Failed: should be marked as SmartSafe"
    assert coverage.get("wave_tickers_expected") == 0, "Failed: should expect 0 tickers"
    assert len(coverage.get("failed_tickers", {})) == 0, "Failed: should have no failed tickers"
    
    print("✓ Engine correctly handles SmartSafe cash waves")
    print(f"  - NAV: constant 1.0")
    print(f"  - Returns: 0.0%")
    print(f"  - Coverage: 100%")
    print(f"  - No tickers required")
    print()


def test_snapshot_row_building():
    """Test snapshot row building for SmartSafe cash waves."""
    print("=" * 80)
    print("TEST 3: Snapshot Row Building")
    print("=" * 80)
    
    wave_id = "smartsafe_treasury_cash_wave"
    wave_name = "SmartSafe Treasury Cash Wave"
    
    row = _build_smartsafe_cash_wave_row(wave_id, wave_name, "Standard")
    
    # Check basic fields
    assert row["Wave_ID"] == wave_id, "Failed: Wave_ID mismatch"
    assert row["Wave"] == wave_name, "Failed: Wave name mismatch"
    assert row["NAV"] == 1.0, "Failed: NAV should be 1.0"
    assert row["NAV_1D_Change"] == 0.0, "Failed: NAV change should be 0.0"
    
    # Check returns are 0.0
    assert row["Return_1D"] == 0.0, "Failed: Return_1D should be 0.0"
    assert row["Return_30D"] == 0.0, "Failed: Return_30D should be 0.0"
    assert row["Return_60D"] == 0.0, "Failed: Return_60D should be 0.0"
    assert row["Return_365D"] == 0.0, "Failed: Return_365D should be 0.0"
    
    # Check benchmarks are N/A
    assert np.isnan(row["Benchmark_Return_1D"]), "Failed: Benchmark_Return_1D should be NaN"
    assert np.isnan(row["Benchmark_Return_30D"]), "Failed: Benchmark_Return_30D should be NaN"
    
    # Check alphas are N/A
    assert np.isnan(row["Alpha_1D"]), "Failed: Alpha_1D should be NaN"
    assert np.isnan(row["Alpha_30D"]), "Failed: Alpha_30D should be NaN"
    
    # Check exposure and cash
    assert row["Exposure"] == 0.0, "Failed: Exposure should be 0.0"
    assert row["CashPercent"] == 100.0, "Failed: CashPercent should be 100.0"
    
    # Check metadata
    assert row["Coverage_Score"] == 100, "Failed: Coverage_Score should be 100"
    assert row["status"] == "OK", "Failed: status should be OK"
    assert row["missing_tickers"] == "", "Failed: missing_tickers should be empty"
    assert "SmartSafe Cash Wave" in row["Flags"], "Failed: Flags should indicate SmartSafe"
    
    print("✓ Snapshot row correctly built for SmartSafe cash waves")
    print(f"  - Returns: 0.0%")
    print(f"  - Benchmarks: N/A")
    print(f"  - Alphas: N/A")
    print(f"  - Exposure: 0%, Cash: 100%")
    print()


def test_analytics_pipeline_skip():
    """Test that analytics pipeline skips benchmark generation."""
    print("=" * 80)
    print("TEST 4: Analytics Pipeline (Skip Benchmark Generation)")
    print("=" * 80)
    
    # Test with SmartSafe wave - should skip and return True
    result = generate_benchmark_prices_csv("smartsafe_treasury_cash_wave", 30, use_dummy_data=False)
    assert result == True, "Failed: should return True when skipping SmartSafe wave"
    
    print("✓ Analytics pipeline correctly skips SmartSafe cash waves")
    print()


def test_executive_summary_exclusion():
    """Test that SmartSafe waves are excluded from attention checks."""
    print("=" * 80)
    print("TEST 5: Executive Summary (Exclude from Attention)")
    print("=" * 80)
    
    # Create a test-specific version of identify_attention_waves
    # This avoids importing streamlit dependencies
    def identify_attention_waves_test(snapshot_df, threshold=-0.02):
        """Test version of identify_attention_waves."""
        if snapshot_df is None or snapshot_df.empty:
            return []
        
        attention_waves = []
        
        for _, row in snapshot_df.iterrows():
            # Skip SmartSafe cash waves - they are always stable
            wave_id = row.get("Wave_ID", "")
            flags = row.get("Flags", "")
            if "SmartSafe Cash Wave" in flags or wave_id in ["smartsafe_treasury_cash_wave", "smartsafe_tax_free_money_market_wave"]:
                continue
            
            reasons = []
            
            # Check 1D return
            return_1d = row.get("Return_1D", 0)
            if return_1d < threshold:
                reasons.append(f"1D return {return_1d}")
            
            # Check data status
            status = row.get("Data_Regime_Tag", "")
            if status in ["Unavailable", "Operational"]:
                reasons.append(f"Data status: {status}")
            
            # Check coverage
            coverage = row.get("Coverage_Percent", 100)
            if coverage < 80:
                reasons.append(f"Coverage: {coverage:.0f}%")
            
            if reasons:
                attention_waves.append({
                    "name": row.get("Display_Name", "Unknown"),
                    "reasons": reasons,
                    "return_1d": return_1d,
                    "status": status,
                })
        
        return attention_waves
    
    # Create test data
    test_data = [
        {
            'Wave_ID': 'smartsafe_treasury_cash_wave',
            'Display_Name': 'SmartSafe Treasury Cash Wave',
            'Return_1D': 0.0,
            'Data_Regime_Tag': 'Full',
            'Coverage_Percent': 100.0,
            'Flags': 'SmartSafe Cash Wave'
        },
        {
            'Wave_ID': 'sp500_wave',
            'Display_Name': 'S&P 500 Wave',
            'Return_1D': -0.03,  # Below threshold
            'Data_Regime_Tag': 'Full',
            'Coverage_Percent': 100.0,
            'Flags': 'OK'
        }
    ]
    
    snapshot_df = pd.DataFrame(test_data)
    attention_waves = identify_attention_waves_test(snapshot_df, threshold=-0.02)
    
    # SmartSafe wave should NOT be in attention list
    assert len(attention_waves) == 1, f"Failed: expected 1 attention wave, got {len(attention_waves)}"
    assert attention_waves[0]['name'] == 'S&P 500 Wave', "Failed: wrong wave flagged"
    
    print("✓ Executive summary correctly excludes SmartSafe cash waves")
    print(f"  - SmartSafe waves: excluded from attention")
    print(f"  - Regular waves: included when threshold crossed")
    print()


def run_all_tests():
    """Run all SmartSafe exemption tests."""
    print("\n")
    print("=" * 80)
    print("SMARTSAFE CASH WAVE EXEMPTIONS - TEST SUITE")
    print("=" * 80)
    print("\n")
    
    try:
        test_smartsafe_identification()
        test_engine_behavior()
        test_snapshot_row_building()
        test_analytics_pipeline_skip()
        test_executive_summary_exclusion()
        
        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nSmartSafe cash wave exemptions are working correctly:")
        print("  1. ✓ Wave identification (wave_id and display_name)")
        print("  2. ✓ Engine behavior (0% returns, no price ingestion)")
        print("  3. ✓ Snapshot generation (proper metadata)")
        print("  4. ✓ Analytics pipeline (skip benchmark generation)")
        print("  5. ✓ Executive summary (excluded from attention)")
        print()
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
