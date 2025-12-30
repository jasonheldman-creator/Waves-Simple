#!/usr/bin/env python3
"""
ROUND 4 Feature Verification Script

This script tests all ROUND 4 features to ensure they work correctly:
1. Wave Registry Validation
2. Snapshot Table Enhancements
3. Executive Summary Generation
4. Diagnostics Artifact Creation
"""

import sys
import json
from datetime import datetime


def test_wave_registry_validator():
    """Test 1: Wave Registry Validator"""
    print("\n" + "="*80)
    print("TEST 1: Wave Registry Validator")
    print("="*80)
    
    try:
        from helpers.wave_registry_validator import validate_wave_registry, load_wave_registry
        from waves_engine import WAVE_WEIGHTS
        
        # Load registry
        registry = load_wave_registry()
        if not registry:
            print("❌ FAIL: Could not load wave registry")
            return False
        
        print(f"✓ Registry loaded: {registry.get('total_waves', 0)} waves")
        
        # Validate registry
        result = validate_wave_registry(wave_weights=WAVE_WEIGHTS)
        
        print(f"\nValidation Result: {result.get_summary()}")
        print(f"  - Errors: {result.error_count}")
        print(f"  - Warnings: {result.warning_count}")
        print(f"  - Info: {len(result.info)}")
        
        if not result.is_valid:
            print("\n❌ FAIL: Registry validation failed")
            print(result.get_detailed_report())
            return False
        
        # Verify 28 enabled waves
        enabled_count = sum(1 for w in registry['waves'] if w.get('enabled', False))
        if enabled_count != 28:
            print(f"❌ FAIL: Expected 28 enabled waves, got {enabled_count}")
            return False
        
        print(f"✓ Exactly 28 enabled waves")
        
        # Verify all have benchmarks
        no_benchmark = [w['display_name'] for w in registry['waves'] 
                       if not w.get('benchmark_ticker')]
        if no_benchmark:
            print(f"❌ FAIL: Waves missing benchmarks: {no_benchmark}")
            return False
        
        print("✓ All waves have benchmark definitions")
        
        print("\n✅ PASS: Wave Registry Validator")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_snapshot_enhancements():
    """Test 2: Snapshot Table Enhancements"""
    print("\n" + "="*80)
    print("TEST 2: Snapshot Table Enhancements")
    print("="*80)
    
    try:
        import pandas as pd
        from snapshot_ledger import load_snapshot
        
        # Try to load existing snapshot or note if needs generation
        snapshot_df = load_snapshot(force_refresh=False)
        
        if snapshot_df is None or snapshot_df.empty:
            print("⚠️  No snapshot available - would need to generate")
            print("✓ Snapshot loading mechanism works (returns None gracefully)")
            return True
        
        print(f"✓ Snapshot loaded: {len(snapshot_df)} waves")
        
        # Verify required columns exist
        required_columns = [
            'Wave_ID', 'Wave', 'Category',
            'Return_1D', 'Return_30D', 'Return_60D', 'Return_365D',
            'Alpha_1D', 'Alpha_30D', 'Alpha_60D', 'Alpha_365D',
            'Exposure', 'CashPercent',
            'Data_Regime_Tag', 'Coverage_Score'
        ]
        
        missing_cols = [col for col in required_columns if col not in snapshot_df.columns]
        if missing_cols:
            print(f"❌ FAIL: Missing columns: {missing_cols}")
            return False
        
        print(f"✓ All required columns present")
        
        # Verify graded status values
        valid_statuses = {'Full', 'Partial', 'Operational', 'Unavailable'}
        actual_statuses = set(snapshot_df['Data_Regime_Tag'].unique())
        invalid_statuses = actual_statuses - valid_statuses
        
        if invalid_statuses:
            print(f"❌ FAIL: Invalid status values: {invalid_statuses}")
            return False
        
        print(f"✓ Valid status values: {actual_statuses}")
        
        # Verify coverage scores are 0-100
        if 'Coverage_Score' in snapshot_df.columns:
            max_coverage = snapshot_df['Coverage_Score'].max()
            min_coverage = snapshot_df['Coverage_Score'].min()
            
            if max_coverage > 100 or min_coverage < 0:
                print(f"❌ FAIL: Coverage out of range: {min_coverage}-{max_coverage}")
                return False
            
            print(f"✓ Coverage scores in valid range: {min_coverage:.0f}-{max_coverage:.0f}")
        
        # Count status breakdown
        status_counts = snapshot_df['Data_Regime_Tag'].value_counts().to_dict()
        print(f"\n  Status Breakdown:")
        for status in ['Full', 'Partial', 'Operational', 'Unavailable']:
            count = status_counts.get(status, 0)
            pct = (count / len(snapshot_df) * 100) if len(snapshot_df) > 0 else 0
            print(f"    - {status}: {count} ({pct:.0f}%)")
        
        print("\n✅ PASS: Snapshot Table Enhancements")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_executive_summary():
    """Test 3: Executive Summary Generation"""
    print("\n" + "="*80)
    print("TEST 3: Executive Summary Generation")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        from helpers.executive_summary import generate_executive_summary
        
        # Create sample snapshot data
        sample_data = {
            "Wave_ID": ["sp500_wave", "crypto_l1_growth_wave", "gold_wave", "income_wave"],
            "Display_Name": ["S&P 500 Wave", "Crypto L1 Growth Wave", "Gold Wave", "Income Wave"],
            "Category": ["Equity", "Crypto", "Commodity", "Fixed Income"],
            "Return_1D": [0.015, 0.05, -0.01, 0.002],
            "Alpha_1D": [0.005, 0.03, -0.005, 0.001],
            "Data_Regime_Tag": ["Full", "Partial", "Full", "Full"],
            "Coverage_Score": [100, 85, 100, 100],
        }
        
        df = pd.DataFrame(sample_data)
        
        # Generate summary without market data
        summary = generate_executive_summary(df, market_data=None)
        
        if not summary or len(summary) < 100:
            print("❌ FAIL: Summary too short or empty")
            return False
        
        print("✓ Summary generated successfully")
        
        # Verify key sections present
        required_sections = [
            "Executive Summary",
            "Platform Status",
            "Data Coverage",
            "Top Outperformers",
            "Alpha Performance"
        ]
        
        missing_sections = [s for s in required_sections if s not in summary]
        if missing_sections:
            print(f"❌ FAIL: Missing sections: {missing_sections}")
            return False
        
        print(f"✓ All required sections present")
        
        # Generate with market data
        market_data = {
            "VIX": 16.5,
            "SPY_1D": 0.01,
            "QQQ_1D": 0.012,
        }
        
        summary_with_market = generate_executive_summary(df, market_data)
        
        if "Market Regime" not in summary_with_market:
            print("❌ FAIL: Market regime section missing when market data provided")
            return False
        
        print("✓ Market regime section included with market data")
        
        # Print sample summary
        print("\n  Sample Summary (first 500 chars):")
        print("  " + "-" * 76)
        for line in summary[:500].split('\n'):
            print(f"  {line}")
        print("  " + "-" * 76)
        
        print("\n✅ PASS: Executive Summary Generation")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diagnostics_artifact():
    """Test 4: Diagnostics Artifact Creation"""
    print("\n" + "="*80)
    print("TEST 4: Diagnostics Artifact Creation")
    print("="*80)
    
    try:
        import pandas as pd
        from helpers.diagnostics_artifact import generate_diagnostics_artifact, load_diagnostics_artifact
        import os
        
        # Create sample snapshot data
        sample_data = {
            "Wave_ID": ["sp500_wave", "russell_3000_wave", "crypto_l1_growth_wave"],
            "Display_Name": ["S&P 500 Wave", "Russell 3000 Wave", "Crypto L1 Growth Wave"],
            "Data_Regime_Tag": ["Full", "Partial", "Operational"],
        }
        
        df = pd.DataFrame(sample_data)
        
        # Generate diagnostics
        diagnostics = generate_diagnostics_artifact(
            snapshot_df=df,
            broken_tickers=["TICKER1", "TICKER2"],
            failure_reasons={"No data available": 5, "API timeout": 3},
            output_path="/tmp/test_diagnostics.json"
        )
        
        if not diagnostics:
            print("❌ FAIL: No diagnostics generated")
            return False
        
        print("✓ Diagnostics artifact generated")
        
        # Verify required fields
        required_fields = [
            "timestamp",
            "snapshot_build_time",
            "waves_processed",
            "status_counts",
            "top_failure_reasons",
            "broken_tickers",
            "summary"
        ]
        
        missing_fields = [f for f in required_fields if f not in diagnostics]
        if missing_fields:
            print(f"❌ FAIL: Missing fields: {missing_fields}")
            return False
        
        print(f"✓ All required fields present")
        
        # Verify counts
        if diagnostics['waves_processed'] != 3:
            print(f"❌ FAIL: Expected 3 waves, got {diagnostics['waves_processed']}")
            return False
        
        print(f"✓ Wave count correct: {diagnostics['waves_processed']}")
        
        # Verify status counts
        expected_counts = {"Full": 1, "Partial": 1, "Operational": 1, "Unavailable": 0}
        if diagnostics['status_counts'] != expected_counts:
            print(f"❌ FAIL: Status counts mismatch")
            print(f"  Expected: {expected_counts}")
            print(f"  Got: {diagnostics['status_counts']}")
            return False
        
        print(f"✓ Status counts correct")
        
        # Verify failure reasons
        if len(diagnostics['top_failure_reasons']) != 2:
            print(f"❌ FAIL: Expected 2 failure reasons, got {len(diagnostics['top_failure_reasons'])}")
            return False
        
        print(f"✓ Failure reasons included: {len(diagnostics['top_failure_reasons'])}")
        
        # Verify broken tickers
        if set(diagnostics['broken_tickers']) != {"TICKER1", "TICKER2"}:
            print(f"❌ FAIL: Broken tickers mismatch")
            return False
        
        print(f"✓ Broken tickers correct: {diagnostics['broken_tickers']}")
        
        # Verify file saved
        if not os.path.exists("/tmp/test_diagnostics.json"):
            print("❌ FAIL: Diagnostics file not saved")
            return False
        
        print("✓ Diagnostics file saved successfully")
        
        # Load and verify
        loaded = load_diagnostics_artifact("/tmp/test_diagnostics.json")
        if loaded != diagnostics:
            print("❌ FAIL: Loaded diagnostics don't match original")
            return False
        
        print("✓ Diagnostics can be loaded correctly")
        
        # Print sample
        print("\n  Sample Diagnostics:")
        print("  " + "-" * 76)
        print(f"  Timestamp: {diagnostics['timestamp']}")
        print(f"  Waves: {diagnostics['waves_processed']}")
        print(f"  Summary: {diagnostics['summary']}")
        print("  " + "-" * 76)
        
        print("\n✅ PASS: Diagnostics Artifact Creation")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("ROUND 4 FEATURE VERIFICATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Wave Registry Validator": test_wave_registry_validator(),
        "Snapshot Enhancements": test_snapshot_enhancements(),
        "Executive Summary": test_executive_summary(),
        "Diagnostics Artifact": test_diagnostics_artifact(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*80)
    print(f"OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
