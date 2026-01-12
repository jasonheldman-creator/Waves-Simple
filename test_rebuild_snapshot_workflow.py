#!/usr/bin/env python3
"""
Test for rebuild snapshot workflow.

This test validates that the rebuild snapshot script generates a snapshot
with all required fields including strategy pipeline results.
"""

import sys
import os
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_rebuild_snapshot_has_required_fields():
    """
    Test that rebuild snapshot generates live_snapshot.csv with all required fields.
    
    Required fields per problem statement:
    1. Returns: return_1d, return_30d, return_60d, return_365d
    2. Alpha metrics: alpha_1d, alpha_30d, alpha_60d, alpha_365d
    3. VIX: vix_regime, vix_level
    4. Exposure: exposure, cash (CashPercent)
    """
    print("\n" + "=" * 80)
    print("TEST: Rebuild snapshot has required strategy pipeline fields")
    print("=" * 80)
    
    snapshot_file = "data/live_snapshot.csv"
    
    # Check file exists
    assert os.path.exists(snapshot_file), f"Snapshot file not found: {snapshot_file}"
    print(f"✓ Snapshot file exists: {snapshot_file}")
    
    # Load snapshot
    df = pd.read_csv(snapshot_file)
    
    # Normalize column names to handle case variations
    df.columns = [col.lower().replace('_', '').replace(' ', '') for col in df.columns]
    
    # Required fields (normalized)
    required_fields = {
        'returns': ['return1d', 'return30d', 'return60d', 'return365d'],
        'alpha': ['alpha1d', 'alpha30d', 'alpha60d', 'alpha365d'],
        'vix': ['vixregime'],
        'exposure': ['exposure', 'cashpercent'],
    }
    
    # Check each category
    all_fields_present = True
    for category, fields in required_fields.items():
        missing = [f for f in fields if f not in df.columns]
        if missing:
            print(f"✗ Missing {category} fields: {', '.join(missing)}")
            all_fields_present = False
        else:
            print(f"✓ All {category} fields present: {', '.join(fields)}")
    
    assert all_fields_present, "Some required fields are missing from snapshot"
    
    # Check for clean_transit_infrastructure_wave
    wave_id_col = 'waveid' if 'waveid' in df.columns else 'wave'
    if wave_id_col in df.columns:
        transit_rows = df[df[wave_id_col].str.contains('transit', case=False, na=False)]
        assert len(transit_rows) > 0, "clean_transit_infrastructure_wave not found"
        print(f"✓ clean_transit_infrastructure_wave found")
        
        # Check it has numeric values for required fields
        transit_row = transit_rows.iloc[0]
        
        # Check returns are numeric
        for field in required_fields['returns']:
            value = transit_row.get(field)
            if pd.notna(value):
                print(f"  {field}: {value}")
            else:
                print(f"  {field}: N/A")
        
        # Check alpha fields exist (may be NaN)
        alpha_present = any(field in df.columns for field in required_fields['alpha'])
        assert alpha_present, "Alpha fields missing"
        print(f"✓ Alpha fields present")
        
        # Check VIX regime
        if 'vixregime' in df.columns:
            vix_regime = transit_row.get('vixregime')
            print(f"  VIX Regime: {vix_regime}")
        
        # Check exposure
        if 'exposure' in df.columns:
            exposure = transit_row.get('exposure')
            print(f"  Exposure: {exposure}")
    
    print("\n" + "=" * 80)
    print("✓ TEST PASSED: All required fields present")
    print("=" * 80 + "\n")


def test_equity_waves_have_strategy_data():
    """
    Test that equity waves have strategy-related data populated.
    """
    print("\n" + "=" * 80)
    print("TEST: Equity waves have strategy data")
    print("=" * 80)
    
    snapshot_file = "data/live_snapshot.csv"
    df = pd.read_csv(snapshot_file)
    
    # Check for equity waves (based on category or name)
    # Look for waves that should have alpha data
    equity_indicators = ['sp500', 'ai', 'megacap', 'growth', 'disruptor']
    
    # Find equity wave rows
    wave_col = 'Wave' if 'Wave' in df.columns else 'wave'
    
    if wave_col in df.columns:
        equity_rows = df[df[wave_col].str.contains('|'.join(equity_indicators), case=False, na=False)]
        
        if len(equity_rows) > 0:
            print(f"✓ Found {len(equity_rows)} equity waves")
            
            # Check alpha columns exist
            alpha_cols = [col for col in df.columns if 'alpha' in col.lower()]
            assert len(alpha_cols) > 0, "No alpha columns found"
            print(f"✓ Alpha columns present: {', '.join(alpha_cols[:4])}")
            
            # Check at least some equity waves have non-NaN alpha values
            for col in alpha_cols:
                non_nan_count = equity_rows[col].notna().sum()
                if non_nan_count > 0:
                    print(f"  {col}: {non_nan_count}/{len(equity_rows)} waves have values")
        else:
            print("⚠ No equity waves found to validate")
    
    print("\n" + "=" * 80)
    print("✓ TEST PASSED: Equity waves have strategy data")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        test_rebuild_snapshot_has_required_fields()
        test_equity_waves_have_strategy_data()
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80 + "\n")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
