"""
Unit Test for Return Pipeline

This test verifies that the return pipeline produces the required columns
for one wave, validating the core functionality and data structure.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_return_pipeline():
    """Test return pipeline produces required columns for one wave."""
    print("=" * 70)
    print("Testing Return Pipeline")
    print("=" * 70)
    
    # Import modules directly to avoid streamlit dependency in helpers/__init__.py
    import importlib.util
    
    # Load return_pipeline module
    spec = importlib.util.spec_from_file_location(
        "return_pipeline",
        os.path.join(os.path.dirname(__file__), "helpers", "return_pipeline.py")
    )
    return_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(return_pipeline)
    compute_wave_returns_pipeline = return_pipeline.compute_wave_returns_pipeline
    
    # Load wave_registry module
    spec = importlib.util.spec_from_file_location(
        "wave_registry",
        os.path.join(os.path.dirname(__file__), "helpers", "wave_registry.py")
    )
    wave_registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wave_registry)
    get_wave_registry = wave_registry.get_wave_registry
    
    # Get registry to find a wave to test
    print("\n1. Loading wave registry...")
    registry = get_wave_registry()
    assert not registry.empty, "Wave registry is empty"
    print(f"   ✓ Loaded {len(registry)} waves")
    
    # Get first active wave
    active_waves = registry[registry['active']]
    assert not active_waves.empty, "No active waves found"
    
    test_wave_id = active_waves.iloc[0]['wave_id']
    test_wave_name = active_waves.iloc[0]['wave_name']
    print(f"   ✓ Testing wave: {test_wave_name} (ID: {test_wave_id})")
    
    # Compute returns
    print("\n2. Computing returns...")
    returns_df = compute_wave_returns_pipeline(test_wave_id)
    
    # Verify output structure
    print("\n3. Verifying output structure...")
    
    # Check that we got a DataFrame
    assert returns_df is not None, "Returns dataframe is None"
    print("   ✓ Returns dataframe is not None")
    
    # Check required columns exist
    required_columns = [
        'wave_return',
        'benchmark_return',
        'alpha',
        'overlay_return_vix',
        'overlay_return_custom'
    ]
    
    for col in required_columns:
        assert col in returns_df.columns, f"Missing required column: {col}"
        print(f"   ✓ Column '{col}' present")
    
    # If we have data, verify column types
    if not returns_df.empty:
        print(f"\n4. Verifying data (found {len(returns_df)} rows)...")
        
        # Check that columns are numeric
        for col in required_columns:
            assert returns_df[col].dtype in ['float64', 'float32'], \
                f"Column {col} has wrong dtype: {returns_df[col].dtype}"
            print(f"   ✓ Column '{col}' is numeric")
        
        # Check that alpha = wave_return - benchmark_return (where not NaN)
        print("\n5. Verifying alpha calculation...")
        mask = ~(returns_df['wave_return'].isna() | returns_df['benchmark_return'].isna())
        if mask.any():
            computed_alpha = returns_df.loc[mask, 'wave_return'] - returns_df.loc[mask, 'benchmark_return']
            stored_alpha = returns_df.loc[mask, 'alpha']
            
            # Allow small floating point differences
            max_diff = (computed_alpha - stored_alpha).abs().max()
            assert max_diff < 1e-10, f"Alpha calculation error: max diff = {max_diff}"
            print(f"   ✓ Alpha = wave_return - benchmark_return (verified)")
        else:
            print("   ⚠ No valid data points to verify alpha calculation")
        
        # Check overlay placeholders are NaN
        print("\n6. Verifying overlay placeholders...")
        assert returns_df['overlay_return_vix'].isna().all(), "overlay_return_vix should be all NaN"
        print("   ✓ overlay_return_vix is all NaN (placeholder)")
        
        assert returns_df['overlay_return_custom'].isna().all(), "overlay_return_custom should be all NaN"
        print("   ✓ overlay_return_custom is all NaN (placeholder)")
        
        # Display sample data
        print("\n7. Sample output:")
        print(returns_df.head(10).to_string())
        
        print("\n8. Summary statistics:")
        print(returns_df.describe().to_string())
    else:
        print("\n   ⚠ Warning: Returns dataframe is empty (no price data available)")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == '__main__':
    test_return_pipeline()
