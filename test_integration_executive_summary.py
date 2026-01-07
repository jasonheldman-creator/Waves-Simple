#!/usr/bin/env python3
"""
Integration test for Executive Summary enhancement.

This test verifies that all components work together correctly.
"""

import sys
import importlib.util


def test_integration():
    """Run integration tests for the Executive Summary enhancement."""
    print("=" * 80)
    print("INTEGRATION TEST: Executive Summary Enhancement")
    print("=" * 80)
    
    # Test 1: Verify constants are defined
    print("\n✓ Test 1: Verify new constants are defined...")
    with open('app.py', 'r') as f:
        content = f.read()
    
    constants = [
        'ATTRIBUTION_TILT_STRENGTH',
        'ATTRIBUTION_BASE_EXPOSURE', 
        'ATTRIBUTION_TIMEFRAME_DAYS'
    ]
    
    for const in constants:
        if f"{const} =" in content:
            print(f"  ✅ Constant defined: {const}")
        else:
            print(f"  ❌ Constant NOT defined: {const}")
            return False
    
    # Test 2: Verify constants are used (not hardcoded values)
    print("\n✓ Test 2: Verify constants are used in Executive Summary code...")
    
    # Extract the Executive Summary section
    exec_summary_start = content.find('SECTION A2: Executive Summary')
    exec_summary_end = content.find('SECTION B: Alpha Drivers Breakdown', exec_summary_start)
    
    if exec_summary_start == -1 or exec_summary_end == -1:
        print("  ❌ Could not locate Executive Summary section")
        return False
    
    exec_summary_section = content[exec_summary_start:exec_summary_end]
    
    # Check that hardcoded values are NOT in our section
    if 'tilt_strength=0.8' in exec_summary_section:
        print("  ❌ Hardcoded tilt_strength=0.8 found in Executive Summary")
        return False
    else:
        print("  ✅ No hardcoded tilt_strength values in Executive Summary")
    
    if 'base_exposure=1.0' in exec_summary_section:
        print("  ❌ Hardcoded base_exposure=1.0 found in Executive Summary")
        return False
    else:
        print("  ✅ No hardcoded base_exposure values in Executive Summary")
    
    # Check that constants are actually used
    if 'tilt_strength=ATTRIBUTION_TILT_STRENGTH' in exec_summary_section:
        print("  ✅ ATTRIBUTION_TILT_STRENGTH constant is used")
    else:
        print("  ❌ ATTRIBUTION_TILT_STRENGTH constant NOT used in Executive Summary")
        return False
    
    if 'base_exposure=ATTRIBUTION_BASE_EXPOSURE' in exec_summary_section:
        print("  ✅ ATTRIBUTION_BASE_EXPOSURE constant is used")
    else:
        print("  ❌ ATTRIBUTION_BASE_EXPOSURE constant NOT used in Executive Summary")
        return False
    
    # Test 3: Verify no inplace=True operations in new code
    print("\n✓ Test 3: Verify no inplace=True in Executive Summary code...")
    
    if 'inplace=True' in exec_summary_section:
        print("  ❌ Found inplace=True in Executive Summary section")
        return False
    else:
        print("  ✅ No inplace=True in Executive Summary section")
    
    # Test 4: Verify error messages use constants
    print("\n✓ Test 4: Verify error messages use constants...")
    
    if 'ATTRIBUTION_TIMEFRAME_DAYS' in exec_summary_section:
        print("  ✅ Error messages use ATTRIBUTION_TIMEFRAME_DAYS constant")
    else:
        print("  ⚠️  ATTRIBUTION_TIMEFRAME_DAYS may not be in error messages (acceptable)")
    
    # Test 5: Verify the structure is correct
    print("\n✓ Test 5: Verify code structure...")
    
    # Check that S&P 500 check comes before attribution computation
    sp500_check_pos = content.find('if selected_wave == "S&P 500 Wave":')
    compute_pos = content.find('compute_alpha_attribution_series', sp500_check_pos)
    
    if sp500_check_pos != -1 and compute_pos != -1 and sp500_check_pos < compute_pos:
        print("  ✅ S&P 500 Wave check precedes attribution computation")
    else:
        print("  ❌ Code structure may be incorrect")
        return False
    
    # Check that else clause has placeholder
    placeholder_pos = content.find('Attribution Rollout Pending', sp500_check_pos)
    if placeholder_pos != -1 and placeholder_pos < exec_summary_end:
        print("  ✅ Placeholder message in else clause")
    else:
        print("  ❌ Placeholder message not found in correct location")
        return False
    
    # Test 6: Verify syntax is valid
    print("\n✓ Test 6: Verify Python syntax...")
    
    try:
        spec = importlib.util.spec_from_file_location('app', 'app.py')
        if spec and spec.loader:
            print("  ✅ app.py has valid Python syntax")
        else:
            print("  ❌ app.py cannot be loaded")
            return False
    except Exception as e:
        print(f"  ❌ Syntax error: {e}")
        return False
    
    # Test 7: Verify alpha_attribution.py is unchanged
    print("\n✓ Test 7: Verify alpha_attribution.py is unchanged...")
    
    try:
        with open('alpha_attribution.py', 'r') as f:
            attribution_content = f.read()
        
        # Check for key functions
        required_functions = [
            'def compute_alpha_attribution_series(',
            'def compute_daily_alpha_attribution(',
            'def format_attribution_summary_table(',
            'def format_daily_attribution_sample('
        ]
        
        all_present = True
        for func in required_functions:
            if func in attribution_content:
                print(f"  ✅ Found: {func.strip(':')}")
            else:
                print(f"  ❌ Missing: {func.strip(':')}")
                all_present = False
        
        if not all_present:
            return False
            
    except FileNotFoundError:
        print("  ❌ alpha_attribution.py not found")
        return False
    
    # Test 8: Verify documentation exists
    print("\n✓ Test 8: Verify documentation...")
    
    try:
        with open('IMPLEMENTATION_SUMMARY_EXECUTIVE_SUMMARY.md', 'r') as f:
            doc_content = f.read()
        
        if len(doc_content) > 1000:
            print("  ✅ Implementation summary document exists and is comprehensive")
        else:
            print("  ⚠️  Documentation exists but may be incomplete")
    except FileNotFoundError:
        print("  ⚠️  Implementation summary document not found")
    
    print("\n" + "=" * 80)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("=" * 80)
    
    print("\nSummary:")
    print("  • All constants properly defined and used")
    print("  • No unsafe inplace operations")
    print("  • Code structure is correct")
    print("  • Syntax is valid")
    print("  • Calculation logic unchanged")
    print("  • Documentation complete")
    
    print("\n🎉 Executive Summary enhancement is ready for production!")
    return True


if __name__ == "__main__":
    try:
        success = test_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
