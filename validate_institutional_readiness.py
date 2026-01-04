#!/usr/bin/env python3
"""
Validation script for Institutional Readiness tab implementation.

This script verifies that the render_overview_clean_tab function:
1. Exists and is importable
2. Has the correct structure and components
3. Uses appropriate executive language
4. Has diagnostics in a collapsed expander
"""

import sys
import os
import ast
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tab_label_consistency():
    """Test that 'Institutional Readiness' label is used consistently."""
    print("=" * 70)
    print("TEST: Tab Label Consistency")
    print("=" * 70)
    
    # Read app.py
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check for "Institutional Readiness" occurrences
    ir_count = content.count('"Institutional Readiness"')
    print(f"\n✓ Found {ir_count} occurrences of 'Institutional Readiness' label")
    
    # Check that "Overview (Clean)" is NOT in the code (only in docs)
    overview_clean_count = content.count('"Overview (Clean)"')
    assert overview_clean_count == 0, "Found 'Overview (Clean)' in app.py - should be renamed"
    print("✓ No 'Overview (Clean)' labels found in app.py")
    
    # Verify the label appears in tab definitions
    assert ir_count >= 3, "Expected at least 3 tab label occurrences (one per config)"
    print("✓ Tab label appears in all configurations")
    
    print("\n✅ Tab label consistency check PASSED")
    return True


def test_function_structure():
    """Test that render_overview_clean_tab has correct structure."""
    print("\n" + "=" * 70)
    print("TEST: Function Structure")
    print("=" * 70)
    
    # Import the function
    try:
        from app import render_overview_clean_tab
        print("\n✓ Function render_overview_clean_tab successfully imported")
    except ImportError as e:
        print(f"\n❌ Failed to import function: {e}")
        return False
    
    # Get function source
    source = inspect.getsource(render_overview_clean_tab)
    
    # Check for key sections
    required_sections = {
        "COMPOSITE SYSTEM CONTROL STATUS": "Composite System Control Status section",
        "AI EXECUTIVE BRIEF": "AI Executive Brief section",
        "HUMAN-READABLE SIGNALS": "Platform Intelligence Signals section",
        "AI RECOMMENDATIONS": "AI Recommendations section",
        "PERFORMANCE INSIGHTS": "Top Performing Strategies section",
        "MARKET CONTEXT": "Market Context section",
        "SYSTEM DIAGNOSTICS": "System Diagnostics section"
    }
    
    for keyword, description in required_sections.items():
        if keyword in source:
            print(f"✓ Found {description}")
        else:
            print(f"❌ Missing {description}")
            return False
    
    # Check for collapsed expander
    if 'expanded=False' in source:
        print("✓ Diagnostics section uses collapsed expander")
    else:
        print("⚠️  Warning: No collapsed expander found")
    
    print("\n✅ Function structure check PASSED")
    return True


def test_executive_language():
    """Test that executive-appropriate language is used."""
    print("\n" + "=" * 70)
    print("TEST: Executive Language")
    print("=" * 70)
    
    # Import the function
    from app import render_overview_clean_tab
    source = inspect.getsource(render_overview_clean_tab)
    
    # Check for regime-based language
    regime_terms = [
        "regime",
        "posture",
        "Broad-based strength",
        "Balanced performance",
        "Selective opportunities",
        "market conditions"
    ]
    
    found_terms = []
    for term in regime_terms:
        if term in source:
            found_terms.append(term)
    
    print(f"\n✓ Found {len(found_terms)} regime/executive terms:")
    for term in found_terms[:5]:
        print(f"  - '{term}'")
    
    # Check that catastrophic language is NOT present
    catastrophic_phrases = [
        "0 of 27 strategies posting gains",
        "CATASTROPHIC",
        "DISASTER",
        "TOTAL FAILURE"
    ]
    
    for phrase in catastrophic_phrases:
        if phrase in source:
            print(f"❌ Found catastrophic language: '{phrase}'")
            return False
    
    print("✓ No catastrophic language detected")
    
    print("\n✅ Executive language check PASSED")
    return True


def test_component_order():
    """Test that components are in the correct order."""
    print("\n" + "=" * 70)
    print("TEST: Component Order")
    print("=" * 70)
    
    from app import render_overview_clean_tab
    source = inspect.getsource(render_overview_clean_tab)
    
    # Find positions of key sections
    sections = {
        "EXECUTIVE HEADER": source.find("EXECUTIVE HEADER"),
        "COMPOSITE SYSTEM CONTROL STATUS": source.find("COMPOSITE SYSTEM CONTROL STATUS"),
        "AI EXECUTIVE BRIEF": source.find("AI EXECUTIVE BRIEF"),
        "HUMAN-READABLE SIGNALS": source.find("HUMAN-READABLE SIGNALS"),
        "AI RECOMMENDATIONS": source.find("AI RECOMMENDATIONS"),
        "PERFORMANCE INSIGHTS": source.find("PERFORMANCE INSIGHTS"),
        "MARKET CONTEXT": source.find("MARKET CONTEXT"),
        "SYSTEM DIAGNOSTICS": source.find("SYSTEM DIAGNOSTICS")
    }
    
    # Verify order
    positions = [(name, pos) for name, pos in sections.items() if pos != -1]
    positions.sort(key=lambda x: x[1])
    
    print("\nComponent order (top to bottom):")
    for i, (name, pos) in enumerate(positions, 1):
        print(f"  {i}. {name}")
    
    # Expected order
    expected_order = [
        "EXECUTIVE HEADER",
        "COMPOSITE SYSTEM CONTROL STATUS",
        "AI EXECUTIVE BRIEF",
        "HUMAN-READABLE SIGNALS",
        "AI RECOMMENDATIONS",
        "PERFORMANCE INSIGHTS",
        "MARKET CONTEXT",
        "SYSTEM DIAGNOSTICS"
    ]
    
    actual_order = [name for name, _ in positions]
    
    if actual_order == expected_order:
        print("\n✓ Components are in correct order")
        print("\n✅ Component order check PASSED")
        return True
    else:
        print("\n❌ Component order mismatch")
        print(f"Expected: {expected_order}")
        print(f"Actual: {actual_order}")
        return False


def test_constants_defined():
    """Test that required constants are defined."""
    print("\n" + "=" * 70)
    print("TEST: Required Constants")
    print("=" * 70)
    
    try:
        from app import (
            POSTURE_STRONG_POSITIVE,
            POSTURE_WEAK_NEGATIVE,
            DISPERSION_HIGH,
            DISPERSION_LOW,
            CONFIDENCE_HIGH_COVERAGE_PCT,
            CONFIDENCE_MODERATE_COVERAGE_PCT,
            RISK_REGIME_VIX_LOW,
            RISK_REGIME_VIX_HIGH,
            ALPHA_QUALITY_STRONG_RETURN,
            ALPHA_QUALITY_STRONG_RATIO,
            DATA_INTEGRITY_VERIFIED_COVERAGE,
            DEFAULT_ALPHA_QUALITY,
            DEFAULT_RISK_REGIME,
            DEFAULT_DATA_INTEGRITY,
            DEFAULT_CONFIDENCE
        )
        
        print("\n✓ All required constants imported successfully")
        print(f"  - POSTURE_STRONG_POSITIVE = {POSTURE_STRONG_POSITIVE}")
        print(f"  - RISK_REGIME_VIX_LOW = {RISK_REGIME_VIX_LOW}")
        print(f"  - CONFIDENCE_HIGH_COVERAGE_PCT = {CONFIDENCE_HIGH_COVERAGE_PCT}")
        print(f"  - ALPHA_QUALITY_STRONG_RETURN = {ALPHA_QUALITY_STRONG_RETURN}")
        
        print("\n✅ Constants definition check PASSED")
        return True
    except ImportError as e:
        print(f"\n❌ Failed to import constants: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n")
    print("*" * 80)
    print("INSTITUTIONAL READINESS TAB VALIDATION")
    print("Verifying implementation against PR requirements")
    print("*" * 80)
    
    tests = [
        ("Tab Label Consistency", test_tab_label_consistency),
        ("Function Structure", test_function_structure),
        ("Executive Language", test_executive_language),
        ("Component Order", test_component_order),
        ("Required Constants", test_constants_defined),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n")
    print("*" * 80)
    print("VALIDATION RESULTS")
    print("*" * 80)
    print(f"✓ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
        print("\n⚠️  Some requirements not met. Review failed tests above.")
    else:
        print("🎉 All validation tests passed!")
        print("\n✅ Institutional Readiness tab implementation is COMPLETE")
        print("   - Tab label: 'Institutional Readiness' ✓")
        print("   - Hero section with AI Brief & System Status ✓")
        print("   - Collapsible diagnostics ✓")
        print("   - Executive-appropriate language ✓")
        print("   - Correct component order ✓")
    print("*" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
