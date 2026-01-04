#!/usr/bin/env python3
"""
Institutional Readiness Tab Validation Script

This headless validation script verifies the Institutional Readiness tab implementation
without running Streamlit or loading UI dependencies. It uses source code inspection
and AST parsing to validate:
- Tab label consistency
- Structure and ordering
- Executive language patterns
- Diagnostics collapse state
- Threshold constants

No functional changes - validation only.
"""

import sys
import ast
import inspect
import re
from pathlib import Path


def validate_tab_label():
    """
    Verify that the tab label 'Institutional Readiness' appears consistently in app.py.
    """
    print("\n[1/6] Validating tab label consistency...")
    
    try:
        app_py = Path(__file__).parent.parent / "app.py"
        with open(app_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for tab label in st.tabs() calls
        tab_label_pattern = r'"Institutional Readiness"'
        matches = re.findall(tab_label_pattern, content)
        
        if len(matches) >= 3:  # Should appear in 3 different tab layouts
            print(f"  ✓ Tab label 'Institutional Readiness' found {len(matches)} times")
            return True
        else:
            print(f"  ✗ Tab label found only {len(matches)} times (expected at least 3)")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_function_exists():
    """
    Verify that render_overview_clean_tab function exists with correct signature.
    """
    print("\n[2/6] Validating function existence and structure...")
    
    try:
        app_py = Path(__file__).parent.parent / "app.py"
        with open(app_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for function definition
        func_pattern = r'def render_overview_clean_tab\(\):'
        if re.search(func_pattern, content):
            print(f"  ✓ Function 'render_overview_clean_tab()' found")
        else:
            print(f"  ✗ Function 'render_overview_clean_tab()' not found")
            return False
        
        # Check for docstring with correct description
        docstring_pattern = r'"""[\s\S]*?Institutional Readiness - Tab 1[\s\S]*?"""'
        if re.search(docstring_pattern, content):
            print(f"  ✓ Function docstring contains 'Institutional Readiness - Tab 1'")
        else:
            print(f"  ✗ Docstring missing or incorrect")
            return False
            
        return True
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_tab_ordering():
    """
    Verify that Institutional Readiness is the FIRST tab in all layouts.
    """
    print("\n[3/6] Validating tab ordering...")
    
    try:
        app_py = Path(__file__).parent.parent / "app.py"
        with open(app_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all st.tabs() calls and check first element
        tabs_pattern = r'st\.tabs\(\[(.*?)\]\)'
        matches = re.findall(tabs_pattern, content, re.DOTALL)
        
        first_tab_count = 0
        for match in matches:
            # Get first tab label
            first_label = match.strip().split('\n')[0].strip().strip(',').strip()
            if 'Institutional Readiness' in first_label:
                first_tab_count += 1
        
        if first_tab_count >= 3:  # Should be first in all 3 layouts
            print(f"  ✓ 'Institutional Readiness' is first tab in {first_tab_count} layouts")
            return True
        else:
            print(f"  ✗ 'Institutional Readiness' is first tab in only {first_tab_count} layouts")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_executive_language():
    """
    Verify executive-friendly language patterns in the function.
    """
    print("\n[4/6] Validating executive language patterns...")
    
    try:
        app_py = Path(__file__).parent.parent / "app.py"
        with open(app_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract function body
        func_start = content.find('def render_overview_clean_tab():')
        if func_start == -1:
            print(f"  ✗ Function not found")
            return False
        
        # Find next function definition to get bounds
        next_func = content.find('\ndef ', func_start + 1)
        if next_func == -1:
            next_func = len(content)
        
        func_body = content[func_start:next_func]
        
        # Check for required executive sections
        required_sections = [
            ('Executive Header', r'🏛️ Institutional Readiness'),
            ('System Status', r'Composite System Control Status'),
            ('Executive Summary', r'Executive Intelligence Summary'),
            ('Platform Signals', r'Platform Intelligence Signals'),
            ('AI Recommendations', r'AI Recommendations'),
            ('Top Performers', r'Top Performing Strategies'),
            ('Market Context', r'Market Context'),
        ]
        
        all_found = True
        for section_name, pattern in required_sections:
            if re.search(pattern, func_body):
                print(f"  ✓ {section_name} section found")
            else:
                print(f"  ✗ {section_name} section missing")
                all_found = False
        
        return all_found
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_diagnostics_collapse():
    """
    Verify that diagnostics are in a collapsed expander.
    """
    print("\n[5/6] Validating diagnostics collapse state...")
    
    try:
        app_py = Path(__file__).parent.parent / "app.py"
        with open(app_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract function body
        func_start = content.find('def render_overview_clean_tab():')
        if func_start == -1:
            print(f"  ✗ Function not found")
            return False
        
        next_func = content.find('\ndef ', func_start + 1)
        if next_func == -1:
            next_func = len(content)
        
        func_body = content[func_start:next_func]
        
        # Check for expander with System Diagnostics
        expander_pattern = r'st\.expander\(["\'].*?System Diagnostics.*?["\'].*?expanded=False\)'
        if re.search(expander_pattern, func_body, re.DOTALL):
            print(f"  ✓ System Diagnostics in collapsed expander (expanded=False)")
            return True
        else:
            print(f"  ✗ System Diagnostics expander missing or not collapsed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_threshold_constants():
    """
    Verify that all required threshold constants are defined in app.py.
    """
    print("\n[6/6] Validating threshold constants...")
    
    try:
        app_py = Path(__file__).parent.parent / "app.py"
        with open(app_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Required constants
        required_constants = [
            'CONFIDENCE_HIGH_COVERAGE_PCT',
            'CONFIDENCE_MODERATE_COVERAGE_PCT',
            'RISK_REGIME_VIX_LOW',
            'RISK_REGIME_VIX_HIGH',
            'RISK_REGIME_PERF_RISK_ON',
            'RISK_REGIME_PERF_RISK_OFF',
            'ALPHA_QUALITY_STRONG_RETURN',
            'ALPHA_QUALITY_STRONG_RATIO',
            'ALPHA_QUALITY_MIXED_RATIO',
            'POSTURE_STRONG_POSITIVE',
            'POSTURE_WEAK_NEGATIVE',
            'DISPERSION_HIGH',
            'DISPERSION_LOW',
            'DATA_INTEGRITY_VERIFIED_COVERAGE',
            'DATA_INTEGRITY_DEGRADED_COVERAGE',
            'DEFAULT_ALPHA_QUALITY',
            'DEFAULT_RISK_REGIME',
            'DEFAULT_DATA_INTEGRITY',
            'DEFAULT_CONFIDENCE',
        ]
        
        all_found = True
        for constant in required_constants:
            pattern = rf'^{constant}\s*='
            if re.search(pattern, content, re.MULTILINE):
                print(f"  ✓ Constant '{constant}' defined")
            else:
                print(f"  ✗ Constant '{constant}' missing")
                all_found = False
        
        return all_found
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """
    Run all validation checks and report results.
    """
    print("=" * 70)
    print("INSTITUTIONAL READINESS TAB VALIDATION")
    print("=" * 70)
    print("\nThis validation script verifies the Institutional Readiness tab")
    print("implementation without running Streamlit or loading UI dependencies.")
    print("\nValidation Scope:")
    print("  - Tab label consistency")
    print("  - Function structure")
    print("  - Tab ordering (first position)")
    print("  - Executive language patterns")
    print("  - Diagnostics collapse state")
    print("  - Threshold constants")
    
    # Run all validations
    results = []
    results.append(("Tab Label Consistency", validate_tab_label()))
    results.append(("Function Structure", validate_function_exists()))
    results.append(("Tab Ordering", validate_tab_ordering()))
    results.append(("Executive Language", validate_executive_language()))
    results.append(("Diagnostics Collapse", validate_diagnostics_collapse()))
    results.append(("Threshold Constants", validate_threshold_constants()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {name}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All validations passed successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} validation(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
