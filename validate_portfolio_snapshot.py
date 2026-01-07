#!/usr/bin/env python3
"""
Validation script for portfolio snapshot implementation.

This script performs basic validation that doesn't require dependencies:
- Syntax validation
- Function signature verification
- Code structure checks
"""

import ast
import sys
import os


def validate_file_syntax(filepath):
    """Validate Python syntax for a file."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, "Syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


def check_function_exists(filepath, function_name):
    """Check if a function exists in a file."""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return True, f"Function '{function_name}' found"
        
        return False, f"Function '{function_name}' not found"
    except Exception as e:
        return False, f"Error checking function: {e}"


def check_imports_in_file(filepath, import_names):
    """Check if specific imports exist in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing_imports = []
        for import_name in import_names:
            if import_name not in content:
                missing_imports.append(import_name)
        
        if missing_imports:
            return False, f"Missing imports: {', '.join(missing_imports)}"
        return True, "All imports found"
    except Exception as e:
        return False, f"Error checking imports: {e}"


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("Portfolio Snapshot Implementation Validation")
    print("=" * 70)
    
    checks = []
    
    # 1. Validate helpers/wave_performance.py
    print("\n1. Validating helpers/wave_performance.py")
    print("-" * 70)
    
    filepath = "helpers/wave_performance.py"
    
    # Syntax check
    success, msg = validate_file_syntax(filepath)
    checks.append(("Syntax validation", success, msg))
    print(f"  {'✓' if success else '✗'} {msg}")
    
    # Function existence checks
    functions = [
        "compute_portfolio_snapshot",
        "compute_portfolio_alpha_attribution",
        "validate_portfolio_diagnostics"
    ]
    
    for func_name in functions:
        success, msg = check_function_exists(filepath, func_name)
        checks.append((f"Function {func_name}", success, msg))
        print(f"  {'✓' if success else '✗'} {msg}")
    
    # 2. Validate app.py
    print("\n2. Validating app.py")
    print("-" * 70)
    
    filepath = "app.py"
    
    # Syntax check
    success, msg = validate_file_syntax(filepath)
    checks.append(("app.py syntax", success, msg))
    print(f"  {'✓' if success else '✗'} {msg}")
    
    # Check for portfolio snapshot imports
    imports = [
        "compute_portfolio_snapshot",
        "compute_portfolio_alpha_attribution",
        "validate_portfolio_diagnostics"
    ]
    
    success, msg = check_imports_in_file(filepath, imports)
    checks.append(("Portfolio snapshot imports", success, msg))
    print(f"  {'✓' if success else '✗'} {msg}")
    
    # Check for UI elements
    ui_elements = [
        "Portfolio Snapshot",
        "💼 Portfolio Snapshot",
        "Alpha Attribution"
    ]
    
    success, msg = check_imports_in_file(filepath, ui_elements)
    checks.append(("UI elements", success, msg))
    print(f"  {'✓' if success else '✗'} {msg}")
    
    # 3. Validate test file
    print("\n3. Validating test_portfolio_snapshot.py")
    print("-" * 70)
    
    filepath = "test_portfolio_snapshot.py"
    
    # Syntax check
    success, msg = validate_file_syntax(filepath)
    checks.append(("Test file syntax", success, msg))
    print(f"  {'✓' if success else '✗'} {msg}")
    
    # Check for test functions
    test_functions = [
        "test_portfolio_snapshot_basic",
        "test_alpha_attribution",
        "test_diagnostics_validation",
        "test_wave_level_snapshot"
    ]
    
    for func_name in test_functions:
        success, msg = check_function_exists(filepath, func_name)
        checks.append((f"Test {func_name}", success, msg))
        print(f"  {'✓' if success else '✗'} {msg}")
    
    # 4. Check documentation
    print("\n4. Validating documentation")
    print("-" * 70)
    
    doc_file = "PORTFOLIO_SNAPSHOT_IMPLEMENTATION.md"
    if os.path.exists(doc_file):
        checks.append(("Documentation exists", True, f"{doc_file} found"))
        print(f"  ✓ {doc_file} found")
    else:
        checks.append(("Documentation exists", False, f"{doc_file} not found"))
        print(f"  ✗ {doc_file} not found")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in checks if success)
    total = len(checks)
    
    for check_name, success, msg in checks:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not success:
            print(f"       {msg}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    print("=" * 70)
    
    if passed == total:
        print("✓ All validation checks passed!")
        return 0
    else:
        print(f"✗ {total - passed} validation check(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
