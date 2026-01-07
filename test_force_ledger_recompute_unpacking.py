#!/usr/bin/env python3
"""
Test to verify force_ledger_recompute() return value unpacking is correct.

This test ensures that all calls to force_ledger_recompute() in the codebase
correctly unpack the 3 return values: (success, message, details).
"""

import re
import os
import sys


def test_unpacking_correctness():
    """Verify all force_ledger_recompute() calls unpack 3 values."""
    
    # Files that call force_ledger_recompute
    files_to_check = [
        'ui_changes_preview.py',
        'app.py',
        'validate_ledger_recompute_fix.py',
        'test_ledger_recompute_network_independent.py'
    ]
    
    print("=" * 70)
    print("Testing force_ledger_recompute() return value unpacking")
    print("=" * 70)
    
    all_correct = True
    total_calls = 0
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"⚠️  {filepath}: File not found (skipping)")
            continue
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find all force_ledger_recompute() calls with unpacking
        # Pattern: variable1, variable2, ... = force_ledger_recompute()
        pattern = r'(\w+(?:\s*,\s*\w+)*)\s*=\s*force_ledger_recompute\(\)'
        
        matches = list(re.finditer(pattern, content))
        
        for match in matches:
            total_calls += 1
            vars_str = match.group(1)
            # Count variables by splitting on comma
            variables = [v.strip() for v in vars_str.split(',')]
            num_vars = len(variables)
            
            line_num = content[:match.start()].count('\n') + 1
            
            if num_vars == 3:
                print(f"✓ {filepath}:{line_num}")
                print(f"  Correctly unpacks 3 values: {vars_str}")
            else:
                print(f"✗ {filepath}:{line_num} - ERROR!")
                print(f"  Unpacks {num_vars} values: {vars_str}")
                print(f"  Expected: success, message, details")
                all_correct = False
    
    print("=" * 70)
    print(f"Total calls checked: {total_calls}")
    
    if all_correct and total_calls > 0:
        print("✅ All force_ledger_recompute() calls correctly unpack 3 values!")
        return True
    elif total_calls == 0:
        print("⚠️  No force_ledger_recompute() calls found")
        return False
    else:
        print("❌ Some calls have incorrect unpacking!")
        return False


def test_function_signature():
    """Verify the function signature returns 3 values."""
    
    print("\n" + "=" * 70)
    print("Checking function signature in operator_toolbox.py")
    print("=" * 70)
    
    filepath = 'helpers/operator_toolbox.py'
    
    if not os.path.exists(filepath):
        print(f"✗ {filepath} not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the function definition - handle Dict[str, Any] nested brackets
    pattern = r'def\s+force_ledger_recompute\([^)]*\)\s*->\s*Tuple\[(.*?)\]:\s*"""'
    
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return_type = match.group(1)
        print(f"✓ Function signature found")
        print(f"  Returns: Tuple[{return_type}]")
        
        # Count top-level commas (not within nested brackets)
        bracket_depth = 0
        comma_count = 0
        for char in return_type:
            if char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            elif char == ',' and bracket_depth == 0:
                comma_count += 1
        
        # Number of types = comma_count + 1
        num_types = comma_count + 1
        
        if num_types == 3:
            print(f"✓ Correctly returns 3 values")
            return True
        else:
            print(f"✗ Returns {num_types} values, expected 3")
            return False
    else:
        print(f"✗ Function signature not found or doesn't match expected pattern")
        return False


if __name__ == '__main__':
    print("\nForce Ledger Recompute Unpacking Test")
    print("=" * 70)
    
    # Test 1: Check function signature
    signature_ok = test_function_signature()
    
    # Test 2: Check all call sites
    unpacking_ok = test_unpacking_correctness()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if signature_ok and unpacking_ok:
        print("✅ All tests passed!")
        print("✅ force_ledger_recompute() correctly returns 3 values")
        print("✅ All call sites correctly unpack 3 values")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        if not signature_ok:
            print("  - Function signature issue")
        if not unpacking_ok:
            print("  - Unpacking issue at call sites")
        sys.exit(1)
