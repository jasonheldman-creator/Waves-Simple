#!/usr/bin/env python3
"""
Test: Verify No Implicit yfinance Calls in Execution Paths

This test ensures that:
1. PRICE_BOOK is used as the single source of truth
2. No implicit yfinance fetching occurs during normal execution
3. yfinance is only used in explicit rebuild paths or optional features

Allowed yfinance usage:
- get_next_earnings_date (optional ticker feature, gracefully degrades)
- Explicit rebuild/refresh button handlers
- Test/diagnostic scripts (not in main execution path)

Not allowed:
- yfinance calls during page load
- yfinance calls during data display
- yfinance calls during readiness checks
"""

import sys
import os
import re


def test_no_yfinance_in_executive_summary():
    """Verify executive summary uses PRICE_BOOK, not yfinance."""
    print("=" * 70)
    print("Testing: Executive Summary uses PRICE_BOOK (not yfinance)")
    print("=" * 70)
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Find the executive summary section
    # Look for the section between "EXECUTIVE SUMMARY NARRATIVE" and "Generate executive summary"
    pattern = r'# EXECUTIVE SUMMARY NARRATIVE.*?# Generate executive summary'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("❌ FAILED: Could not find executive summary section")
        return False
    
    section = match.group(0)
    
    # Check that section does NOT contain "import yfinance"
    if 'import yfinance' in section:
        print("❌ FAILED: Executive summary section still imports yfinance")
        print(f"   Found at position {section.find('import yfinance')}")
        return False
    
    # Check that section DOES contain get_price_book
    if 'get_price_book' not in section:
        print("❌ FAILED: Executive summary section doesn't use get_price_book")
        return False
    
    # Check for PRICE_BOOK usage
    if 'price_book' not in section.lower():
        print("❌ FAILED: Executive summary section doesn't reference price_book")
        return False
    
    print("✅ SUCCESS: Executive summary uses PRICE_BOOK (not yfinance)")
    print("   - No 'import yfinance' found in section")
    print("   - Uses get_price_book() correctly")
    return True


def test_allowed_yfinance_usage():
    """Verify yfinance is only used in allowed contexts."""
    print("\n" + "=" * 70)
    print("Testing: yfinance usage is only in allowed contexts")
    print("=" * 70)
    
    with open('app.py', 'r') as f:
        lines = f.readlines()
    
    # Find all lines with "import yfinance"
    yfinance_imports = []
    for i, line in enumerate(lines, 1):
        if 'import yfinance' in line and not line.strip().startswith('#'):
            yfinance_imports.append((i, line.strip()))
    
    print(f"\nFound {len(yfinance_imports)} yfinance import(s):")
    
    allowed_contexts = [
        'get_next_earnings_date',  # Optional earnings date feature
        'get_fed_decision_info',   # Optional Fed info feature
    ]
    
    all_allowed = True
    for line_num, line_content in yfinance_imports:
        # Get surrounding context (10 lines before)
        start = max(0, line_num - 10)
        end = min(len(lines), line_num + 5)
        context = ''.join(lines[start:end])
        
        # Check if in an allowed context
        in_allowed_context = any(ctx in context for ctx in allowed_contexts)
        
        if in_allowed_context:
            print(f"   ✓ Line {line_num}: {line_content}")
            print(f"      Context: Allowed (optional feature)")
        else:
            print(f"   ✗ Line {line_num}: {line_content}")
            print(f"      Context: NOT in allowed context list")
            print(f"      Surrounding code:")
            print("      " + "-" * 60)
            for ctx_line in lines[start:end]:
                print(f"      {ctx_line.rstrip()}")
            print("      " + "-" * 60)
            all_allowed = False
    
    if all_allowed:
        print(f"\n✅ SUCCESS: All {len(yfinance_imports)} yfinance import(s) are in allowed contexts")
        return True
    else:
        print(f"\n❌ FAILED: Found yfinance imports in non-allowed contexts")
        return False


def test_price_book_usage():
    """Verify PRICE_BOOK is used consistently."""
    print("\n" + "=" * 70)
    print("Testing: PRICE_BOOK is imported and used")
    print("=" * 70)
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check for get_price_book imports
    if 'from helpers.price_book import' not in content and 'get_price_book' not in content:
        print("❌ FAILED: get_price_book not imported")
        return False
    
    # Count usages of get_price_book
    usage_count = content.count('get_price_book(')
    
    if usage_count == 0:
        print("❌ FAILED: get_price_book is imported but never used")
        return False
    
    print(f"✅ SUCCESS: PRICE_BOOK is properly imported and used")
    print(f"   - get_price_book() called {usage_count} time(s)")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("No Implicit yfinance Calls - Test Suite")
    print("=" * 70)
    print()
    
    results = {
        'Executive Summary uses PRICE_BOOK': test_no_yfinance_in_executive_summary(),
        'yfinance only in allowed contexts': test_allowed_yfinance_usage(),
        'PRICE_BOOK properly used': test_price_book_usage(),
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    print("=" * 70)
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! No implicit yfinance calls detected.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
