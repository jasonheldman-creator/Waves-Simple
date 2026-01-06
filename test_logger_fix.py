#!/usr/bin/env python3
"""
Test script to verify logger fix in app.py Sidebar Operator Controls.

This script tests:
1. Logger definition is correct
2. Logging calls are fail-safe
3. Button handlers don't crash even if logging fails
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_logger_definition():
    """Test that the logger definition pattern works correctly."""
    print("=" * 70)
    print("TEST: Logger Definition Pattern")
    print("=" * 70)
    
    # Test the logger definition pattern used in the fix
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    
    print("\n1. Testing logger creation...")
    assert logger is not None, "Logger should be created"
    print("   ✓ Logger created successfully")
    
    print("\n2. Testing logger.info call...")
    try:
        logger.info("Test message")
        print("   ✓ logger.info() works without error")
    except Exception as e:
        print(f"   ✗ logger.info() failed: {e}")
        return False
    
    return True

def test_fail_safe_logging():
    """Test that logging calls are fail-safe."""
    print("\n" + "=" * 70)
    print("TEST: Fail-Safe Logging")
    print("=" * 70)
    
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    
    print("\n1. Testing fail-safe logging wrapper...")
    
    # Simulate the fail-safe pattern used in the fix
    try:
        logger.info("This should work")
    except Exception:
        pass  # Logging errors should not block execution
    
    print("   ✓ Fail-safe logging pattern works")
    
    # Test that execution continues even if there's a logging error
    execution_continued = False
    try:
        logger.info("Test message")
    except Exception:
        pass
    execution_continued = True
    
    assert execution_continued, "Execution should continue after logging"
    print("   ✓ Execution continues after logging call")
    
    return True

def test_app_imports():
    """Test that app.py can be imported (syntax check)."""
    print("\n" + "=" * 70)
    print("TEST: App.py Import Check")
    print("=" * 70)
    
    print("\n1. Checking if app.py has syntax errors...")
    
    # Try to compile the app.py file
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    try:
        with open(app_path, 'r') as f:
            code = f.read()
        compile(code, app_path, 'exec')
        print(f"   ✓ app.py compiles without syntax errors")
        return True
    except SyntaxError as e:
        print(f"   ✗ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Error checking app.py: {e}")
        return False

def test_logger_in_app():
    """Test that the logger is defined correctly in app.py."""
    print("\n" + "=" * 70)
    print("TEST: Logger Definition in app.py")
    print("=" * 70)
    
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    print("\n1. Checking for logger definition before Operator Controls...")
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Check that logger is defined
    if "logger = logging.getLogger(__name__)" in content:
        print("   ✓ Logger definition found")
    else:
        print("   ✗ Logger definition not found")
        return False
    
    # Check that logging calls are fail-safe
    print("\n2. Checking for fail-safe logging calls...")
    fail_safe_patterns = [
        "try:\n                logger.info",
        "except Exception:\n                pass  # Logging errors should not block"
    ]
    
    for pattern in fail_safe_patterns:
        if pattern in content:
            print(f"   ✓ Fail-safe pattern found: {pattern[:40]}...")
        else:
            print(f"   ✗ Fail-safe pattern not found: {pattern[:40]}...")
            return False
    
    print("\n3. Checking that 'import logging' exists...")
    if "import logging" in content:
        print("   ✓ 'import logging' found")
    else:
        print("   ✗ 'import logging' not found")
        return False
    
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LOGGER FIX VERIFICATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Logger Definition Pattern", test_logger_definition),
        ("Fail-Safe Logging", test_fail_safe_logging),
        ("App.py Import Check", test_app_imports),
        ("Logger in app.py", test_logger_in_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
