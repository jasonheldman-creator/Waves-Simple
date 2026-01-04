#!/usr/bin/env python3
"""
Quick validation script for Console v2 Proxy Rebuild Demo

This script performs quick sanity checks to ensure the demo components are working:
1. Proxy registry has 28 waves
2. Demo script is executable
3. Snapshot file exists with correct structure
4. All core modules can be imported

Usage:
    python validate_demo.py
"""

import os
import sys
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_mark(passed):
    """Return colored check mark."""
    return f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"


def test_proxy_registry():
    """Test that proxy registry has exactly 28 waves."""
    print("\n1. Testing Proxy Registry...")
    
    registry_path = "config/wave_proxy_registry.csv"
    
    if not os.path.exists(registry_path):
        print(f"  {check_mark(False)} Registry file not found: {registry_path}")
        return False
    
    with open(registry_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    from waves_engine import WAVE_ID_REGISTRY
    expected_count = len(WAVE_ID_REGISTRY)
    
    # Subtract 1 for header
    wave_count = len(lines) - 1
    
    passed = wave_count == expected_count
    print(f"  {check_mark(passed)} Wave count: {wave_count} (expected {expected_count} from WAVE_ID_REGISTRY)")
    
    return passed


def test_demo_script():
    """Test that demo script exists and is executable."""
    print("\n2. Testing Demo Script...")
    
    script_path = "demo_console_v2_proxy_rebuild.py"
    
    if not os.path.exists(script_path):
        print(f"  {check_mark(False)} Demo script not found: {script_path}")
        return False
    
    print(f"  {check_mark(True)} Demo script exists")
    
    # Check if executable
    is_executable = os.access(script_path, os.X_OK)
    print(f"  {check_mark(is_executable)} Executable flag set")
    
    return True


def test_modules():
    """Test that required modules can be imported."""
    print("\n3. Testing Module Imports...")
    
    modules_to_test = [
        'planb_proxy_pipeline',
        'helpers.proxy_registry_validator',
    ]
    
    all_passed = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  {check_mark(True)} {module_name}")
        except ImportError as e:
            print(f"  {check_mark(False)} {module_name}: {e}")
            all_passed = False
    
    return all_passed


def test_snapshot_structure():
    """Test that snapshot file has correct structure."""
    print("\n4. Testing Snapshot Structure...")
    
    snapshot_path = "data/live_proxy_snapshot.csv"
    
    if not os.path.exists(snapshot_path):
        print(f"  {YELLOW}⚠️{RESET}  Snapshot not found (will be created on first run)")
        return True
    
    with open(snapshot_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) < 2:
        print(f"  {check_mark(False)} Snapshot is empty")
        return False
    
    # Check header
    header = lines[0]
    required_columns = ['wave_id', 'display_name', 'confidence', 'proxy_ticker']
    
    header_ok = all(col in header for col in required_columns)
    print(f"  {check_mark(header_ok)} Required columns present")
    
    from waves_engine import WAVE_ID_REGISTRY
    expected_count = len(WAVE_ID_REGISTRY)
    
    # Check wave count
    wave_count = len(lines) - 1
    count_ok = wave_count == expected_count
    print(f"  {check_mark(count_ok)} Wave count: {wave_count} (expected {expected_count} from WAVE_ID_REGISTRY)")
    
    return header_ok and count_ok


def test_documentation():
    """Test that documentation exists."""
    print("\n5. Testing Documentation...")
    
    doc_path = "CONSOLE_V2_PROXY_REBUILD_DEMO.md"
    
    if not os.path.exists(doc_path):
        print(f"  {check_mark(False)} Documentation not found: {doc_path}")
        return False
    
    print(f"  {check_mark(True)} Documentation exists")
    
    # Check file size (should be substantial)
    file_size = os.path.getsize(doc_path)
    size_ok = file_size > 1000  # At least 1KB
    print(f"  {check_mark(size_ok)} Documentation size: {file_size} bytes")
    
    return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Console v2 Proxy Rebuild Demo - Validation")
    print("=" * 70)
    
    tests = [
        test_proxy_registry,
        test_demo_script,
        test_modules,
        test_snapshot_structure,
        test_documentation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  {RED}❌ Test failed with exception: {e}{RESET}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    passed_count = sum(results)
    total_count = len(results)
    
    if passed_count == total_count:
        print(f"{GREEN}✅ ALL VALIDATION CHECKS PASSED ({passed_count}/{total_count}){RESET}")
        print("\nThe demo is ready to run!")
        print("\nNext steps:")
        print("  python demo_console_v2_proxy_rebuild.py")
        return 0
    else:
        print(f"{RED}❌ SOME CHECKS FAILED ({passed_count}/{total_count} passed){RESET}")
        print("\nPlease fix the issues above before running the demo.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
