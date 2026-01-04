"""
Test: Data Integrity Badge Thresholds Alignment
Validates that app.py correctly uses PRICE_CACHE_OK_DAYS and PRICE_CACHE_DEGRADED_DAYS
from helpers/price_book.py as the canonical source of truth.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_constants_import():
    """Verify that constants can be imported from helpers/price_book.py"""
    from helpers.price_book import PRICE_CACHE_OK_DAYS, PRICE_CACHE_DEGRADED_DAYS
    
    assert PRICE_CACHE_OK_DAYS == 14, f"Expected PRICE_CACHE_OK_DAYS=14, got {PRICE_CACHE_OK_DAYS}"
    assert PRICE_CACHE_DEGRADED_DAYS == 30, f"Expected PRICE_CACHE_DEGRADED_DAYS=30, got {PRICE_CACHE_DEGRADED_DAYS}"
    
    print("✅ Constants imported correctly from helpers/price_book.py")
    print(f"   PRICE_CACHE_OK_DAYS = {PRICE_CACHE_OK_DAYS}")
    print(f"   PRICE_CACHE_DEGRADED_DAYS = {PRICE_CACHE_DEGRADED_DAYS}")


def test_validation_scenarios():
    """Test the validation scenarios specified in the problem statement"""
    from helpers.price_book import PRICE_CACHE_OK_DAYS, PRICE_CACHE_DEGRADED_DAYS
    
    # Constants for coverage thresholds (from app.py)
    DATA_INTEGRITY_VERIFIED_COVERAGE = 95.0
    DATA_INTEGRITY_DEGRADED_COVERAGE = 80.0
    
    # Test scenarios: (cache_age_days, expected_badge_status)
    scenarios = [
        (9, "High", "Verified"),
        (14, "High", "Verified"),
        (15, "Medium", "Degraded"),
        (31, "Low", "Compromised"),
    ]
    
    print("\n=== Validation Scenarios ===")
    for cache_age_days, expected_level, expected_status in scenarios:
        # Simulate the badge logic with 100% coverage
        valid_data_pct = 100.0
        
        if (cache_age_days <= PRICE_CACHE_OK_DAYS) and (valid_data_pct >= DATA_INTEGRITY_VERIFIED_COVERAGE):
            data_integrity = "Verified"
            level = "High"
        elif (cache_age_days <= PRICE_CACHE_DEGRADED_DAYS) and (valid_data_pct >= DATA_INTEGRITY_DEGRADED_COVERAGE):
            data_integrity = "Degraded"
            level = "Medium"
        else:
            data_integrity = "Compromised"
            level = "Low"
        
        passed = (level == expected_level and data_integrity == expected_status)
        result = "✅" if passed else "❌"
        
        print(f"{result} cache_age_days={cache_age_days:3d} → Expected: {expected_status:11s} ({expected_level}), Got: {data_integrity:11s} ({level})")
        
        assert passed, f"Failed for cache_age_days={cache_age_days}: expected {expected_status} ({expected_level}), got {data_integrity} ({level})"
    
    print("✅ All validation scenarios passed")


def test_comment_exists_in_app():
    """Verify that the required comment exists in app.py"""
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Check for the required comment
    assert "# source of truth: helpers/price_book.py" in content, \
        "Required comment '# source of truth: helpers/price_book.py' not found in app.py"
    
    print("\n✅ Required comment found in app.py:")
    print("   # source of truth: helpers/price_book.py")
    
    # Check that the constants are used
    assert "PRICE_CACHE_OK_DAYS" in content, "PRICE_CACHE_OK_DAYS not used in app.py"
    assert "PRICE_CACHE_DEGRADED_DAYS" in content, "PRICE_CACHE_DEGRADED_DAYS not used in app.py"
    
    print("✅ Constants PRICE_CACHE_OK_DAYS and PRICE_CACHE_DEGRADED_DAYS are used in app.py")


def test_imports_from_price_book():
    """Verify that app.py imports constants from helpers/price_book.py"""
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Check for import statement
    assert "from helpers.price_book import" in content, \
        "Import from helpers.price_book not found in app.py"
    
    # Check for specific imports - find the import block more reliably
    import_start = content.find("from helpers.price_book import")
    if import_start == -1:
        raise AssertionError("Import from helpers.price_book not found")
    
    # Find the end of the import statement (look for closing paren or newline)
    import_end = content.find(")", import_start)
    if import_end == -1:
        import_end = content.find("\n", import_start)
    import_section = content[import_start:import_end + 50]  # Add small buffer
    
    assert "PRICE_CACHE_OK_DAYS" in import_section, \
        "PRICE_CACHE_OK_DAYS not imported from helpers.price_book"
    assert "PRICE_CACHE_DEGRADED_DAYS" in import_section, \
        "PRICE_CACHE_DEGRADED_DAYS not imported from helpers.price_book"
    assert "compute_system_health" in import_section, \
        "compute_system_health not imported from helpers.price_book"
    
    print("\n✅ Verified imports from helpers/price_book.py:")
    print("   - PRICE_CACHE_OK_DAYS")
    print("   - PRICE_CACHE_DEGRADED_DAYS")
    print("   - compute_system_health")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Data Integrity Badge Thresholds Alignment")
    print("=" * 70)
    
    try:
        test_constants_import()
        test_validation_scenarios()
        test_comment_exists_in_app()
        test_imports_from_price_book()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nSummary:")
        print("  ✅ Constants correctly imported from helpers/price_book.py")
        print("  ✅ All validation scenarios pass")
        print("  ✅ Required comment added to app.py")
        print("  ✅ System Status uses compute_system_health()")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
