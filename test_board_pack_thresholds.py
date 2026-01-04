"""
Test to validate that generate_board_pack_html uses canonical constants
from helpers/price_book.py instead of hardcoded values.
"""
import re


def test_board_pack_uses_canonical_constants():
    """
    Verify that generate_board_pack_html function uses PRICE_CACHE_OK_DAYS
    and PRICE_CACHE_DEGRADED_DAYS instead of hardcoded values.
    """
    # Read the app.py file
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Find the generate_board_pack_html function
    function_pattern = r'def generate_board_pack_html\(\):.*?(?=\ndef |\nclass |\Z)'
    match = re.search(function_pattern, content, re.DOTALL)
    
    if not match:
        raise AssertionError("Could not find generate_board_pack_html function")
    
    function_body = match.group(0)
    
    # Check that the function contains the comment
    assert '# source of truth: helpers/price_book.py' in function_body, \
        "Missing comment '# source of truth: helpers/price_book.py'"
    
    # Check that the function uses PRICE_CACHE_OK_DAYS
    assert 'PRICE_CACHE_OK_DAYS' in function_body, \
        "Function does not use PRICE_CACHE_OK_DAYS constant"
    
    # Check that the function uses PRICE_CACHE_DEGRADED_DAYS
    assert 'PRICE_CACHE_DEGRADED_DAYS' in function_body, \
        "Function does not use PRICE_CACHE_DEGRADED_DAYS constant"
    
    # Find the Data Integrity section
    data_integrity_pattern = r'# Data Integrity Section.*?html \+= "</div>"'
    integrity_match = re.search(data_integrity_pattern, function_body, re.DOTALL)
    
    if not integrity_match:
        raise AssertionError("Could not find Data Integrity Section")
    
    data_integrity_section = integrity_match.group(0)
    
    # Verify no hardcoded threshold values (1 or 3) in the conditional checks
    # Allow for other uses of these numbers (like in strings), but not in comparisons
    hardcoded_pattern = r'data_age_days\s*<=\s*[13]\b'
    hardcoded_matches = re.findall(hardcoded_pattern, data_integrity_section)
    
    if hardcoded_matches:
        raise AssertionError(
            f"Found hardcoded threshold values in data_age_days comparisons: {hardcoded_matches}. "
            "Should use PRICE_CACHE_OK_DAYS and PRICE_CACHE_DEGRADED_DAYS instead."
        )
    
    print("✅ All checks passed!")
    print("   - Comment '# source of truth: helpers/price_book.py' is present")
    print("   - Function uses PRICE_CACHE_OK_DAYS constant")
    print("   - Function uses PRICE_CACHE_DEGRADED_DAYS constant")
    print("   - No hardcoded threshold values (1, 3) found in data_age_days comparisons")


if __name__ == '__main__':
    test_board_pack_uses_canonical_constants()
    print("\n✅ Test completed successfully!")
