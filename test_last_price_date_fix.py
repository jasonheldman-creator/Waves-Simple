"""
Test suite for Last Price Date sidebar fix.

This test validates that the sidebar 'Last Price Date' is computed from 
the raw price_book dataframe, not from filtered/derived dataframes.
"""

import os
import sys
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_get_latest_data_timestamp_uses_price_book():
    """Test that get_latest_data_timestamp uses price_book instead of wave_history."""
    print("=" * 80)
    print("TEST: get_latest_data_timestamp uses PRICE_BOOK")
    print("=" * 80)
    
    # Import the function from app
    from app import get_latest_data_timestamp
    from helpers.price_book import get_price_book
    
    # Get the timestamp from the function
    sidebar_date = get_latest_data_timestamp()
    print(f"\n1. Sidebar Last Price Date: {sidebar_date}")
    
    # Get the max date directly from price_book
    try:
        price_book = get_price_book(active_tickers=None)
        if price_book is not None and not price_book.empty:
            price_book_max_date = price_book.index.max().strftime('%Y-%m-%d')
            print(f"2. Price Book Max Date: {price_book_max_date}")
            
            # They should match
            assert sidebar_date == price_book_max_date, \
                f"Sidebar date ({sidebar_date}) should match price_book max date ({price_book_max_date})"
            
            print(f"\n✅ SUCCESS: Sidebar Last Price Date matches Price Book max date")
            print(f"   Both values: {sidebar_date}")
        else:
            print("⚠️ WARNING: Price book is empty or None - cannot validate")
            # If price_book is empty, function should return "unknown"
            assert sidebar_date == "unknown", \
                f"When price_book is empty, should return 'unknown', got '{sidebar_date}'"
            print("✅ SUCCESS: Returns 'unknown' when price_book is empty")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise
    
    print("=" * 80)


def test_diagnostics_panel_includes_validation():
    """Test that the diagnostics panel would show the validation data."""
    print("\n" + "=" * 80)
    print("TEST: Diagnostics Panel Date Validation")
    print("=" * 80)
    
    # Import required functions
    from app import get_latest_data_timestamp
    from helpers.price_book import get_price_book
    
    try:
        # Simulate what the diagnostics panel does
        price_book = get_price_book(active_tickers=None)
        
        price_book_max_date = "N/A"
        if price_book is not None and not price_book.empty:
            price_book_max_date = price_book.index.max().strftime('%Y-%m-%d')
        
        sidebar_last_price_date_used = get_latest_data_timestamp()
        
        date_validation = {
            "price_book_max_date": price_book_max_date,
            "sidebar_last_price_date_used": sidebar_last_price_date_used,
            "dates_match": price_book_max_date == sidebar_last_price_date_used
        }
        
        print(f"\n1. Price Book Max Date: {date_validation['price_book_max_date']}")
        print(f"2. Sidebar Last Price Date Used: {date_validation['sidebar_last_price_date_used']}")
        print(f"3. Dates Match: {date_validation['dates_match']}")
        
        # Verify the structure is correct
        assert "price_book_max_date" in date_validation
        assert "sidebar_last_price_date_used" in date_validation
        assert "dates_match" in date_validation
        
        # If price_book has data, dates should match
        if price_book_max_date != "N/A":
            assert date_validation["dates_match"], \
                "Dates should match when price_book has data"
            print("\n✅ SUCCESS: Diagnostics validation structure is correct and dates match")
        else:
            print("\n✅ SUCCESS: Diagnostics validation structure is correct (price_book empty)")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "🧪 LAST PRICE DATE FIX - TEST SUITE" + "\n")
    
    try:
        test_get_latest_data_timestamp_uses_price_book()
        test_diagnostics_panel_includes_validation()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80 + "\n")
        
    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 80 + "\n")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        print("=" * 80 + "\n")
        traceback.print_exc()
        sys.exit(1)
