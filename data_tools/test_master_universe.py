#!/usr/bin/env python3
"""
Test script to validate the master universe CSV.

Tests:
1. Required fields are present
2. Row counts fall within expected ranges
3. No duplicates exist after deduplication
4. Data types are valid
5. Deduplication report is valid
"""

import sys
import os
import csv
import json
from typing import Set, Tuple


# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_UNIVERSE_CSV = os.path.join(REPO_ROOT, "data", "master_universe.csv")
DEDUPE_REPORT = os.path.join(REPO_ROOT, "data_tools", "master_universe_dedupe_report.json")

# Configuration
REQUIRED_FIELDS = ["Ticker", "Company", "Sector"]
MIN_EXPECTED_ROWS = 200  # Minimum expected total rows
MAX_EXPECTED_ROWS = 5000  # Maximum expected total rows
MIN_CRYPTO_ROWS = 50  # Minimum cryptocurrencies
MIN_ETF_ROWS = 30  # Minimum ETFs


def test_file_exists():
    """Test that the master universe CSV exists."""
    print("=" * 70)
    print("TEST 1: File Exists")
    print("=" * 70)
    
    if not os.path.exists(MASTER_UNIVERSE_CSV):
        print(f"❌ FAILED: {MASTER_UNIVERSE_CSV} does not exist")
        return False
    
    print(f"✅ PASSED: {MASTER_UNIVERSE_CSV} exists")
    return True


def test_required_fields():
    """Test that all required fields are present in all rows."""
    print("\n" + "=" * 70)
    print("TEST 2: Required Fields")
    print("=" * 70)
    
    try:
        with open(MASTER_UNIVERSE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Check header has required fields
            missing_fields = set(REQUIRED_FIELDS) - set(fieldnames or [])
            if missing_fields:
                print(f"❌ FAILED: Missing required fields in header: {missing_fields}")
                return False
            
            # Check each row has required fields
            row_num = 1
            rows_with_missing = []
            for row in reader:
                row_num += 1
                for field in REQUIRED_FIELDS:
                    if not row.get(field, "").strip():
                        rows_with_missing.append((row_num, field, row.get("Ticker", "N/A")))
            
            if rows_with_missing:
                print(f"❌ FAILED: {len(rows_with_missing)} rows have missing required fields:")
                for rnum, field, ticker in rows_with_missing[:10]:  # Show first 10
                    print(f"   Row {rnum} (Ticker: {ticker}): missing '{field}'")
                if len(rows_with_missing) > 10:
                    print(f"   ... and {len(rows_with_missing) - 10} more")
                return False
            
            print(f"✅ PASSED: All rows have required fields: {REQUIRED_FIELDS}")
            return True
            
    except Exception as e:
        print(f"❌ FAILED: Error reading CSV: {e}")
        return False


def test_row_counts():
    """Test that row counts fall within expected ranges."""
    print("\n" + "=" * 70)
    print("TEST 3: Row Count Ranges")
    print("=" * 70)
    
    try:
        with open(MASTER_UNIVERSE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            total_rows = len(rows)
            print(f"Total rows: {total_rows}")
            
            # Count by asset class
            crypto_count = sum(1 for r in rows if "crypto" in r.get("Sector", "").lower() or 
                              any(kw in r.get("Sector", "").lower() for kw in 
                                  ["store of value", "smart contract", "defi", "stablecoin", "layer"]))
            etf_count = sum(1 for r in rows if "etf" in r.get("Sector", "").lower())
            equity_count = total_rows - crypto_count - etf_count
            
            print(f"Cryptocurrencies: {crypto_count}")
            print(f"ETFs: {etf_count}")
            print(f"Equities: {equity_count}")
            
            # Validate ranges
            all_passed = True
            
            if not (MIN_EXPECTED_ROWS <= total_rows <= MAX_EXPECTED_ROWS):
                print(f"❌ FAILED: Total rows {total_rows} not in range [{MIN_EXPECTED_ROWS}, {MAX_EXPECTED_ROWS}]")
                all_passed = False
            else:
                print(f"✅ Total rows in expected range")
            
            if crypto_count < MIN_CRYPTO_ROWS:
                print(f"❌ FAILED: Crypto count {crypto_count} below minimum {MIN_CRYPTO_ROWS}")
                all_passed = False
            else:
                print(f"✅ Crypto count above minimum ({MIN_CRYPTO_ROWS})")
            
            if etf_count < MIN_ETF_ROWS:
                print(f"❌ FAILED: ETF count {etf_count} below minimum {MIN_ETF_ROWS}")
                all_passed = False
            else:
                print(f"✅ ETF count above minimum ({MIN_ETF_ROWS})")
            
            return all_passed
            
    except Exception as e:
        print(f"❌ FAILED: Error reading CSV: {e}")
        return False


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol for comparison."""
    return symbol.strip().upper() if symbol else ""


def normalize_asset_class(sector: str) -> str:
    """Normalize asset class for comparison."""
    if not sector:
        return "Unknown"
    
    sector_lower = sector.lower()
    
    # Cryptocurrency
    crypto_keywords = ["store of value", "settlement", "smart contract", "layer 1", "layer 2",
                       "scaling", "defi", "exchange", "oracle", "gaming", "metaverse",
                       "payments", "remittance", "yield", "staking", "stablecoin", "nft", "cryptocurrency"]
    if any(kw in sector_lower for kw in crypto_keywords):
        return "Cryptocurrency"
    
    # ETF
    if "etf" in sector_lower:
        return "ETF"
    
    # Equity
    return "Equity"


def test_no_duplicates():
    """Test that no duplicates exist after deduplication."""
    print("\n" + "=" * 70)
    print("TEST 4: Zero Duplicates")
    print("=" * 70)
    
    try:
        with open(MASTER_UNIVERSE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Track unique keys
            seen_keys: Set[Tuple[str, str]] = set()
            duplicates = []
            
            for idx, row in enumerate(rows):
                symbol = row.get("Ticker", "")
                sector = row.get("Sector", "")
                
                norm_symbol = normalize_symbol(symbol)
                norm_asset_class = normalize_asset_class(sector)
                
                if not norm_symbol:
                    continue
                
                key = (norm_asset_class, norm_symbol)
                
                if key in seen_keys:
                    duplicates.append({
                        "row": idx + 2,  # +2 for 1-indexed and header
                        "symbol": symbol,
                        "company": row.get("Company", ""),
                        "sector": sector,
                        "normalized_key": key
                    })
                else:
                    seen_keys.add(key)
            
            if duplicates:
                print(f"❌ FAILED: Found {len(duplicates)} duplicate entries:")
                for dup in duplicates[:10]:  # Show first 10
                    print(f"   Row {dup['row']}: {dup['symbol']} ({dup['company']}) - {dup['sector']}")
                    print(f"      Normalized key: {dup['normalized_key']}")
                if len(duplicates) > 10:
                    print(f"   ... and {len(duplicates) - 10} more")
                return False
            
            print(f"✅ PASSED: No duplicates found (checked {len(seen_keys)} unique entries)")
            return True
            
    except Exception as e:
        print(f"❌ FAILED: Error checking duplicates: {e}")
        return False


def test_data_types():
    """Test that numeric fields have valid data types."""
    print("\n" + "=" * 70)
    print("TEST 5: Data Type Validation")
    print("=" * 70)
    
    try:
        with open(MASTER_UNIVERSE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            numeric_fields = ["Weight", "MarketValue", "Price"]
            invalid_values = []
            
            for idx, row in enumerate(rows):
                for field in numeric_fields:
                    value = row.get(field, "")
                    if value and value.strip():  # Only check non-empty values
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            invalid_values.append({
                                "row": idx + 2,
                                "field": field,
                                "value": value,
                                "ticker": row.get("Ticker", "N/A")
                            })
            
            if invalid_values:
                print(f"❌ FAILED: Found {len(invalid_values)} invalid numeric values:")
                for inv in invalid_values[:10]:  # Show first 10
                    print(f"   Row {inv['row']} (Ticker: {inv['ticker']}): "
                          f"{inv['field']} = '{inv['value']}' is not numeric")
                if len(invalid_values) > 10:
                    print(f"   ... and {len(invalid_values) - 10} more")
                return False
            
            print(f"✅ PASSED: All numeric fields have valid values")
            return True
            
    except Exception as e:
        print(f"❌ FAILED: Error validating data types: {e}")
        return False


def test_dedupe_report():
    """Test that the deduplication report is valid."""
    print("\n" + "=" * 70)
    print("TEST 6: Deduplication Report")
    print("=" * 70)
    
    if not os.path.exists(DEDUPE_REPORT):
        print(f"⚠️  WARNING: {DEDUPE_REPORT} does not exist")
        print("   This is expected if the build script hasn't been run yet.")
        return True  # Not a failure, just a warning
    
    try:
        with open(DEDUPE_REPORT, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Check required fields
        required_report_fields = ["timestamp", "input_count", "output_count", 
                                   "duplicates_removed", "collisions", "rules_applied"]
        missing = [f for f in required_report_fields if f not in report]
        
        if missing:
            print(f"❌ FAILED: Report missing required fields: {missing}")
            return False
        
        # Validate counts
        if report["output_count"] + report["duplicates_removed"] != report["input_count"]:
            print(f"❌ FAILED: Count mismatch in report:")
            print(f"   Input: {report['input_count']}")
            print(f"   Output: {report['output_count']}")
            print(f"   Duplicates removed: {report['duplicates_removed']}")
            print(f"   Expected: output + duplicates = input")
            return False
        
        print(f"✅ PASSED: Deduplication report is valid")
        print(f"   Input rows: {report['input_count']}")
        print(f"   Output rows: {report['output_count']}")
        print(f"   Duplicates removed: {report['duplicates_removed']}")
        print(f"   Collisions detected: {len(report['collisions'])}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ FAILED: Invalid JSON in report: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Error reading report: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MASTER UNIVERSE VALIDATION TESTS")
    print("=" * 70)
    
    tests = [
        test_file_exists,
        test_required_fields,
        test_row_counts,
        test_no_duplicates,
        test_data_types,
        test_dedupe_report,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
