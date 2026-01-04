#!/usr/bin/env python3
"""
Test suite for diagnostics module

Tests that the diagnostics module imports cleanly and prevents syntax errors.
Also tests the failed tickers diagnostics functionality.
"""

import sys
import os
import tempfile
import csv


def test_diagnostics_import():
    """Test that diagnostics.data_contact module imports without syntax errors."""
    print("Testing diagnostics.data_contact import...")
    
    try:
        import diagnostics.data_contact
        print("✅ diagnostics.data_contact imported successfully")
        return True
        
    except SyntaxError as e:
        print(f"❌ SyntaxError when importing diagnostics.data_contact: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"❌ Error when importing diagnostics.data_contact: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_diagnostics_classes():
    """Test that diagnostics classes can be instantiated."""
    print("\nTesting diagnostics classes...")
    
    try:
        from diagnostics.data_contact import SimArtifacts, SnapshotArtifacts, DefinitionArtifacts, diagnostics_support
        
        # Test SimArtifacts
        sim = SimArtifacts()
        assert sim is not None, "SimArtifacts should be instantiable"
        print("✅ SimArtifacts instantiated successfully")
        
        # Test SnapshotArtifacts
        snapshot = SnapshotArtifacts()
        assert snapshot is not None, "SnapshotArtifacts should be instantiable"
        print("✅ SnapshotArtifacts instantiated successfully")
        
        # Test DefinitionArtifacts
        definition = DefinitionArtifacts()
        assert definition is not None, "DefinitionArtifacts should be instantiable"
        print("✅ DefinitionArtifacts instantiated successfully")
        
        # Test diagnostics_support function
        assert callable(diagnostics_support), "diagnostics_support should be callable"
        diagnostics_support()  # Should not raise an error
        print("✅ diagnostics_support function is callable")
        
        print("✅ All diagnostics classes test passed")
        return True
        
    except Exception as e:
        print(f"❌ Diagnostics classes test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_load_broken_tickers():
    """Test loading broken tickers from CSV."""
    print("\nTesting load_broken_tickers_from_csv...")
    
    try:
        from helpers.ticker_diagnostics import load_broken_tickers_from_csv
        
        # Test with actual file if it exists
        if os.path.exists("data/broken_tickers.csv"):
            tickers = load_broken_tickers_from_csv("data/broken_tickers.csv")
            assert isinstance(tickers, list), "Should return a list"
            print(f"✅ Loaded {len(tickers)} broken tickers from data/broken_tickers.csv")
            
            # Validate structure if tickers exist
            if tickers:
                ticker = tickers[0]
                assert 'ticker_original' in ticker, "Should have ticker_original field"
                assert 'failure_type' in ticker, "Should have failure_type field"
                assert 'failure_count' in ticker, "Should have failure_count field"
                assert 'impacted_waves' in ticker, "Should have impacted_waves field"
                print("✅ Ticker structure validation passed")
                
                # Check top 50 logic
                top_50 = tickers[:50]
                assert len(top_50) <= 50, "Top 50 should have at most 50 items"
                print(f"✅ Top 50 logic works (got {len(top_50)} items)")
        else:
            print("⚠️  data/broken_tickers.csv not found, skipping actual file test")
        
        # Test with non-existent file (should return empty list)
        tickers = load_broken_tickers_from_csv("nonexistent_file.csv")
        assert tickers == [], "Should return empty list for non-existent file"
        print("✅ Non-existent file handling works correctly")
        
        # Create a temporary test CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['ticker_original', 'ticker_normalized', 'failure_type', 'error_message', 
                           'impacted_waves', 'suggested_fix', 'first_seen', 'last_seen', 'is_fatal'])
            writer.writerow(['AAPL', 'AAPL', 'PROVIDER_EMPTY', 'Empty data returned', 
                           'Wave1, Wave2, Wave3', 'Check data provider', 
                           '2026-01-01T10:00:00', '2026-01-01T11:00:00', 'True'])
            writer.writerow(['MSFT', 'MSFT', 'NETWORK_TIMEOUT', 'Connection timeout', 
                           'Wave1', 'Retry with backoff', 
                           '2026-01-01T10:00:00', '2026-01-01T11:00:00', 'True'])
            temp_file = f.name
        
        try:
            # Test loading from temp file
            tickers = load_broken_tickers_from_csv(temp_file)
            assert len(tickers) == 2, "Should load 2 tickers"
            assert tickers[0]['ticker_original'] == 'AAPL', "First ticker should be AAPL"
            assert tickers[0]['failure_count'] == 3, "AAPL should have 3 impacted waves"
            assert tickers[1]['failure_count'] == 1, "MSFT should have 1 impacted wave"
            # Check sorting (AAPL with 3 waves should come before MSFT with 1 wave)
            assert tickers[0]['failure_count'] >= tickers[1]['failure_count'], "Should be sorted by failure count desc"
            print("✅ Test CSV loading and sorting works correctly")
        finally:
            os.unlink(temp_file)
        
        print("✅ load_broken_tickers_from_csv test passed")
        return True
        
    except Exception as e:
        print(f"❌ load_broken_tickers_from_csv test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_export_failed_tickers():
    """Test exporting failed tickers to cache."""
    print("\nTesting export_failed_tickers_to_cache...")
    
    try:
        from helpers.ticker_diagnostics import export_failed_tickers_to_cache
        
        # Create test data
        test_data = [
            {
                'ticker_original': 'TEST1',
                'ticker_normalized': 'TEST1',
                'failure_type': 'PROVIDER_EMPTY',
                'error_message': 'Test error',
                'failure_count': 2,
                'impacted_waves': ['Wave1', 'Wave2'],
                'impacted_waves_str': 'Wave1, Wave2',
                'suggested_fix': 'Test fix',
                'first_seen': '2026-01-01T10:00:00',
                'last_seen': '2026-01-01T11:00:00',
                'is_fatal': True
            }
        ]
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_failed_tickers.csv')
            
            # Test export
            result = export_failed_tickers_to_cache(test_data, output_path)
            assert result == True, "Export should succeed"
            assert os.path.exists(output_path), "Output file should exist"
            print("✅ Export succeeded")
            
            # Verify CSV content
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1, "Should have 1 row"
                assert rows[0]['ticker_original'] == 'TEST1', "Ticker should be TEST1"
                assert rows[0]['failure_count'] == '2', "Failure count should be 2"
                print("✅ CSV content verification passed")
        
        # Test with empty data
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'empty_failed_tickers.csv')
            result = export_failed_tickers_to_cache([], output_path)
            assert result == True, "Export should succeed with empty data"
            assert os.path.exists(output_path), "Output file should exist"
            
            # Should have header only
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1, "Should have header only"
                print("✅ Empty data export works correctly")
        
        print("✅ export_failed_tickers_to_cache test passed")
        return True
        
    except Exception as e:
        print(f"❌ export_failed_tickers_to_cache test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostics tests."""
    print("\n🧪 Diagnostics Module Tests\n")
    print("="*60)
    
    results = []
    
    # Test import
    results.append(test_diagnostics_import())
    
    # Test classes
    results.append(test_diagnostics_classes())
    
    # Test failed tickers functionality
    results.append(test_load_broken_tickers())
    results.append(test_export_failed_tickers())
    
    print("\n" + "="*60)
    
    if all(results):
        print("✅ All diagnostics tests passed")
        return 0
    else:
        print("❌ Some diagnostics tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
