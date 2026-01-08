"""
Test Portfolio Snapshot Aggregation in Force Ledger Recompute Pipeline

This test validates that the portfolio snapshot aggregation is correctly
integrated into the force_ledger_recompute pipeline.
"""

import unittest
import os
import sys
import json

# Disable network fetches for testing
os.environ['PRICE_FETCH_ENABLED'] = 'false'
os.environ['ALLOW_NETWORK_FETCH'] = 'false'


class TestPortfolioSnapshotAggregation(unittest.TestCase):
    """Test portfolio snapshot aggregation in the ledger recompute pipeline."""
    
    def test_portfolio_snapshot_file_created(self):
        """Test that portfolio snapshot file is created during force_ledger_recompute."""
        from helpers.operator_toolbox import force_ledger_recompute
        
        # Execute force_ledger_recompute
        success, message, details = force_ledger_recompute()
        
        # Verify recompute succeeded
        self.assertTrue(success, f"force_ledger_recompute should succeed: {message}")
        
        # Verify portfolio snapshot file was created
        snapshot_cache_path = os.path.join(os.getcwd(), 'data', 'cache', 'portfolio_snapshot.json')
        self.assertTrue(
            os.path.exists(snapshot_cache_path),
            "Portfolio snapshot cache file should exist after force_ledger_recompute"
        )
        
        print(f"✓ Portfolio snapshot file created: {snapshot_cache_path}")
    
    def test_portfolio_snapshot_content_valid(self):
        """Test that portfolio snapshot file contains valid data."""
        snapshot_cache_path = os.path.join(os.getcwd(), 'data', 'cache', 'portfolio_snapshot.json')
        
        # Verify file exists
        self.assertTrue(
            os.path.exists(snapshot_cache_path),
            "Portfolio snapshot cache file should exist"
        )
        
        # Load and validate content
        with open(snapshot_cache_path, 'r') as f:
            snapshot_data = json.load(f)
        
        # Verify required fields are present
        required_fields = [
            'success', 'mode', 'portfolio_returns', 'benchmark_returns',
            'alphas', 'wave_count', 'date_range', 'latest_date',
            'has_portfolio_returns_series', 'has_portfolio_benchmark_series',
            'timestamp'
        ]
        
        for field in required_fields:
            self.assertIn(
                field, snapshot_data,
                f"Portfolio snapshot should contain '{field}' field"
            )
        
        # Verify success flag
        self.assertTrue(
            snapshot_data['success'],
            "Portfolio snapshot success flag should be True"
        )
        
        # Verify wave count is reasonable
        self.assertGreater(
            snapshot_data['wave_count'], 0,
            "Wave count should be greater than 0"
        )
        
        # Verify mode is 'Standard'
        self.assertEqual(
            snapshot_data['mode'], 'Standard',
            "Portfolio snapshot mode should be 'Standard'"
        )
        
        # Verify returns and alphas have expected periods
        expected_periods = ['1D', '30D', '60D', '365D']
        for period in expected_periods:
            self.assertIn(
                period, snapshot_data['portfolio_returns'],
                f"portfolio_returns should have {period} key"
            )
            self.assertIn(
                period, snapshot_data['benchmark_returns'],
                f"benchmark_returns should have {period} key"
            )
            self.assertIn(
                period, snapshot_data['alphas'],
                f"alphas should have {period} key"
            )
        
        print(f"✓ Portfolio snapshot content is valid")
        print(f"  Wave count: {snapshot_data['wave_count']}")
        print(f"  Latest date: {snapshot_data['latest_date']}")
        print(f"  Mode: {snapshot_data['mode']}")
    
    def test_metadata_includes_portfolio_snapshot(self):
        """Test that metadata file includes portfolio snapshot information."""
        from helpers.operator_toolbox import force_ledger_recompute
        
        # Execute force_ledger_recompute
        success, message, details = force_ledger_recompute()
        
        # Verify recompute succeeded
        self.assertTrue(success, f"force_ledger_recompute should succeed: {message}")
        
        # Verify metadata file exists
        metadata_path = os.path.join(os.getcwd(), 'data', 'cache', 'data_health_metadata.json')
        self.assertTrue(
            os.path.exists(metadata_path),
            "Metadata file should exist after force_ledger_recompute"
        )
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify portfolio snapshot fields are present
        portfolio_snapshot_fields = [
            'portfolio_snapshot_success',
            'portfolio_snapshot_latest_date',
            'portfolio_snapshot_wave_count'
        ]
        
        for field in portfolio_snapshot_fields:
            self.assertIn(
                field, metadata,
                f"Metadata should contain '{field}' field"
            )
        
        # Verify portfolio snapshot succeeded
        self.assertTrue(
            metadata['portfolio_snapshot_success'],
            "portfolio_snapshot_success should be True in metadata"
        )
        
        print(f"✓ Metadata includes portfolio snapshot information")
        print(f"  Portfolio snapshot success: {metadata['portfolio_snapshot_success']}")
        print(f"  Portfolio snapshot wave count: {metadata['portfolio_snapshot_wave_count']}")
        print(f"  Portfolio snapshot latest date: {metadata['portfolio_snapshot_latest_date']}")
    
    def test_details_dict_includes_portfolio_snapshot(self):
        """Test that details dict from force_ledger_recompute includes portfolio snapshot info."""
        from helpers.operator_toolbox import force_ledger_recompute
        
        # Execute force_ledger_recompute
        success, message, details = force_ledger_recompute()
        
        # Verify recompute succeeded
        self.assertTrue(success, f"force_ledger_recompute should succeed: {message}")
        
        # Verify details dict contains portfolio snapshot fields
        self.assertIn(
            'portfolio_snapshot_success', details,
            "details dict should contain 'portfolio_snapshot_success'"
        )
        self.assertIn(
            'portfolio_snapshot_wave_count', details,
            "details dict should contain 'portfolio_snapshot_wave_count'"
        )
        self.assertIn(
            'portfolio_snapshot_latest_date', details,
            "details dict should contain 'portfolio_snapshot_latest_date'"
        )
        
        # Verify portfolio snapshot succeeded
        self.assertTrue(
            details['portfolio_snapshot_success'],
            "portfolio_snapshot_success should be True in details dict"
        )
        
        print(f"✓ Details dict includes portfolio snapshot information")
        print(f"  Keys: {list(details.keys())}")
    
    def test_portfolio_snapshot_dates_align_with_ledger(self):
        """Test that portfolio snapshot dates align with ledger dates."""
        from helpers.operator_toolbox import force_ledger_recompute
        
        # Execute force_ledger_recompute
        success, message, details = force_ledger_recompute()
        
        # Verify recompute succeeded
        self.assertTrue(success, f"force_ledger_recompute should succeed: {message}")
        
        # Get dates from details
        ledger_max_date = details.get('ledger_max_date')
        portfolio_snapshot_latest_date = details.get('portfolio_snapshot_latest_date')
        
        self.assertIsNotNone(ledger_max_date, "ledger_max_date should be present in details")
        self.assertIsNotNone(
            portfolio_snapshot_latest_date,
            "portfolio_snapshot_latest_date should be present in details"
        )
        
        # Verify dates match
        self.assertEqual(
            ledger_max_date, portfolio_snapshot_latest_date,
            "Ledger max date should match portfolio snapshot latest date"
        )
        
        print(f"✓ Portfolio snapshot dates align with ledger dates")
        print(f"  Ledger max date: {ledger_max_date}")
        print(f"  Portfolio snapshot latest date: {portfolio_snapshot_latest_date}")


def run_tests():
    """Run all tests and print results."""
    print("=" * 70)
    print("PORTFOLIO SNAPSHOT AGGREGATION TEST SUITE")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPortfolioSnapshotAggregation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print()
        print("Validated:")
        print("  - Portfolio snapshot file is created during force_ledger_recompute")
        print("  - Portfolio snapshot file contains valid data")
        print("  - Metadata includes portfolio snapshot information")
        print("  - Details dict includes portfolio snapshot information")
        print("  - Portfolio snapshot dates align with ledger dates")
        return 0
    else:
        if result.failures:
            print(f"❌ {len(result.failures)} test(s) failed")
        if result.errors:
            print(f"❌ {len(result.errors)} test(s) had errors")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
