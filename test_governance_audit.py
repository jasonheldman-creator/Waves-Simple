"""
test_governance_audit.py

Tests for the Governance & Audit functionality including:
- Governance metadata module
- Snapshot immutability
- Metadata generation and storage
"""

import os
import json
import time
import unittest
from datetime import datetime

# Set up path
import sys
sys.path.insert(0, os.path.dirname(__file__))


class TestGovernanceMetadata(unittest.TestCase):
    """Test governance metadata functionality"""
    
    def test_import_governance_metadata(self):
        """Test that governance_metadata module can be imported"""
        try:
            import governance_metadata
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import governance_metadata: {e}")
    
    def test_get_current_governance_info(self):
        """Test getting current governance information"""
        from governance_metadata import get_current_governance_info
        
        info = get_current_governance_info()
        
        # Check required fields exist
        self.assertIn('platform_version', info)
        self.assertIn('git_branch', info)
        self.assertIn('data_regime', info)
        self.assertIn('wave_registry_version', info)
        self.assertIn('benchmark_registry_version', info)
        self.assertIn('safe_mode_status', info)
        self.assertIn('degraded_wave_count', info)
        self.assertIn('broken_ticker_count', info)
        self.assertIn('total_wave_count', info)
        
        # Check data types
        self.assertIsInstance(info['platform_version'], str)
        self.assertIsInstance(info['git_branch'], str)
        self.assertIn(info['data_regime'], ['LIVE', 'SANDBOX', 'HYBRID', 'UNKNOWN'])
        self.assertIn(info['safe_mode_status'], ['ON', 'OFF', 'UNKNOWN'])
        self.assertIsInstance(info['degraded_wave_count'], int)
        self.assertIsInstance(info['broken_ticker_count'], int)
        self.assertIsInstance(info['total_wave_count'], int)
    
    def test_snapshot_metadata_creation(self):
        """Test creating snapshot metadata"""
        import pandas as pd
        from governance_metadata import create_snapshot_metadata
        
        # Create a dummy snapshot DataFrame
        snapshot_df = pd.DataFrame({
            'Wave': ['Test Wave 1', 'Test Wave 2'],
            'Data_Regime_Tag': ['Full', 'Partial']
        })
        
        metadata = create_snapshot_metadata(
            snapshot_df,
            generation_reason='manual',
            snapshot_id='test-snapshot-001'
        )
        
        # Check required fields
        self.assertEqual(metadata['snapshot_id'], 'test-snapshot-001')
        self.assertEqual(metadata['generation_reason'], 'manual')
        self.assertIn('snapshot_hash', metadata)
        self.assertIn('software_version', metadata)
        self.assertIn('registry_version', metadata)
        self.assertIn('data_regime', metadata)
        self.assertEqual(metadata['wave_count'], 2)


class TestSnapshotImmutability(unittest.TestCase):
    """Test snapshot immutability"""
    
    def test_snapshot_id_uniqueness(self):
        """Test that each snapshot generation creates a unique ID"""
        from governance_metadata import generate_snapshot_id
        
        # Generate multiple snapshot IDs
        ids = [generate_snapshot_id() for _ in range(10)]
        
        # Check all are unique
        self.assertEqual(len(ids), len(set(ids)), "Snapshot IDs should be unique")
        
        # Check format
        for snap_id in ids:
            self.assertTrue(snap_id.startswith('snap-'), "Snapshot ID should start with 'snap-'")
            self.assertEqual(len(snap_id), 21, "Snapshot ID should be 21 characters (snap- + 16 hex)")
    
    def test_snapshot_hash_consistency(self):
        """Test that identical DataFrames produce identical hashes"""
        import pandas as pd
        from governance_metadata import calculate_snapshot_hash
        
        # Create identical DataFrames
        df1 = pd.DataFrame({
            'Wave': ['Wave 1', 'Wave 2'],
            'NAV': [100.0, 200.0]
        })
        
        df2 = pd.DataFrame({
            'Wave': ['Wave 1', 'Wave 2'],
            'NAV': [100.0, 200.0]
        })
        
        hash1 = calculate_snapshot_hash(df1)
        hash2 = calculate_snapshot_hash(df2)
        
        self.assertEqual(hash1, hash2, "Identical DataFrames should produce identical hashes")
    
    def test_snapshot_hash_difference(self):
        """Test that different DataFrames produce different hashes"""
        import pandas as pd
        from governance_metadata import calculate_snapshot_hash
        
        # Create different DataFrames
        df1 = pd.DataFrame({
            'Wave': ['Wave 1', 'Wave 2'],
            'NAV': [100.0, 200.0]
        })
        
        df2 = pd.DataFrame({
            'Wave': ['Wave 1', 'Wave 2'],
            'NAV': [100.0, 201.0]  # Different value
        })
        
        hash1 = calculate_snapshot_hash(df1)
        hash2 = calculate_snapshot_hash(df2)
        
        self.assertNotEqual(hash1, hash2, "Different DataFrames should produce different hashes")


class TestSnapshotLedgerIntegration(unittest.TestCase):
    """Test snapshot_ledger integration with governance metadata"""
    
    def test_snapshot_metadata_file_creation(self):
        """Test that snapshot generation creates metadata file"""
        # This test requires the actual data files to exist
        if not os.path.exists('data/wave_registry.csv'):
            self.skipTest("Wave registry not available")
        
        from snapshot_ledger import generate_snapshot
        
        # Generate a snapshot
        df = generate_snapshot(force_refresh=True, generation_reason='test')
        
        # Check metadata file was created
        metadata_file = 'data/snapshot_metadata.json'
        self.assertTrue(os.path.exists(metadata_file), "Metadata file should be created")
        
        # Load and validate metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check required fields
        self.assertIn('snapshot_id', metadata)
        self.assertIn('snapshot_hash', metadata)
        self.assertIn('generation_reason', metadata)
        self.assertEqual(metadata['generation_reason'], 'test')
        self.assertIn('software_version', metadata)
        self.assertIn('data_regime', metadata)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGovernanceMetadata))
    suite.addTests(loader.loadTestsFromTestCase(TestSnapshotImmutability))
    suite.addTests(loader.loadTestsFromTestCase(TestSnapshotLedgerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
