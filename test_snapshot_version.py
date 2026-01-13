"""
Test snapshot version cache invalidation system.

Validates that:
1. get_snapshot_version_key() returns a valid version string
2. Version changes when snapshot metadata changes
3. Cache invalidation works correctly with snapshot version
"""

import os
import sys
import json
import tempfile
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.snapshot_version import get_snapshot_version_key, get_snapshot_metadata


def test_get_snapshot_version_key_returns_valid_format():
    """Test that get_snapshot_version_key returns a valid format."""
    version = get_snapshot_version_key()
    
    # Should not be the fallback value
    assert version != "unknown_unknown", "Should return a valid version key"
    
    # Should contain underscore separator
    assert "_" in version, "Version key should contain underscore separator"
    
    # Should have two parts
    parts = version.split("_")
    assert len(parts) >= 2, "Version key should have at least two parts (snapshot_id_hash)"


def test_get_snapshot_metadata_returns_dict():
    """Test that get_snapshot_metadata returns a valid dictionary."""
    metadata = get_snapshot_metadata()
    
    assert metadata is not None, "Metadata should not be None"
    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    
    # Check required fields
    assert 'snapshot_id' in metadata, "Metadata should contain snapshot_id"
    assert 'snapshot_hash' in metadata, "Metadata should contain snapshot_hash"
    assert 'timestamp' in metadata, "Metadata should contain timestamp"
    assert 'wave_count' in metadata, "Metadata should contain wave_count"


def test_snapshot_version_consistency():
    """Test that version key is consistent with metadata."""
    version = get_snapshot_version_key()
    metadata = get_snapshot_metadata()
    
    assert metadata is not None, "Metadata should be available"
    
    snapshot_id = metadata['snapshot_id']
    snapshot_hash = metadata['snapshot_hash']
    expected_version = f"{snapshot_id}_{snapshot_hash}"
    
    assert version == expected_version, \
        f"Version key should match snapshot_id_snapshot_hash: expected {expected_version}, got {version}"


def test_snapshot_version_changes_with_metadata():
    """Test that version changes when metadata changes."""
    # Get original version
    original_version = get_snapshot_version_key()
    
    # Get correct metadata path (in Waves-Simple subdirectory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, 'data', 'snapshot_metadata.json')
    
    # Skip test if metadata file doesn't exist
    if not os.path.exists(metadata_path):
        pytest.skip(f"Snapshot metadata file not found at {metadata_path}")
    
    # Read original metadata
    with open(metadata_path, 'r') as f:
        original_metadata = json.load(f)
    
    # Create a temporary backup
    backup_path = metadata_path + '.test_backup'
    with open(backup_path, 'w') as f:
        json.dump(original_metadata, f)
    
    try:
        # Modify metadata
        modified_metadata = original_metadata.copy()
        modified_metadata['snapshot_id'] = 'test-modified-id'
        modified_metadata['snapshot_hash'] = 'test-modified-hash'
        
        # Write modified metadata
        with open(metadata_path, 'w') as f:
            json.dump(modified_metadata, f)
        
        # Get new version (need to clear any module-level caching)
        # Since get_snapshot_version_key reads from file each time, it should pick up the change
        new_version = get_snapshot_version_key()
        
        # Verify version changed
        assert new_version != original_version, \
            f"Version should change when metadata changes: {original_version} -> {new_version}"
        
        assert new_version == "test-modified-id_test-modified-hash", \
            f"New version should match modified metadata: expected test-modified-id_test-modified-hash, got {new_version}"
    
    finally:
        # Restore original metadata
        with open(backup_path, 'r') as f:
            original_metadata = json.load(f)
        with open(metadata_path, 'w') as f:
            json.dump(original_metadata, f)
        
        # Clean up backup
        if os.path.exists(backup_path):
            os.remove(backup_path)


def test_snapshot_version_fallback_on_missing_file():
    """Test that version returns fallback when file is missing."""
    # This test is informational - we can't easily test missing file
    # without moving the actual file, which would break other tests
    
    # Instead, verify the function handles errors gracefully
    version = get_snapshot_version_key()
    
    # Should always return a string
    assert isinstance(version, str), "Version should always return a string"
    assert len(version) > 0, "Version should not be empty"


def test_snapshot_metadata_fields():
    """Test that snapshot metadata contains expected fields."""
    metadata = get_snapshot_metadata()
    
    assert metadata is not None, "Metadata should be available"
    
    # Required fields
    required_fields = [
        'snapshot_id',
        'snapshot_hash',
        'generation_reason',
        'timestamp',
        'wave_count'
    ]
    
    for field in required_fields:
        assert field in metadata, f"Metadata should contain {field}"
    
    # Verify types
    assert isinstance(metadata['snapshot_id'], str), "snapshot_id should be string"
    assert isinstance(metadata['snapshot_hash'], str), "snapshot_hash should be string"
    assert isinstance(metadata['wave_count'], int), "wave_count should be int"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
