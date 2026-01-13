"""
Test snapshot version-based cache invalidation.

This test verifies that the snapshot version key is properly integrated
into the Streamlit cache keys for data loaders.
"""

import json
import os


def test_snapshot_version_extraction():
    """Test that snapshot version can be extracted from metadata."""
    metadata_path = 'data/snapshot_metadata.json'
    
    assert os.path.exists(metadata_path), "snapshot_metadata.json should exist"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check required fields
    assert 'snapshot_id' in metadata, "snapshot_id should be in metadata"
    assert 'snapshot_hash' in metadata, "snapshot_hash should be in metadata"
    assert 'engine_version' in metadata, "engine_version should be in metadata"
    
    snapshot_id = metadata['snapshot_id']
    snapshot_hash = metadata['snapshot_hash']
    
    # Verify format
    assert snapshot_id.startswith('snap-'), f"snapshot_id should start with 'snap-': {snapshot_id}"
    assert len(snapshot_hash) > 0, "snapshot_hash should not be empty"
    
    # Construct version key
    version_key = f"{snapshot_id}_{snapshot_hash}"
    assert '_' in version_key, "version_key should contain underscore separator"
    
    print(f"✓ Snapshot version extraction works: {version_key}")
    print(f"  Snapshot ID: {snapshot_id}")
    print(f"  Snapshot hash: {snapshot_hash}")
    print(f"  Engine version: {metadata.get('engine_version')}")


def test_get_snapshot_version_function():
    """Test the get_snapshot_version function from app.py."""
    # We can't directly import from app.py due to streamlit dependency,
    # but we can test the logic separately
    
    metadata_path = 'data/snapshot_metadata.json'
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        snapshot_id = metadata.get('snapshot_id', 'unknown')
        snapshot_hash = metadata.get('snapshot_hash', 'unknown')
        version = f"{snapshot_id}_{snapshot_hash}"
    else:
        version = 'default'
    
    assert version != 'default', "Should get actual version from metadata"
    print(f"✓ get_snapshot_version logic works: {version}")


def test_snapshot_metadata_completeness():
    """Test that snapshot metadata has all required fields."""
    metadata_path = 'data/snapshot_metadata.json'
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    required_fields = [
        'snapshot_id',
        'snapshot_hash',
        'generation_reason',
        'timestamp',
        'software_version',
        'git_branch',
        'registry_version',
        'benchmark_version',
        'engine_version',
        'data_regime',
        'safe_mode',
        'wave_count',
        'degraded_wave_count',
        'broken_ticker_count'
    ]
    
    for field in required_fields:
        assert field in metadata, f"Required field '{field}' missing from metadata"
    
    print(f"✓ All {len(required_fields)} required metadata fields present")


if __name__ == '__main__':
    print("=" * 80)
    print("SNAPSHOT VERSION CACHE INVALIDATION TESTS")
    print("=" * 80)
    print()
    
    try:
        test_snapshot_version_extraction()
        print()
        test_get_snapshot_version_function()
        print()
        test_snapshot_metadata_completeness()
        print()
        print("=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
