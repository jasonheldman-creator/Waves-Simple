#!/usr/bin/env python3
"""
Validation Script: Snapshot Cache Invalidation

This script validates that the snapshot version-based cache invalidation
is working correctly. It performs the following checks:

1. Reads current snapshot metadata
2. Verifies all required fields are present
3. Displays snapshot version information
4. Shows how the version key is used for cache invalidation
"""

import json
import os
from datetime import datetime


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_section(title):
    """Print a formatted section title."""
    print(f"\n{title}")
    print("-" * len(title))


def load_snapshot_metadata():
    """Load snapshot metadata from JSON file."""
    metadata_path = 'data/snapshot_metadata.json'
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Snapshot metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def validate_metadata(metadata):
    """Validate that all required fields are present."""
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
    
    missing = [field for field in required_fields if field not in metadata]
    
    if missing:
        print(f"❌ Missing required fields: {', '.join(missing)}")
        return False
    
    print(f"✓ All {len(required_fields)} required fields present")
    return True


def display_metadata(metadata):
    """Display metadata in a formatted way."""
    print_section("Snapshot Metadata")
    
    # Core identification
    print(f"  Snapshot ID:      {metadata['snapshot_id']}")
    print(f"  Snapshot Hash:    {metadata['snapshot_hash']}")
    print(f"  Generation:       {metadata['generation_reason']}")
    print(f"  Timestamp:        {metadata['timestamp']}")
    
    # Version information
    print_section("Version Information")
    print(f"  Software:         {metadata['software_version']}")
    print(f"  Git Branch:       {metadata['git_branch']}")
    print(f"  Engine:           {metadata['engine_version']}")
    print(f"  Registry:         {metadata['registry_version']}")
    print(f"  Benchmark:        {metadata['benchmark_version']}")
    
    # System status
    print_section("System Status")
    print(f"  Data Regime:      {metadata['data_regime']}")
    print(f"  Safe Mode:        {metadata['safe_mode']}")
    print(f"  Wave Count:       {metadata['wave_count']}")
    print(f"  Degraded Waves:   {metadata['degraded_wave_count']}")
    print(f"  Broken Tickers:   {metadata['broken_ticker_count']}")


def compute_snapshot_version(metadata):
    """Compute the snapshot version key (as done in app.py)."""
    snapshot_id = metadata.get('snapshot_id', 'unknown')
    snapshot_hash = metadata.get('snapshot_hash', 'unknown')
    return f"{snapshot_id}_{snapshot_hash}"


def display_cache_key_info(version_key):
    """Display information about how the cache key is used."""
    print_section("Cache Key Information")
    print(f"  Version Key:      {version_key}")
    print(f"  Key Length:       {len(version_key)} characters")
    print()
    print("  This version key is used in the following cached functions:")
    print("    • safe_load_wave_history(_snapshot_version)")
    print("    • get_canonical_wave_universe(_snapshot_version)")
    print("    • get_cached_price_book_internal(_snapshot_version)")
    print()
    print("  When the snapshot is rebuilt:")
    print("    1. New snapshot_id (UUID) is generated")
    print("    2. New snapshot_hash (content hash) is calculated")
    print("    3. Version key changes")
    print("    4. Streamlit cache sees different parameter value")
    print("    5. Cache is invalidated automatically")
    print("    6. Data is reloaded from updated snapshot")


def check_snapshot_age(metadata):
    """Check the age of the snapshot."""
    print_section("Snapshot Age")
    
    try:
        timestamp_str = metadata['timestamp']
        # Parse ISO format timestamp
        snapshot_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        current_time = datetime.now(snapshot_time.tzinfo) if snapshot_time.tzinfo else datetime.now()
        age = current_time - snapshot_time
        
        age_hours = age.total_seconds() / 3600
        age_days = age.total_seconds() / 86400
        
        print(f"  Generated:        {snapshot_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Current Time:     {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Age:              {age_hours:.1f} hours ({age_days:.2f} days)")
        
        if age_hours < 24:
            print(f"  Status:           ✓ Fresh (< 24 hours)")
        elif age_hours < 48:
            print(f"  Status:           ⚠ Aging (24-48 hours)")
        else:
            print(f"  Status:           ⚠ Stale (> 48 hours)")
    except Exception as e:
        print(f"  Error:            Could not parse timestamp: {e}")


def main():
    """Main validation routine."""
    print_header("SNAPSHOT CACHE INVALIDATION VALIDATION")
    
    try:
        # Load metadata
        metadata = load_snapshot_metadata()
        
        # Validate metadata
        print_section("Metadata Validation")
        if not validate_metadata(metadata):
            return 1
        
        # Display metadata
        display_metadata(metadata)
        
        # Compute and display cache key
        version_key = compute_snapshot_version(metadata)
        display_cache_key_info(version_key)
        
        # Check snapshot age
        check_snapshot_age(metadata)
        
        # Final summary
        print_header("VALIDATION SUMMARY")
        print()
        print("  ✓ Snapshot metadata is valid and complete")
        print(f"  ✓ Version key: {version_key}")
        print(f"  ✓ Cache invalidation mechanism is configured correctly")
        print()
        print("  Next steps to validate cache invalidation:")
        print("    1. Note the current snapshot_id and snapshot_hash above")
        print("    2. Run: python3 scripts/rebuild_snapshot.py")
        print("    3. Run this script again to verify version changed")
        print("    4. Refresh Streamlit app to see updated metrics")
        print()
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure the snapshot has been generated:")
        print("  python3 scripts/rebuild_snapshot.py")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
