"""
Snapshot Version Key Helper

Provides cache invalidation keys based on snapshot metadata.
Used by Streamlit @st.cache_data decorators to ensure cache refreshes
when snapshot data updates.

Usage:
    from helpers.snapshot_version import get_snapshot_version_key
    
    @st.cache_data
    def load_data(snapshot_version: str):
        # snapshot_version parameter forces cache invalidation on change
        return pd.read_csv("data/live_snapshot.csv")
    
    # In calling code:
    snapshot_version = get_snapshot_version_key()
    data = load_data(snapshot_version)
"""

import json
import os
from typing import Optional

SNAPSHOT_METADATA_FILE = "data/snapshot_metadata.json"


def get_snapshot_version_key() -> str:
    """
    Get a version key for snapshot-based cache invalidation.
    
    Reads snapshot_metadata.json and combines snapshot_id and snapshot_hash
    into a single version string. When snapshot is regenerated, this key
    changes, invalidating any Streamlit caches that depend on it.
    
    Returns:
        Version key string (e.g., "snap-227bfd8d8a364c9b:84bb6b118fa2885d")
        If metadata file doesn't exist or is invalid, returns "unknown:0"
        
    Example:
        >>> version = get_snapshot_version_key()
        >>> print(version)
        'snap-227bfd8d8a364c9b:84bb6b118fa2885d'
    """
    try:
        if not os.path.exists(SNAPSHOT_METADATA_FILE):
            return "unknown:0"
        
        with open(SNAPSHOT_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        snapshot_id = metadata.get('snapshot_id', 'unknown')
        snapshot_hash = metadata.get('snapshot_hash', '0')
        
        return f"{snapshot_id}:{snapshot_hash}"
    
    except (json.JSONDecodeError, IOError, KeyError) as e:
        # If metadata file is corrupt or unreadable, return fallback
        return "unknown:0"


def get_snapshot_metadata() -> dict:
    """
    Get full snapshot metadata dictionary.
    
    Returns:
        Dictionary with snapshot metadata fields:
        - snapshot_id: Unique identifier for this snapshot generation
        - snapshot_hash: Hash of snapshot content
        - timestamp: Generation timestamp
        - engine_version: Engine version used
        - wave_count: Number of waves in snapshot
        - etc.
        
        Returns empty dict if metadata file doesn't exist or is invalid.
    """
    try:
        if not os.path.exists(SNAPSHOT_METADATA_FILE):
            return {}
        
        with open(SNAPSHOT_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    except (json.JSONDecodeError, IOError) as e:
        return {}
