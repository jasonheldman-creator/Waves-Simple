"""
Snapshot Version Helper

Provides utilities for cache invalidation based on snapshot metadata.
When a snapshot is rebuilt, the snapshot_id and snapshot_hash change,
allowing cached data loaders to detect and refresh their caches.

Usage:
    from helpers.snapshot_version import get_snapshot_version_key
    
    @st.cache_data
    def my_cached_loader(snapshot_version=None):
        # snapshot_version triggers cache invalidation
        return load_data()
    
    # In app code:
    version = get_snapshot_version_key()
    data = my_cached_loader(snapshot_version=version)
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_snapshot_version_key() -> str:
    """
    Get a unique version key for the current snapshot.
    
    Reads snapshot_id and snapshot_hash from data/snapshot_metadata.json
    and combines them into a single version string. This version changes
    whenever a snapshot is rebuilt, triggering cache invalidation.
    
    Returns:
        str: Combined version key in format "{snapshot_id}_{snapshot_hash}"
             Returns "unknown_unknown" if metadata file is not found or invalid.
    
    Example:
        >>> version = get_snapshot_version_key()
        >>> print(version)
        'snap-227bfd8d8a364c9b_84bb6b118fa2885d'
    """
    try:
        # Get metadata file path
        metadata_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'snapshot_metadata.json'
        )
        
        # Check if file exists
        if not os.path.exists(metadata_path):
            logger.warning(f"Snapshot metadata file not found: {metadata_path}")
            return "unknown_unknown"
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract snapshot_id and snapshot_hash
        snapshot_id = metadata.get('snapshot_id', 'unknown')
        snapshot_hash = metadata.get('snapshot_hash', 'unknown')
        
        # Combine into version key
        version_key = f"{snapshot_id}_{snapshot_hash}"
        
        logger.debug(f"Snapshot version key: {version_key}")
        return version_key
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse snapshot metadata JSON: {e}")
        return "unknown_unknown"
    except Exception as e:
        logger.error(f"Error reading snapshot metadata: {e}")
        return "unknown_unknown"


def get_snapshot_metadata() -> Optional[dict]:
    """
    Get full snapshot metadata dictionary.
    
    Returns:
        dict: Snapshot metadata including snapshot_id, snapshot_hash, timestamp, etc.
        None: If metadata file is not found or invalid.
    
    Example:
        >>> metadata = get_snapshot_metadata()
        >>> if metadata:
        >>>     print(f"Snapshot generated at: {metadata['timestamp']}")
        >>>     print(f"Wave count: {metadata['wave_count']}")
    """
    try:
        # Get metadata file path
        metadata_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'snapshot_metadata.json'
        )
        
        # Check if file exists
        if not os.path.exists(metadata_path):
            logger.warning(f"Snapshot metadata file not found: {metadata_path}")
            return None
        
        # Load and return metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse snapshot metadata JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading snapshot metadata: {e}")
        return None
