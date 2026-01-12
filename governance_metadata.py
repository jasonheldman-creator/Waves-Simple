"""
governance_metadata.py

GOVERNANCE & AUDIT METADATA LAYER

This module provides comprehensive governance and audit tracking for the 
Waves-Simple application. It tracks snapshot metadata, version information,
and system health metrics for transparency and accountability.

Key Features:
- Immutable snapshot identification (UUID, content hash)
- Software and registry version tracking
- Data regime detection (LIVE/SANDBOX/HYBRID)
- Safe mode status tracking
- Degraded waves and broken ticker counts
- Snapshot provenance tracking
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


# Constants
DATA_DIR = "data"
SNAPSHOT_FILE = "data/live_snapshot.csv"
WAVE_REGISTRY_FILE = "data/wave_registry.csv"
BROKEN_TICKERS_FILE = "data/broken_tickers.csv"
DIAGNOSTICS_FILE = "data/diagnostics_run.json"


def get_software_version() -> str:
    """
    Get the software version from git commit hash.
    
    Returns:
        Git commit hash (short) or 'unknown' if not available
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return os.environ.get('BUILD_ID', 'unknown')
    except Exception:
        return os.environ.get('BUILD_ID', 'unknown')


def get_git_branch() -> str:
    """
    Get the current git branch name.
    
    Returns:
        Branch name or 'unknown' if not available
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return 'unknown'
    except Exception:
        return 'unknown'


def get_registry_version() -> str:
    """
    Get the wave registry version based on last modification time.
    
    Returns:
        Version string in format 'v{timestamp}' or 'unknown' if file not found
    """
    try:
        if os.path.exists(WAVE_REGISTRY_FILE):
            # Read registry and check updated_at field
            df = pd.read_csv(WAVE_REGISTRY_FILE)
            if 'updated_at' in df.columns and not df.empty:
                # Get the most recent update timestamp
                latest_update = pd.to_datetime(df['updated_at']).max()
                return f"v{latest_update.strftime('%Y%m%d_%H%M%S')}"
            
            # Fallback to file modification time
            mtime = os.path.getmtime(WAVE_REGISTRY_FILE)
            timestamp = datetime.fromtimestamp(mtime)
            return f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"
        return 'unknown'
    except Exception as e:
        print(f"Warning: Could not determine registry version: {e}")
        return 'unknown'


def get_benchmark_registry_version() -> str:
    """
    Get the benchmark registry version.
    
    For now, this is derived from wave_registry since benchmarks are defined there.
    In the future, this could point to a separate benchmark configuration file.
    
    Returns:
        Version string
    """
    # Benchmarks are defined in wave_registry.csv (benchmark_spec column)
    # So we use the same version as the wave registry
    return get_registry_version()


def calculate_snapshot_hash(snapshot_df: pd.DataFrame) -> str:
    """
    Calculate a content hash of the snapshot data for integrity verification.
    
    Args:
        snapshot_df: Snapshot DataFrame
        
    Returns:
        SHA-256 hash of the snapshot content
    """
    try:
        # Convert DataFrame to CSV string (without index)
        csv_content = snapshot_df.to_csv(index=False)
        
        # Calculate SHA-256 hash
        hash_object = hashlib.sha256(csv_content.encode('utf-8'))
        return hash_object.hexdigest()[:16]  # Use first 16 chars for brevity
    except Exception as e:
        print(f"Warning: Could not calculate snapshot hash: {e}")
        return 'unknown'


def detect_data_regime(snapshot_df: Optional[pd.DataFrame] = None) -> str:
    """
    Detect the current data regime based on snapshot coverage.
    
    Args:
        snapshot_df: Optional snapshot DataFrame (will load if not provided)
        
    Returns:
        'LIVE' if all waves have full data
        'SANDBOX' if majority of waves have no/limited data
        'HYBRID' if mix of full and partial data
    """
    try:
        if snapshot_df is None:
            if os.path.exists(SNAPSHOT_FILE):
                snapshot_df = pd.read_csv(SNAPSHOT_FILE)
            else:
                return 'UNKNOWN'
        
        if snapshot_df.empty:
            return 'UNKNOWN'
        
        # Check Data_Regime_Tag column if available
        if 'Data_Regime_Tag' in snapshot_df.columns:
            regime_counts = snapshot_df['Data_Regime_Tag'].value_counts()
            
            total_waves = len(snapshot_df)
            full_count = regime_counts.get('Full', 0)
            partial_count = regime_counts.get('Partial', 0)
            unavailable_count = regime_counts.get('Unavailable', 0)
            
            # LIVE: >80% full data
            if full_count / total_waves > 0.8:
                return 'LIVE'
            # SANDBOX: >50% unavailable or limited data
            elif (unavailable_count + partial_count) / total_waves > 0.7:
                return 'SANDBOX'
            # HYBRID: mix of full and partial
            else:
                return 'HYBRID'
        
        return 'UNKNOWN'
    except Exception as e:
        print(f"Warning: Could not detect data regime: {e}")
        return 'UNKNOWN'


def get_safe_mode_status() -> str:
    """
    Get the current Safe Mode status.
    
    Returns:
        'ON', 'OFF', or 'UNKNOWN'
    """
    try:
        # Check if Safe Mode is enabled by looking for environment variable or config
        # This is a placeholder - adjust based on actual Safe Mode implementation
        safe_mode_env = os.environ.get('SAFE_MODE', 'false').lower()
        
        if safe_mode_env in ['true', '1', 'on']:
            return 'ON'
        else:
            return 'OFF'
    except Exception:
        return 'UNKNOWN'


def count_degraded_waves(snapshot_df: Optional[pd.DataFrame] = None) -> int:
    """
    Count the number of degraded or partial waves.
    
    Args:
        snapshot_df: Optional snapshot DataFrame (will load if not provided)
        
    Returns:
        Count of waves with 'Partial' or degraded data regime
    """
    try:
        if snapshot_df is None:
            if os.path.exists(SNAPSHOT_FILE):
                snapshot_df = pd.read_csv(SNAPSHOT_FILE)
            else:
                return 0
        
        if snapshot_df.empty:
            return 0
        
        if 'Data_Regime_Tag' in snapshot_df.columns:
            # Count Partial and Unavailable as degraded
            partial_count = (snapshot_df['Data_Regime_Tag'] == 'Partial').sum()
            return int(partial_count)
        
        return 0
    except Exception as e:
        print(f"Warning: Could not count degraded waves: {e}")
        return 0


def count_broken_tickers() -> int:
    """
    Count the number of broken tickers.
    
    Returns:
        Count of broken tickers from broken_tickers.csv
    """
    try:
        if os.path.exists(BROKEN_TICKERS_FILE):
            df = pd.read_csv(BROKEN_TICKERS_FILE)
            # Count unique tickers (excluding header)
            if not df.empty and 'ticker' in df.columns:
                return len(df)
            elif not df.empty:
                # If no 'ticker' column, count rows
                return len(df)
        return 0
    except Exception as e:
        print(f"Warning: Could not count broken tickers: {e}")
        return 0


def get_last_successful_snapshot_time() -> Optional[datetime]:
    """
    Get the timestamp of the last successful full snapshot.
    
    Returns:
        Datetime of last snapshot or None if not available
    """
    try:
        if os.path.exists(SNAPSHOT_FILE):
            # Read the snapshot date from the CSV
            df = pd.read_csv(SNAPSHOT_FILE)
            if 'Date' in df.columns and not df.empty:
                snapshot_date = pd.to_datetime(df['Date'].iloc[0])
                return snapshot_date
            
            # Fallback to file modification time
            mtime = os.path.getmtime(SNAPSHOT_FILE)
            return datetime.fromtimestamp(mtime)
        return None
    except Exception as e:
        print(f"Warning: Could not get snapshot time: {e}")
        return None


def generate_snapshot_id() -> str:
    """
    Generate a unique snapshot ID (UUID).
    
    Returns:
        UUID string in format 'snap-{uuid}' using 16 hex characters for better uniqueness
    """
    return f"snap-{uuid.uuid4().hex[:16]}"


def create_snapshot_metadata(
    snapshot_df: pd.DataFrame,
    generation_reason: str = 'auto',
    snapshot_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for a snapshot.
    
    Args:
        snapshot_df: The snapshot DataFrame
        generation_reason: Reason for generation ('auto', 'manual', 'fallback', 'version_change')
        snapshot_id: Optional snapshot ID (will generate if not provided)
        
    Returns:
        Dictionary with complete snapshot metadata
    """
    if snapshot_id is None:
        snapshot_id = generate_snapshot_id()
    
    # Calculate snapshot hash
    snapshot_hash = calculate_snapshot_hash(snapshot_df)
    
    # Get version information
    software_version = get_software_version()
    git_branch = get_git_branch()
    registry_version = get_registry_version()
    benchmark_version = get_benchmark_registry_version()
    
    # Get engine version
    engine_version = 'unknown'
    try:
        from waves_engine import get_engine_version
        engine_version = get_engine_version()
    except ImportError:
        pass
    
    # Get system status
    data_regime = detect_data_regime(snapshot_df)
    safe_mode = get_safe_mode_status()
    degraded_count = count_degraded_waves(snapshot_df)
    broken_tickers_count = count_broken_tickers()
    
    # Get snapshot timestamp
    snapshot_time = datetime.now()
    
    metadata = {
        # Snapshot identification
        'snapshot_id': snapshot_id,
        'snapshot_hash': snapshot_hash,
        'generation_reason': generation_reason,
        'timestamp': snapshot_time.isoformat(),
        
        # Version information
        'software_version': software_version,
        'git_branch': git_branch,
        'registry_version': registry_version,
        'benchmark_version': benchmark_version,
        'engine_version': engine_version,  # NEW: Track engine version for cache invalidation
        
        # System status
        'data_regime': data_regime,
        'safe_mode': safe_mode,
        'wave_count': len(snapshot_df),
        'degraded_wave_count': degraded_count,
        'broken_ticker_count': broken_tickers_count,
    }
    
    return metadata


def get_current_governance_info() -> Dict[str, Any]:
    """
    Get current governance and audit information for display.
    
    Returns:
        Dictionary with governance metrics
    """
    # Load current snapshot if available
    snapshot_df = None
    try:
        if os.path.exists(SNAPSHOT_FILE):
            snapshot_df = pd.read_csv(SNAPSHOT_FILE)
    except Exception:
        pass
    
    # Get snapshot metadata (may be limited if no snapshot exists)
    metadata = {
        'platform_version': get_software_version(),
        'git_branch': get_git_branch(),
        'snapshot_timestamp': get_last_successful_snapshot_time(),
        'data_regime': detect_data_regime(snapshot_df),
        'wave_registry_version': get_registry_version(),
        'benchmark_registry_version': get_benchmark_registry_version(),
        'safe_mode_status': get_safe_mode_status(),
        'degraded_wave_count': count_degraded_waves(snapshot_df),
        'broken_ticker_count': count_broken_tickers(),
        'total_wave_count': len(snapshot_df) if snapshot_df is not None else 0,
    }
    
    return metadata
