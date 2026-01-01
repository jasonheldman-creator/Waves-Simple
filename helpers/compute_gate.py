"""
compute_gate.py

Shared Compute Gate Mechanism - Infinite Loop Prevention

This module provides a centralized gating mechanism to prevent infinite rebuild loops
and unnecessary recomputation. All data fetching and heavy compute operations should
be guarded by this gate.

STEP 3: Shared Compute Gate Mechanism

Rules for allowing builds:
1. User explicitly clicked a button in the current Run ID, OR
2. Snapshot is missing and no build has been attempted in the last 10 minutes, OR
3. Snapshot is stale and no build has been attempted in the last 10 minutes

Otherwise, return cached results if available.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Constants
BUILD_COOLDOWN_MINUTES = 10  # Minimum time between automatic build attempts
SNAPSHOT_STALE_THRESHOLD_MINUTES = 60  # Consider snapshot stale after this time


def should_allow_build(
    snapshot_path: str,
    session_state: Optional[Dict] = None,
    build_key: str = "default",
    explicit_button_click: bool = False
) -> Tuple[bool, str]:
    """
    Determine whether a build/compute operation should be allowed.
    
    This is the central gate that prevents infinite rebuild loops.
    
    IMPORTANT: Auto-rebuilds on stale snapshots are DISABLED.
    Only explicit user actions trigger rebuilds.
    
    Args:
        snapshot_path: Path to the snapshot file (e.g., "data/live_snapshot.csv")
        session_state: Streamlit session_state dict (if available)
        build_key: Unique key for this build operation (e.g., "engine_snapshot", "planb_snapshot")
        explicit_button_click: True if user explicitly clicked a rebuild button
        
    Returns:
        Tuple of (should_build: bool, reason: str)
    """
    # ========================================================================
    # STEP 1: Safe Mode Check - Global Kill Switch (NEW)
    # ========================================================================
    if session_state is not None and session_state.get("safe_mode_no_fetch", True):
        # Only allow builds if explicit button click
        if not explicit_button_click:
            return False, "Safe Mode active - all auto-builds suppressed"
    
    # ========================================================================
    # STEP 2: SAFE DEMO MODE Check - Global Kill Switch (LEGACY)
    # ========================================================================
    if session_state is not None and session_state.get("safe_demo_mode", False):
        return False, "SAFE DEMO MODE active - all builds suppressed"
    
    # ========================================================================
    # STEP 3: ONE RUN ONLY Check - Prevent background operations after initial load
    # ========================================================================
    if session_state is not None and session_state.get("one_run_only_block", False):
        # Only allow builds if explicit button click
        if not explicit_button_click:
            return False, "ONE RUN ONLY latch active - user interaction required"
    
    # ========================================================================
    # Rule 1: Explicit user action always allowed
    # ========================================================================
    if explicit_button_click:
        # Record this build attempt
        if session_state is not None:
            _record_build_attempt(session_state, build_key)
        return True, "Explicit user button click"
    
    # ========================================================================
    # Check build cooldown (STEP 6: Prevent state-driven rebuild loops)
    # ========================================================================
    if session_state is not None:
        # Check if build is already in progress
        if session_state.get(f"{build_key}_build_in_progress", False):
            return False, f"Build already in progress for {build_key}"
        
        # Check if build was completed this run
        current_run_id = session_state.get("run_id", 0)
        last_build_run = session_state.get(f"{build_key}_last_build_run_id")
        if last_build_run is not None and last_build_run == current_run_id:
            return False, f"Build already completed this run (run_id: {current_run_id})"
    
    # ========================================================================
    # Check snapshot status
    # ========================================================================
    snapshot_exists = os.path.exists(snapshot_path)
    
    # Rule 2: Snapshot doesn't exist - DO NOT auto-build, require user action
    # This prevents automatic rebuilds on missing snapshots
    if not snapshot_exists:
        return False, "Snapshot missing - manual rebuild required (click button to rebuild)"
    
    # Rule 3: Snapshot is stale - DO NOT auto-build, just mark as stale
    # This prevents automatic rebuilds on stale snapshots
    try:
        snapshot_age_minutes = _get_file_age_minutes(snapshot_path)
        if snapshot_age_minutes > SNAPSHOT_STALE_THRESHOLD_MINUTES:
            # Mark snapshot as stale in session state for banner display
            if session_state is not None:
                session_state[f"{build_key}_snapshot_stale"] = True
                session_state[f"{build_key}_snapshot_age_minutes"] = snapshot_age_minutes
            return False, f"Snapshot is stale ({snapshot_age_minutes:.1f}m old) - manual rebuild required"
    except Exception as e:
        logger.warning(f"Error checking snapshot age: {e}")
    
    # Otherwise, don't build
    return False, f"Snapshot is fresh - no build needed"


def _record_build_attempt(session_state: Dict, build_key: str):
    """Record that a build attempt is being made."""
    session_state[f"{build_key}_last_build_attempt"] = datetime.now()
    session_state[f"{build_key}_build_in_progress"] = True
    session_state[f"{build_key}_last_build_run_id"] = session_state.get("run_id", 0)


def mark_build_complete(session_state: Dict, build_key: str, success: bool = True):
    """Mark a build operation as complete."""
    if session_state is not None:
        session_state[f"{build_key}_build_in_progress"] = False
        session_state[f"{build_key}_last_build_success"] = success
        session_state[f"{build_key}_last_build_complete"] = datetime.now()


def get_build_diagnostics(session_state: Dict, build_key: str) -> Dict:
    """
    Get diagnostics information for a specific build key.
    
    Returns:
        Dictionary with diagnostic information
    """
    if session_state is None:
        return {}
    
    last_build_attempt = session_state.get(f"{build_key}_last_build_attempt")
    last_build_complete = session_state.get(f"{build_key}_last_build_complete")
    
    return {
        "build_key": build_key,
        "build_in_progress": session_state.get(f"{build_key}_build_in_progress", False),
        "last_build_attempt": last_build_attempt.isoformat() if last_build_attempt else None,
        "last_build_complete": last_build_complete.isoformat() if last_build_complete else None,
        "last_build_success": session_state.get(f"{build_key}_last_build_success"),
        "last_build_run_id": session_state.get(f"{build_key}_last_build_run_id"),
        "minutes_since_last_attempt": (
            (datetime.now() - last_build_attempt).total_seconds() / 60
            if last_build_attempt else None
        ),
    }


def _get_file_age_minutes(file_path: str) -> float:
    """Get age of a file in minutes."""
    if not os.path.exists(file_path):
        return float('inf')
    
    # Use consistent datetime approach with rest of module
    from datetime import datetime
    mtime_timestamp = os.path.getmtime(file_path)
    file_time = datetime.fromtimestamp(mtime_timestamp)
    age_seconds = (datetime.now() - file_time).total_seconds()
    return age_seconds / 60


def check_stale_snapshot(
    snapshot_path: str,
    session_state: Optional[Dict] = None,
    build_key: str = "default"
) -> Tuple[bool, float]:
    """
    Check if a snapshot is stale and return status.
    
    Args:
        snapshot_path: Path to the snapshot file
        session_state: Streamlit session_state dict (if available)
        build_key: Unique key for this build operation
        
    Returns:
        Tuple of (is_stale: bool, age_minutes: float)
    """
    if not os.path.exists(snapshot_path):
        return True, float('inf')
    
    try:
        age_minutes = _get_file_age_minutes(snapshot_path)
        is_stale = age_minutes > SNAPSHOT_STALE_THRESHOLD_MINUTES
        
        # Update session state
        if session_state is not None:
            session_state[f"{build_key}_snapshot_stale"] = is_stale
            session_state[f"{build_key}_snapshot_age_minutes"] = age_minutes
        
        return is_stale, age_minutes
    except Exception as e:
        logger.warning(f"Error checking snapshot staleness: {e}")
        return False, 0.0
