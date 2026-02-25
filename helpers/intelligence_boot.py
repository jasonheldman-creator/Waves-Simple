"""
WAVES Intelligence Boot (Streamlit-compatible)

Purpose
-------
Provides a deterministic initialization phase that runs once per
Streamlit session BEFORE any UI rendering occurs.

Design Goals
------------
- Import-safe (no UI rendering at import time)
- Runs once per Streamlit session
- Fail-open (never breaks the application)
- Initializes required session_state structures
- Provides internal health diagnostics (no UI output)
"""

from __future__ import annotations

import streamlit as st
from typing import Dict, Any


# -------------------------------------------------------------------
# INTERNAL CONSTANTS
# -------------------------------------------------------------------

BOOT_FLAG = "_waves_intelligence_boot_complete"
HEALTH_KEY = "_waves_intelligence_health"
ERROR_KEY = "_waves_intelligence_boot_error"


# -------------------------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------------------------

def _initialize_session_structures() -> Dict[str, Any]:
    """
    Ensure all core intelligence containers exist.

    Returns a health dictionary describing initialization results.
    """
    health: Dict[str, Any] = {}

    # Core intelligence containers
    st.session_state.setdefault("intelligence", {})
    st.session_state.setdefault("intelligence_registry", [])
    st.session_state.setdefault("intelligence_runtime", {})

    health["session_initialized"] = True
    health["registry_ready"] = True

    return health


# -------------------------------------------------------------------
# PUBLIC BOOT FUNCTION
# -------------------------------------------------------------------

def intelligence_boot() -> None:
    """
    Execute WAVES intelligence initialization once per session.

    MUST be called before any Streamlit rendering.
    Safe to call multiple times (idempotent).
    """

    # Prevent rerun execution
    if st.session_state.get(BOOT_FLAG):
        return

    health: Dict[str, Any] = {
        "boot_started": True,
    }

    try:
        # Initialize session containers
        health.update(_initialize_session_structures())

        # Mark boot success
        st.session_state[BOOT_FLAG] = True
        health["boot"] = "ok"

    except Exception as exc:  # Fail-open by design
        health["boot"] = "error"
        health["error"] = str(exc)
        st.session_state[ERROR_KEY] = str(exc)

    # Store internal diagnostics (never rendered automatically)
    st.session_state[HEALTH_KEY] = health