"""
WAVES Intelligence Boot (Streamlit-compatible)

Purpose
-------
Provides a deterministic initialization phase that runs once per
Streamlit session BEFORE any UI rendering occurs.

This module:
- Initializes required session_state containers
- Executes registered intelligence producers
- Never renders UI
- Never breaks the app (fail-open design)

SAFE TO CALL MULTIPLE TIMES.
"""

from __future__ import annotations

import streamlit as st
from typing import Callable, Dict, Any, List


# ================================================================
# INTERNAL CONSTANTS
# ================================================================

BOOT_FLAG = "_waves_intelligence_boot_complete"
HEALTH_KEY = "_waves_intelligence_health"
ERROR_KEY = "_waves_intelligence_boot_error"

REGISTRY_KEY = "_waves_registered_intelligence"
REGISTRY_RAN_FLAG = "_waves_registry_executed"


# ================================================================
# REGISTRY SYSTEM
# ================================================================

def register_intelligence(func: Callable) -> Callable:
    """
    Decorator used by intelligence producers.

    Example:
        @register_intelligence
        def build_alpha_state():
            st.session_state["alpha_state"] = ...
    """
    registry: List[Callable] = st.session_state.setdefault(
        REGISTRY_KEY, []
    )

    if func not in registry:
        registry.append(func)

    return func


def _run_registered_intelligence() -> None:
    """
    Execute all registered intelligence functions once per session.
    Fail-open by design.
    """

    if st.session_state.get(REGISTRY_RAN_FLAG):
        return

    registry = st.session_state.get(REGISTRY_KEY, [])

    for func in registry:
        try:
            func()
        except Exception:
            # Never interrupt application execution
            pass

    st.session_state[REGISTRY_RAN_FLAG] = True


# ================================================================
# SESSION INITIALIZATION
# ================================================================

def _initialize_session_structures() -> Dict[str, Any]:
    """
    Ensure all core intelligence containers exist.
    """
    health: Dict[str, Any] = {}

    st.session_state.setdefault("intelligence", {})
    st.session_state.setdefault("intelligence_registry", [])
    st.session_state.setdefault("intelligence_runtime", {})

    health["session_initialized"] = True
    health["registry_ready"] = True

    return health


# ================================================================
# PUBLIC BOOT FUNCTION
# ================================================================

def intelligence_boot() -> None:
    """
    Run WAVES intelligence boot sequence.

    MUST be called before any Streamlit rendering.
    Safe across reruns (idempotent).
    """

    # Prevent duplicate execution during reruns
    if st.session_state.get(BOOT_FLAG):
        return

    health: Dict[str, Any] = {"boot_started": True}

    try:
        # Initialize required containers
        health.update(_initialize_session_structures())

        # Execute registered intelligence producers
        _run_registered_intelligence()

        # Mark successful boot
        st.session_state[BOOT_FLAG] = True
        health["boot"] = "ok"

    except Exception as exc:
        # Fail-open — never break app rendering
        health["boot"] = "error"
        health["error"] = str(exc)
        st.session_state[ERROR_KEY] = str(exc)

    # Store internal diagnostics (not rendered automatically)
    st.session_state[HEALTH_KEY] = health