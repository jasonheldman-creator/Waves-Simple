"""
WAVES Intelligence Boot (Streamlit-compatible)

Safe deterministic initialization layer.

Goals:
- Runs once per Streamlit session
- Never blocks app startup
- Registers intelligence producers safely
- Executes producers fail-open
- No UI rendering
"""

from __future__ import annotations

import importlib
import streamlit as st
from typing import Callable, Dict, Any, List


# ================================================================
# CONSTANTS
# ================================================================

BOOT_FLAG = "_waves_intelligence_boot_complete"
HEALTH_KEY = "_waves_intelligence_health"

REGISTRY_KEY = "_waves_registered_intelligence"
REGISTRY_RAN_FLAG = "_waves_registry_executed"


# ================================================================
# REGISTRY
# ================================================================

def register_intelligence(func: Callable) -> Callable:
    registry: List[Callable] = st.session_state.setdefault(
        REGISTRY_KEY, []
    )

    if func not in registry:
        registry.append(func)

    return func


def _run_registered_intelligence() -> None:
    if st.session_state.get(REGISTRY_RAN_FLAG):
        return

    for func in st.session_state.get(REGISTRY_KEY, []):
        try:
            func()
        except Exception:
            # Never break app execution
            pass

    st.session_state[REGISTRY_RAN_FLAG] = True


# ================================================================
# SESSION SETUP
# ================================================================

def _initialize_session_structures() -> Dict[str, Any]:
    health: Dict[str, Any] = {}

    st.session_state.setdefault("intelligence", {})
    st.session_state.setdefault("intelligence_registry", [])
    st.session_state.setdefault("intelligence_runtime", {})

    health["session_initialized"] = True
    return health


# ================================================================
# SAFE PRODUCER IMPORT
# ================================================================

def _safe_import_producers(health: Dict[str, Any]) -> None:
    """
    Import producers WITHOUT allowing syntax errors
    to crash Streamlit startup.
    """
    try:
        importlib.import_module("helpers.intelligence_producers")
        health["producers_imported"] = True
    except Exception as exc:
        # swallow ALL import failures
        health["producers_imported"] = False
        health["producer_error"] = str(exc)


# ================================================================
# PUBLIC BOOT
# ================================================================

def intelligence_boot() -> None:

    if st.session_state.get(BOOT_FLAG):
        return

    health: Dict[str, Any] = {"boot_started": True}

    try:
        # Initialize containers
        health.update(_initialize_session_structures())

        # Import producers safely
        _safe_import_producers(health)

        # Run registered producers
        _run_registered_intelligence()

        st.session_state[BOOT_FLAG] = True
        health["boot"] = "ok"

    except Exception as exc:
        # absolute fail-open guarantee
        health["boot"] = "error"
        health["error"] = str(exc)

    st.session_state[HEALTH_KEY] = health