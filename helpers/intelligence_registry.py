"""
WAVES Intelligence Registry

Central execution layer for intelligence producers.

Purpose
-------
Provides a single deterministic place where intelligence
functions register themselves and execute once per session.

No UI output.
Fail-open design.
"""

from __future__ import annotations

import streamlit as st
from typing import Callable, List

REGISTRY_KEY = "_waves_registered_intelligence"
RAN_FLAG = "_waves_registry_executed"


def register_intelligence(func: Callable) -> Callable:
    """
    Decorator used to register intelligence producers.
    """

    registry: List[Callable] = st.session_state.setdefault(
        REGISTRY_KEY, []
    )

    if func not in registry:
        registry.append(func)

    return func


def run_registered_intelligence() -> None:
    """
    Executes all registered intelligence functions once.
    """

    if st.session_state.get(RAN_FLAG):
        return

    registry = st.session_state.get(REGISTRY_KEY, [])

    for func in registry:
        try:
            func()
        except Exception:
            # Fail-open by design
            pass

    st.session_state[RAN_FLAG] = True