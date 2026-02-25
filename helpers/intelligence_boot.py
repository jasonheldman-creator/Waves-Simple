"""
WAVES Intelligence Boot (SAFE RESET VERSION)

Purpose:
Provide a minimal, guaranteed-safe initialization layer.

This version intentionally avoids dynamic imports so the
application can always start successfully.
"""

from __future__ import annotations

import streamlit as st
from typing import Callable, List


BOOT_FLAG = "_waves_intelligence_boot_complete"
REGISTRY_KEY = "_waves_registered_intelligence"


# ------------------------------------------------------------
# REGISTRY DECORATOR
# ------------------------------------------------------------

def register_intelligence(func: Callable) -> Callable:
    registry: List[Callable] = st.session_state.setdefault(
        REGISTRY_KEY, []
    )

    if func not in registry:
        registry.append(func)

    return func


# ------------------------------------------------------------
# BOOT
# ------------------------------------------------------------

def intelligence_boot() -> None:
    """
    Minimal safe boot.
    Never imports external modules.
    Never crashes.
    """

    if st.session_state.get(BOOT_FLAG):
        return

    # Ensure required containers exist
    st.session_state.setdefault("intelligence", {})
    st.session_state.setdefault("intelligence_runtime", {})
    st.session_state.setdefault(REGISTRY_KEY, [])

    # mark boot complete
    st.session_state[BOOT_FLAG] = True