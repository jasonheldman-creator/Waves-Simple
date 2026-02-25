"""
WAVES Intelligence Boot (Streamlit-compatible)

Requirements:
- Import-safe (no UI rendering on import)
- Runs once per Streamlit session
- Executes before any st.* rendering
- Fail-open (never break existing behavior)
"""

import streamlit as st

BOOT_FLAG = "_waves_intelligence_boot_complete"

def intelligence_boot():
    """
    Initialize intelligence runtime once per session.
    Must execute before any Streamlit rendering.
    """

    # Prevent execution during Streamlit reruns
    if st.session_state.get(BOOT_FLAG):
        return

    try:
        # Ensure required session structures always exist
        st.session_state.setdefault("intelligence", {})
        st.session_state.setdefault("intelligence_registry", [])
        st.session_state.setdefault("intelligence_runtime", {})

        # Mark boot complete
        st.session_state[BOOT_FLAG] = True

    except Exception as e:
        # Fail-open: never interrupt application rendering
        st.session_state["_waves_intelligence_boot_error"] = str(e)
