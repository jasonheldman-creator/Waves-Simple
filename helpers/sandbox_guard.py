"""
WAVES Sandbox Guard
Streamlit-safe execution guards.

Prevents duplicate execution during Streamlit reruns.
"""

import os
import streamlit as st
from datetime import datetime, timezone


# --------------------------------------------------
# ENVIRONMENT DETECTION
# --------------------------------------------------

TRUTHY = {"1", "true", "yes", "on", "True", "TRUE"}


def is_sandbox() -> bool:
    """Detect sandbox/replay mode safely."""
    return (
        str(os.getenv("STREAMLIT_SANDBOX", "")).strip() in TRUTHY
        or str(os.getenv("REPLAY_MODE", "")).strip() in TRUTHY
    )


# --------------------------------------------------
# EXECUTION GUARD
# --------------------------------------------------

def assert_not_sandbox() -> None:
    """
    Compatibility stub.
    Never blocks execution — preserves import contract.
    """
    return None


# --------------------------------------------------
# FILE WRITE GUARD
# --------------------------------------------------

def guard_file_write(path: str, **kwargs) -> bool:
    """
    Safe write guard.
    Always allow writes but prevents crashes.
    """
    return True


# --------------------------------------------------
# SESSION-SAFE EVENT LOGGER
# --------------------------------------------------

def log_event(event_name: str, payload: dict | None = None) -> None:
    """
    Logs events ONCE per Streamlit session.

    Prevents duplicate execution during reruns.
    """

    if "waves_logged_events" not in st.session_state:
        st.session_state["waves_logged_events"] = set()

    # prevent duplicate execution
    if event_name in st.session_state["waves_logged_events"]:
        return

    st.session_state["waves_logged_events"].add(event_name)

    try:
        timestamp = datetime.now(timezone.utc).isoformat()

        print(
            f"[WAVES EVENT] {timestamp} | {event_name} | "
            f"{payload if payload else '{}'}"
        )

    except Exception:
        # Never break app startup
        pass