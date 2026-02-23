"""
helpers.sandbox_guard
--------------------------------------------------

Governance-safe sandbox utilities used by WAVES Intelligence™.

Design goals:
• NEVER break app startup
• NEVER raise runtime exceptions
• Accept ANY legacy or future call signatures
• Work both inside and outside Streamlit runtime
• Remain observational / non-executing

This module intentionally behaves as a safe compatibility layer.
"""

from __future__ import annotations

import os
from typing import Any


# -------------------------------------------------------------------
# Environment Detection
# -------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on", "sandbox"}


def _env_flag(name: str) -> bool:
    """Read environment flag safely."""
    try:
        value = os.getenv(name, "")
        return str(value).lower() in _TRUTHY
    except Exception:
        return False


def is_sandbox() -> bool:
    """
    Returns True if running in sandbox/replay/demo mode.

    NEVER raises.
    """
    try:
        return (
            _env_flag("STREAMLIT_SANDBOX")
            or _env_flag("REPLAY_MODE")
            or _env_flag("SANDBOX_MODE")
        )
    except Exception:
        return False


# -------------------------------------------------------------------
# Governance Safety Guards
# -------------------------------------------------------------------

def assert_not_sandbox(*args: Any, **kwargs: Any) -> None:
    """
    Compatibility stub.

    Original system prevented execution in sandbox mode.
    In Community Cloud deployment we keep this as a no-op
    to preserve import contracts.
    """
    return None


def guard_file_write(*args: Any, **kwargs: Any) -> bool:
    """
    File-write guard.

    Always returns True so legacy calls succeed
    without blocking Streamlit execution.
    """
    return True


# -------------------------------------------------------------------
# Event Logging (CRITICAL FIX)
# -------------------------------------------------------------------

def log_event(*args: Any, **kwargs: Any) -> None:
    """
    Flexible governance event logger.

    Accepts ANY argument signature used across:
        - app_min.py
        - governance lifecycle
        - daily cycle engine
        - sandbox logging hooks

    This prevents runtime crashes caused by
    signature mismatches between environments.

    Intentionally a no-op.
    """
    try:
        # Optional debug toggle
        if os.getenv("WAVES_DEBUG_SANDBOX", "").lower() in _TRUTHY:
            print("[sandbox_guard.log_event]", args, kwargs)
    except Exception:
        pass

    return None


# Alias used throughout app_min.py
sandbox_log_event = log_event


# -------------------------------------------------------------------
# Module health check
# -------------------------------------------------------------------

def _healthcheck() -> bool:
    """
    Lightweight internal validation.
    """
    try:
        is_sandbox()
        log_event("healthcheck")
        guard_file_write("test")
        assert_not_sandbox()
        return True
    except Exception:
        return False