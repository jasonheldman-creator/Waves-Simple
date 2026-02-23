"""
sandbox_guard.py
WAVES Intelligence™ — Governance Safety Layer

Purpose:
Provide sandbox / replay safety controls so the application
can run in Streamlit Cloud without execution capability.

This module intentionally contains SAFE STUBS.
No trading, allocation, or execution logic exists here.
"""

# ---------------------------------------------------
# GLOBAL MODE FLAGS
# ---------------------------------------------------

SANDBOX_MODE = True
REPLAY_MODE = False


# ---------------------------------------------------
# MODE HELPERS
# ---------------------------------------------------

def is_sandbox() -> bool:
    """
    Returns True when running in sandbox mode.
    Used across UI and governance layers.
    """
    return SANDBOX_MODE


def is_replay_mode() -> bool:
    """
    Indicates replay/testing environment.
    """
    return REPLAY_MODE


# ---------------------------------------------------
# GOVERNANCE SAFETY GUARDS
# ---------------------------------------------------

def execution_allowed() -> bool:
    """
    Execution is NEVER allowed in Streamlit deployment.
    """
    return False


def trading_enabled() -> bool:
    """
    Explicitly disables trading paths.
    """
    return False


def allocation_changes_allowed() -> bool:
    """
    Prevents portfolio allocation execution.
    """
    return False


# ---------------------------------------------------
# SAFE NO-OP ACTION HANDLERS
# ---------------------------------------------------

def guard_action(*args, **kwargs):
    """
    Placeholder guard used by governance calls.
    Performs no action.
    """
    return {
        "status": "blocked",
        "reason": "Sandbox mode — execution disabled"
    }


def approve_action(*args, **kwargs):
    """
    Simulation-only approval handler.
    """
    return {"status": "approved_simulation"}


def reject_action(*args, **kwargs):
    return {"status": "rejected_simulation"}


def defer_action(*args, **kwargs):
    return {"status": "deferred_simulation"}


# ---------------------------------------------------
# SYSTEM STATUS
# ---------------------------------------------------

def sandbox_status():
    return {
        "sandbox": SANDBOX_MODE,
        "replay": REPLAY_MODE,
        "execution_enabled": False,
    }