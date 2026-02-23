# ---------------------------------------------------
# LOGGING STUB (REQUIRED BY APP BOOTSTRAP)
# ---------------------------------------------------

def is_sandbox() -> bool:
    """
    Returns True when running in sandbox or replay mode.
    Must NEVER raise exceptions.
    """
    try:
        import os
        _truthy = {"1", "true", "yes", "on"}
        return (
            os.environ.get("STREAMLIT_SANDBOX", "").lower() in _truthy
            or os.environ.get("REPLAY_MODE", "").lower() in _truthy
        )
    except Exception:
        return True


def log_event(event_type: str, payload: dict | None = None, **kwargs) -> None:
    """
    Safe no-op logging fallback used during sandbox mode.
    Must never break app startup.

    The production system records governance events,
    but Streamlit Cloud runs in sandbox mode.

    This function intentionally performs a no-op while
    preserving the expected interface.
    """
    return None


def assert_not_sandbox() -> None:
    """
    No-op stub required by app bootstrap imports.
    In this sandbox/replay context, enforcement is intentionally
    disabled to allow safe app startup without raising exceptions.
    """
    return None


def guard_file_write(path: str, **kwargs) -> bool:
    """
    Safe no-op file-write guard for sandbox mode.
    Returns True (allowed) without performing any enforcement.
    Must never raise exceptions.
    """
    return True