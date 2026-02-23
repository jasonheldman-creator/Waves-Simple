import os
from datetime import datetime


TRUTHY = {"1", "true", "yes", "on", "sandbox"}


def is_sandbox() -> bool:
    """
    Detect whether app is running in sandbox/replay mode.
    Never raises.
    """
    try:
        return (
            str(os.getenv("STREAMLIT_SANDBOX", "")).lower() in TRUTHY
            or str(os.getenv("REPLAY_MODE", "")).lower() in TRUTHY
        )
    except Exception:
        return False


def assert_not_sandbox():
    """
    Compatibility guard — intentionally non-blocking.
    """
    return True


def guard_file_write(path: str, **kwargs) -> bool:
    """
    Allow writes but keep interface stable.
    """
    return True


def log_event(event: str, **kwargs) -> None:
    """
    Lightweight event logger (no-op safe).
    """
    try:
        _ = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            **kwargs,
        }
    except Exception:
        pass