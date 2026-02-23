# ---------------------------------------------------
# LOGGING STUB (REQUIRED BY APP BOOTSTRAP)
# ---------------------------------------------------

def log_event(event_type: str, payload: dict | None = None, **kwargs):
    """
    Governance-safe logging stub.

    The production system records governance events,
    but Streamlit Cloud runs in sandbox mode.

    This function intentionally performs a no-op while
    preserving the expected interface.
    """
    return {
        "logged": False,
        "event_type": event_type,
        "reason": "Sandbox mode — logging disabled"
    }