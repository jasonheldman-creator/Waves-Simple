"""
auto_refresh_config.py
SAFE MODE: no import-time execution
"""

def configure_auto_refresh(st):
    """
    Explicitly invoked from app.py AFTER UI is initialized.
    """
    if "AUTO_REFRESH_ENABLED" not in st.session_state:
        st.session_state["AUTO_REFRESH_ENABLED"] = False

    # Optional future logic goes here