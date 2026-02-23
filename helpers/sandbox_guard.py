"""
Compatibility sandbox guard stub.
Allows app_min.py to boot safely on Streamlit Cloud.

This file intentionally provides all expected exports as no-ops.
"""

# --------------------------------------------------
# MODE FLAGS
# --------------------------------------------------

SANDBOX_MODE = True
REPLAY_MODE = False


# --------------------------------------------------
# CORE GUARD OBJECT
# --------------------------------------------------

class SandboxGuard:
    def __init__(self, *args, **kwargs):
        pass

    def allow(self, *args, **kwargs):
        return True

    def validate(self, *args, **kwargs):
        return True

    def check(self, *args, **kwargs):
        return True


sandbox_guard = SandboxGuard()


# --------------------------------------------------
# FUNCTION EXPORTS (NO-OP SAFE)
# --------------------------------------------------

def guard_action(*args, **kwargs):
    return True


def is_action_allowed(*args, **kwargs):
    return True


def validate_governance_action(*args, **kwargs):
    return True


def get_sandbox_guard(*args, **kwargs):
    return sandbox_guard


def initialize_sandbox(*args, **kwargs):
    return sandbox_guard