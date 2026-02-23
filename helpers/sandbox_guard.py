"""
Temporary compatibility stub for sandbox guard.

Provides required symbols so app_min.py can import safely.
No execution or governance logic enabled.
"""

# ---- MODE FLAGS ----
SANDBOX_MODE = True
REPLAY_MODE = False


# ---- SAFE NO-OP FUNCTIONS ----
def sandbox_guard(*args, **kwargs):
    return True


def guard_action(*args, **kwargs):
    return True


def is_action_allowed(*args, **kwargs):
    return True


def validate_governance_action(*args, **kwargs):
    return True