"""Policy hash utilities - stub implementation."""
import hashlib
import json
import os
from datetime import datetime, timezone

_POLICY_VERSION = "1.0.0"
_POLICY_FILES = [
    "wave_config.csv",
    "wave_weights.csv",
    "config/",
]


def compute_policy_hash():
    """Return a short hex hash representing the current policy state."""
    h = hashlib.sha256()
    base = os.path.join(os.path.dirname(__file__), "..")
    for rel_path in _POLICY_FILES:
        full = os.path.join(base, rel_path)
        try:
            if os.path.isfile(full):
                with open(full, "rb") as f:
                    h.update(f.read())
            elif os.path.isdir(full):
                for fname in sorted(os.listdir(full)):
                    fpath = os.path.join(full, fname)
                    if os.path.isfile(fpath):
                        with open(fpath, "rb") as f:
                            h.update(f.read())
        except Exception:
            pass
    return h.hexdigest()[:12]


def get_policy_version():
    """Return the current policy version string."""
    return _POLICY_VERSION
