"""Strategy instructions management - stub implementation."""
import json
import os
from datetime import datetime, timezone

_DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "strategy_instructions.json")

_WINDOW_TIERS = {
    "urgent": {"label": "Urgent", "description": "4-hour escalation window", "urgency": "High"},
    "standard": {"label": "Standard", "description": "48-hour governance window", "urgency": "Medium"},
    "extended": {"label": "Extended", "description": "7-day review window", "urgency": "Low"},
}

_RESOLVED_STATUSES = {"Resolved", "Approved", "Rejected", "Expired"}


def load_instructions():
    """Load strategy instructions from data file."""
    try:
        with open(_DATA_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_instructions(instructions):
    """Persist strategy instructions to data file."""
    try:
        os.makedirs(os.path.dirname(_DATA_FILE), exist_ok=True)
        with open(_DATA_FILE, "w") as f:
            json.dump(instructions, f, indent=2)
    except Exception:
        pass


def process_expired_instructions():
    """Mark expired instructions as Expired in-place."""
    instructions = load_instructions()
    now = datetime.now(timezone.utc).isoformat()
    changed = False
    for instr in instructions:
        exp = instr.get("expires_at")
        if exp and instr.get("status") not in _RESOLVED_STATUSES:
            if exp < now:
                instr["status"] = "Expired"
                changed = True
    if changed:
        save_instructions(instructions)


def get_pending_instructions():
    """Return instructions that are not resolved/expired/rejected."""
    return [i for i in load_instructions() if i.get("status") not in _RESOLVED_STATUSES]


def compute_time_remaining(expires_at_str):
    """Return human-readable time remaining string."""
    if not expires_at_str:
        return "—"
    try:
        exp = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        delta = exp - datetime.now(timezone.utc)
        total_seconds = int(delta.total_seconds())
        if total_seconds <= 0:
            return "Expired"
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    except Exception:
        return "—"


def get_window_tier(window_type):
    """Return tier dict for a window type."""
    return _WINDOW_TIERS.get(str(window_type).lower(),
                              _WINDOW_TIERS["standard"]).copy()


def update_instruction_status(instruction_id, status, actor=None, notes=None):
    """Update status of a strategy instruction."""
    instructions = load_instructions()
    for instr in instructions:
        if instr.get("id") == instruction_id:
            instr["status"] = status
            if actor:
                instr["actor"] = actor
            if notes:
                instr["notes"] = notes
            break
    save_instructions(instructions)


def get_resolved_instructions():
    """Return instructions that are resolved/approved/rejected."""
    return [i for i in load_instructions() if i.get("status") in _RESOLVED_STATUSES]


def get_outcome_label(status):
    """Return human-readable outcome label for a status."""
    labels = {
        "Resolved": "Resolved",
        "Approved": "Approved",
        "Rejected": "Rejected",
        "Expired": "Expired (No Action)",
    }
    return labels.get(status, str(status))
