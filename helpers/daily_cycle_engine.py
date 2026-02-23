"""Daily cycle engine - stub implementation."""
import json
import os
from datetime import datetime, timezone

_STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "daily_cycle_state.json")

_DEFAULT_STATE = {
    "date": None,
    "morning_run": False,
    "midday_run": False,
    "close_run": False,
}


def load_cycle_state():
    """Load daily cycle state from disk."""
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return dict(_DEFAULT_STATE)


def save_cycle_state(state):
    """Persist daily cycle state to disk."""
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        with open(_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def reset_if_new_day(state):
    """Reset cycle flags if the date has changed; return (possibly updated) state."""
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("date") != today:
        state = {**_DEFAULT_STATE, "date": today}
    return state


def should_run_morning(cycle_state):
    """Return True if current UTC time is in the morning run window (09:00-09:30) and not yet run today."""
    now = datetime.now(timezone.utc)
    return (now.hour == 9 and now.minute < 30) and not cycle_state.get("morning_run")


def should_run_midday(cycle_state):
    """Return True if current UTC time is in the midday run window (13:00-13:30) and not yet run today."""
    now = datetime.now(timezone.utc)
    return (now.hour == 13 and now.minute < 30) and not cycle_state.get("midday_run")


def should_run_close(cycle_state):
    """Return True if current UTC time is in the close run window (20:00-20:30) and not yet run today."""
    now = datetime.now(timezone.utc)
    return (now.hour == 20 and now.minute < 30) and not cycle_state.get("close_run")


def run_cycle(phase=None, cycle_state=None, snapshot_df=None, attrib_df=None, review_signals=None, adaptive_state=None):
    """Run the appropriate daily cycle phase and return a status dict."""
    if cycle_state is None:
        cycle_state = load_cycle_state()
        cycle_state = reset_if_new_day(cycle_state)
    timestamp = datetime.now(timezone.utc).isoformat()

    if phase == "morning" or (phase is None and should_run_morning(cycle_state)):
        cycle_state["morning_run"] = True
        save_cycle_state(cycle_state)
        return {"status": "ok", "message": "Morning cycle executed.", "timestamp": timestamp}

    if phase == "midday" or (phase is None and should_run_midday(cycle_state)):
        cycle_state["midday_run"] = True
        save_cycle_state(cycle_state)
        return {"status": "ok", "message": "Midday cycle executed.", "timestamp": timestamp}

    if phase == "close" or (phase is None and should_run_close(cycle_state)):
        cycle_state["close_run"] = True
        save_cycle_state(cycle_state)
        return {"status": "ok", "message": "Close cycle executed.", "timestamp": timestamp}

    return {"status": "skipped", "message": "No cycle due at this time.", "timestamp": timestamp}
