"""Governance replay engine - stub implementation."""
from datetime import datetime, timedelta, timezone


def build_replay_state(source_path=None, days=30):
    """Return a replay state dict with events, waves, and date range."""
    events = load_replay_events(source_path=source_path)
    waves = sorted({e.get("wave") for e in events if e.get("wave")})
    tr = get_replay_time_range(source_path=source_path, days=days)
    return {"events": events, "waves": waves, "date_range": tr}


def get_replay_time_range(source_path=None, days=30):
    """Return dict with start, end, and total_days for the replay window."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    return {
        "start": start_dt.date().isoformat(),
        "end": end_dt.date().isoformat(),
        "total_days": days,
    }


def load_replay_events(source_path=None, start_date=None, end_date=None):
    """Load replay events from source_path (optional); return list of dicts."""
    if source_path is None:
        return []
    try:
        import json
        with open(source_path) as f:
            events = json.load(f)
        if start_date:
            events = [e for e in events if e.get("date", "") >= str(start_date)]
        if end_date:
            events = [e for e in events if e.get("date", "") <= str(end_date)]
        return events
    except Exception:
        return []


def generate_replay_timeline(events):
    """Convert events list into a timeline list of dicts."""
    timeline = []
    for e in events:
        timeline.append({
            "date": e.get("date") or e.get("created", ""),
            "event_type": e.get("event_type") or e.get("decision_type", ""),
            "wave": e.get("wave", ""),
            "description": e.get("description") or e.get("context_notes", ""),
        })
    return sorted(timeline, key=lambda x: x.get("date", ""))
