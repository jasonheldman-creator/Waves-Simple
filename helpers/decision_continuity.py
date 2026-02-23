"""Decision continuity tracker - stub implementation."""
from datetime import datetime, timezone


def get_continuity_summary():
    """Return a high-level continuity summary dict."""
    return {
        "active_decisions": 0,
        "pending_reviews": 0,
        "continuity_score": 1.0,
        "assessment": "No active governance decisions. Continuity is intact.",
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def get_continuity_assessments():
    """Return per-wave continuity assessment list."""
    waves = ["Growth", "Income", "Defensive", "SP500"]
    return [
        {
            "wave": w,
            "status": "Stable",
            "score": 1.0,
            "last_activity": datetime.now(timezone.utc).date().isoformat(),
            "notes": "No pending decisions.",
        }
        for w in waves
    ]
