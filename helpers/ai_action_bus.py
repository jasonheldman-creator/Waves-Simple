"""AI action bus - stub implementation."""
import json
import os
import uuid
from datetime import datetime, timezone

_REVIEWS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ai_action_reviews.json")
_ACTIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ai_actions.json")


def route_ai_action(action_type, wave=None, ticker=None, context=None, source=None):
    """Route an AI action and return a routing receipt dict."""
    action_id = str(uuid.uuid4())[:8]
    return {
        "action_id": action_id,
        "status": "routed",
        "routed_at": datetime.now(timezone.utc).isoformat(),
    }


def load_pending_ai_reviews():
    """Load pending AI action reviews from disk."""
    try:
        with open(_REVIEWS_FILE) as f:
            data = json.load(f)
        return [r for r in data if r.get("status") not in ("Approved", "Rejected", "Resolved")]
    except Exception:
        return []


def update_ai_review_status(action_id, status, actor=None, notes=None):
    """Update status of an AI action review."""
    try:
        with open(_REVIEWS_FILE) as f:
            reviews = json.load(f)
    except Exception:
        reviews = []

    for r in reviews:
        if r.get("id") == action_id or r.get("action_id") == action_id:
            r["status"] = status
            if actor:
                r["actor"] = actor
            if notes:
                r["notes"] = notes
            break

    try:
        os.makedirs(os.path.dirname(_REVIEWS_FILE), exist_ok=True)
        with open(_REVIEWS_FILE, "w") as f:
            json.dump(reviews, f, indent=2)
    except Exception:
        pass


def load_ai_actions():
    """Load AI actions from disk."""
    try:
        with open(_ACTIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []
