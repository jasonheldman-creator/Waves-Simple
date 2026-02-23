"""Escalation engine - stub implementation."""
import json
import os

_QUEUE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "governance_queue.json")


def load_governance_queue():
    """Load governance queue from disk; return list of dicts."""
    try:
        with open(_QUEUE_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_governance_queue(queue):
    """Persist governance queue to disk."""
    try:
        os.makedirs(os.path.dirname(_QUEUE_FILE), exist_ok=True)
        with open(_QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)
    except Exception:
        pass


def add_to_queue(item):
    """Append an item to the governance queue."""
    queue = load_governance_queue()
    queue.append(item)
    save_governance_queue(queue)


def remove_from_queue(item_id):
    """Remove an item from the governance queue by id."""
    queue = load_governance_queue()
    queue = [i for i in queue if i.get("id") != item_id]
    save_governance_queue(queue)


def get_escalated_items():
    """Return items in the queue that are marked as escalated."""
    return [i for i in load_governance_queue() if i.get("escalated")]
