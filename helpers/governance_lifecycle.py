"""Governance lifecycle management - stub implementation."""
import json
import os
from datetime import datetime, timezone

_DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "governance_decisions.json")
_QUEUE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "governance_queue.json")


def _load_decisions():
    try:
        with open(_DATA_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_decisions(decisions):
    try:
        os.makedirs(os.path.dirname(_DATA_FILE), exist_ok=True)
        with open(_DATA_FILE, "w") as f:
            json.dump(decisions, f, indent=2)
    except Exception:
        pass


def detect_duplicates():
    """Return list of warning strings for duplicate decisions."""
    decisions = _load_decisions()
    seen, warnings = set(), []
    for d in decisions:
        key = (d.get("wave"), d.get("decision_type"))
        if key in seen:
            warnings.append(f"Duplicate: wave={key[0]}, type={key[1]}")
        seen.add(key)
    return warnings


def deduplicate_stores():
    """Remove duplicate decisions from the store."""
    decisions = _load_decisions()
    seen, deduped = set(), []
    for d in decisions:
        key = (d.get("id"),)
        if key not in seen:
            deduped.append(d)
            seen.add(key)
    _save_decisions(deduped)


def process_expired_with_lifecycle():
    """Mark expired decisions as Expired."""
    decisions = _load_decisions()
    now = datetime.now(timezone.utc).isoformat()
    changed = False
    for d in decisions:
        exp = d.get("expires_at")
        if exp and d.get("status") not in ("Resolved", "Expired", "Rejected"):
            try:
                if exp < now:
                    d["status"] = "Expired"
                    changed = True
            except Exception:
                pass
    if changed:
        _save_decisions(decisions)


def get_all_pending():
    """Return list of pending governance decision dicts."""
    decisions = _load_decisions()
    _default_keys = {
        "id": None, "status": "Pending", "wave": None, "decision_type": None,
        "created": None, "expires_at": None, "actor": None, "context_notes": "",
        "is_instruction": False, "trigger_count": 0, "window_type": "standard",
        "source_surface": None, "rationale": "",
    }
    result = []
    for d in decisions:
        if d.get("status") not in ("Resolved", "Expired", "Rejected"):
            item = {**_default_keys, **d}
            result.append(item)
    return result


def get_governance_counts():
    """Return dict of governance decision counts."""
    decisions = _load_decisions()
    now = datetime.now(timezone.utc).isoformat()
    counts = {"total": 0, "active": 0, "expiring_soon": 0, "escalated": 0,
              "overnight": 0, "under_deliberation": 0}
    for d in decisions:
        counts["total"] += 1
        status = d.get("status", "")
        if status not in ("Resolved", "Expired", "Rejected"):
            counts["active"] += 1
        exp = d.get("expires_at", "")
        if exp:
            try:
                diff_hours = (datetime.fromisoformat(exp.replace("Z", "+00:00"))
                              - datetime.now(timezone.utc)).total_seconds() / 3600
                if 0 < diff_hours < 24:
                    counts["expiring_soon"] += 1
            except Exception:
                pass
        if d.get("escalated"):
            counts["escalated"] += 1
        if d.get("window_type") == "overnight":
            counts["overnight"] += 1
        if status == "Under Deliberation":
            counts["under_deliberation"] += 1
    return counts


def get_market_session_label():
    """Return current market session label."""
    hour = datetime.now(timezone.utc).hour
    minute = datetime.now(timezone.utc).minute
    total_minutes = hour * 60 + minute
    if 13 * 60 + 30 <= total_minutes <= 20 * 60:
        return "Market Hours"
    return "After Hours"


def get_auto_executed_today():
    """Return list of auto-executed decisions today."""
    decisions = _load_decisions()
    today = datetime.now(timezone.utc).date().isoformat()
    return [d for d in decisions if d.get("auto_executed") and
            (d.get("created", "") or "").startswith(today)]


def get_morning_briefing():
    """Return morning briefing summary dict."""
    pending = get_all_pending()
    return {
        "overnight_count": 0,
        "carryover_count": len(pending),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def propagate_action(decision_id, action, actor=None, notes=None,
                     source_surface=None, wave=None, decision_type=None,
                     is_instruction=False):
    """Propagate an action to a governance decision."""
    decisions = _load_decisions()
    for d in decisions:
        if d.get("id") == decision_id:
            d["status"] = action
            if actor:
                d["actor"] = actor
            if notes:
                d["context_notes"] = notes
            break
    _save_decisions(decisions)


def request_extension(decision_id, extension_hours=24, actor=None):
    """Request a time extension for a governance decision."""
    decisions = _load_decisions()
    for d in decisions:
        if d.get("id") == decision_id:
            try:
                exp = datetime.fromisoformat(
                    d["expires_at"].replace("Z", "+00:00"))
                from datetime import timedelta
                d["expires_at"] = (exp + timedelta(hours=extension_hours)).isoformat()
            except Exception:
                pass
            break
    _save_decisions(decisions)


def get_decision_detail(decision_id):
    """Return detail dict for a decision, or None."""
    for d in _load_decisions():
        if d.get("id") == decision_id:
            return d
    return None


def log_deliberation_artifact(decision_id, artifact_name, content,
                               actor=None, source_surface=None):
    """Log a deliberation artifact to a decision."""
    decisions = _load_decisions()
    for d in decisions:
        if d.get("id") == decision_id:
            artifacts = d.setdefault("artifacts", [])
            artifacts.append({
                "name": artifact_name,
                "content": content,
                "actor": actor,
                "source_surface": source_surface,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            break
    _save_decisions(decisions)


def create_governance_decision(decision_id, wave, instruction_type,
                                trigger_source, context_snapshot, source):
    """Create and persist a new governance decision."""
    decisions = _load_decisions()
    decisions.append({
        "id": decision_id,
        "wave": wave,
        "decision_type": instruction_type,
        "trigger_source": trigger_source,
        "context_snapshot": context_snapshot,
        "source": source,
        "status": "Awaiting Approval",
        "created": datetime.now(timezone.utc).isoformat(),
    })
    _save_decisions(decisions)


def get_governance_queue():
    """Load and return governance queue."""
    try:
        with open(_QUEUE_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_governance_decision(decision):
    """Persist a single governance decision."""
    decisions = _load_decisions()
    for i, d in enumerate(decisions):
        if d.get("id") == decision.get("id"):
            decisions[i] = decision
            _save_decisions(decisions)
            return
    decisions.append(decision)
    _save_decisions(decisions)
