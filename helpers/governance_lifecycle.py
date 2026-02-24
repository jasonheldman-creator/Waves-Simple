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


_STATUS_NORMALIZER = {
    "Awaiting Approval": "Awaiting Governance Review",
    "Pending": "Awaiting Governance Review",
}

_WINDOW_COLOR_MAP = {
    "standard": "#60A5FA",
    "extended": "#F59E0B",
    "overnight": "#9CA3AF",
    "escalated": "#EF4444",
    "deliberation": "#A78BFA",
}


def _normalize_pending_item(item):
    """Normalize a raw decision dict to the schema expected by the front-end renderer.

    Adds all fields required by the Executive Snapshot governance renderer,
    using .get() fallbacks so missing or misaligned fields never cause a
    KeyError.  Fields already set on the item are preserved unchanged.
    """
    status = _STATUS_NORMALIZER.get(item.get("status"), item.get("status", "Awaiting Governance Review"))

    decision_type = item.get("type") or item.get("decision_type") or "Governance Review"

    window_type = item.get("window_type", "standard") or "standard"
    window_color = item.get("window_color") or _WINDOW_COLOR_MAP.get(window_type, "#60A5FA")
    window_label = item.get("window_label") or window_type.replace("_", " ").title()

    source = (item.get("source") or item.get("source_surface")
              or item.get("trigger_source") or "System")

    # Context: prefer explicit context field, then context_notes/rationale,
    # then fall back to the action text stored in context_snapshot.
    context = (item.get("context") or item.get("context_notes") or item.get("rationale") or "")
    if not context:
        ctx_snap = item.get("context_snapshot")
        if isinstance(ctx_snap, dict):
            context = ctx_snap.get("action", "")

    time_remaining = item.get("time_remaining", "")
    time_color = item.get("time_color", "#60A5FA")
    time_pct = item.get("time_pct", 0)

    expires_at = item.get("expires_at")
    if expires_at and time_remaining == "":
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            diff = (exp - now).total_seconds()
            if diff <= 0:
                time_remaining = "Expired"
                time_color = "#EF4444"
                time_pct = 100
            else:
                hours = diff / 3600
                if hours < 1:
                    time_remaining = f"{int(diff / 60)}m"
                    time_color = "#EF4444"
                elif hours < 4:
                    time_remaining = f"{hours:.1f}h"
                    time_color = "#F59E0B"
                elif hours < 24:
                    time_remaining = f"{hours:.0f}h"
                    time_color = "#60A5FA"
                else:
                    time_remaining = f"{hours / 24:.1f}d"
                    time_color = "#9CA3AF"
                created_at = item.get("created")
                if created_at:
                    try:
                        cr = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        total = (exp - cr).total_seconds()
                        elapsed = (now - cr).total_seconds()
                        time_pct = min(100, max(0, int(elapsed / total * 100))) if total > 0 else 0
                    except Exception:
                        time_pct = 0
        except Exception:
            pass

    # If time_remaining is still unset after the expiry calculation, apply a
    # semantic fallback: overnight items show "Overnight"; all others "No expiry".
    if time_remaining == "":
        time_remaining = "Overnight" if window_type == "overnight" else "No expiry"

    return {
        **item,
        "type": decision_type,
        "status": status,
        "wave": item.get("wave") or "Unknown",
        "window_color": window_color,
        "window_label": window_label,
        "source": source,
        "context": context,
        "change": item.get("change", ""),
        "impact": item.get("impact", ""),
        "time_color": time_color,
        "time_remaining": time_remaining,
        "time_pct": time_pct,
    }


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
            result.append(_normalize_pending_item(item))
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
    modified = False
    for d in decisions:
        if d.get("id") == decision_id:
            d["status"] = action
            if actor:
                d["actor"] = actor
            if notes:
                d["context_notes"] = notes
            modified = True
            break
    if modified:
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
    modified = False
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
            modified = True
            break
    if modified:
        _save_decisions(decisions)


def create_governance_decision(decision_id, wave, instruction_type,
                                trigger_source, context_snapshot, source):
    """Create and persist a new governance decision.

    Returns ``True`` if the decision was created, ``False`` if a decision with
    the same ``decision_id`` already exists (duplicate guard).
    """
    decisions = _load_decisions()
    if any(d.get("id") == decision_id for d in decisions):
        return False
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
    return True


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


def get_executive_metrics():
    """Return executive snapshot metrics derived from governance_decisions.json.

    Computes:
    - ``pending_governance``: count of decisions with status "Awaiting Governance Review"
    - ``near_review_window``: count of decisions expiring within 24 hours
    - ``monitoring_decisions``: count of decisions with status "Monitoring"
    - ``total_active``: count of all non-terminal decisions
    """
    decisions = _load_decisions()
    now = datetime.now(timezone.utc)
    pending = 0
    near_review = 0
    monitoring = 0
    total_active = 0
    for d in decisions:
        raw_status = d.get("status", "")
        if raw_status in ("Resolved", "Expired", "Rejected"):
            continue
        total_active += 1
        normalized = _STATUS_NORMALIZER.get(raw_status, raw_status)
        if normalized == "Awaiting Governance Review":
            pending += 1
        elif normalized == "Monitoring":
            monitoring += 1
        exp = d.get("expires_at")
        if exp:
            try:
                exp_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                diff_hours = (exp_dt - now).total_seconds() / 3600
                if 0 < diff_hours <= 24:
                    near_review += 1
            except Exception:
                pass
    return {
        "pending_governance": pending,
        "near_review_window": near_review,
        "monitoring_decisions": monitoring,
        "total_active": total_active,
    }
