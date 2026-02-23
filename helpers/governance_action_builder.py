"""Governance action builder - stub implementation."""


def build_proposed_action(decision_type, scope, wave=None):
    """Return a structured proposed action dict based on inputs."""
    _risk_map = {
        "Rebalance": "Medium",
        "Exit": "High",
        "Hold": "Low",
        "Review": "Low",
        "Escalate": "High",
    }
    _approval_required = {"Exit", "Rebalance", "Escalate"}

    risk_level = _risk_map.get(str(decision_type), "Medium")
    requires_ic = str(decision_type) in _approval_required

    if wave:
        proposed_action = f"{decision_type} {scope} within {wave} wave"
        rationale = (f"Governance review triggered for {decision_type} "
                     f"action on {scope} in {wave} wave.")
    else:
        proposed_action = f"{decision_type} {scope}"
        rationale = f"Governance review triggered for {decision_type} action on {scope}."

    return {
        "decision_type": decision_type,
        "scope": scope,
        "wave": wave,
        "proposed_action": proposed_action,
        "rationale": rationale,
        "risk_level": risk_level,
        "requires_ic_approval": requires_ic,
    }
