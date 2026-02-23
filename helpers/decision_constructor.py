"""Decision constructor - stub implementation."""
import uuid
from datetime import datetime, timezone


def build_governance_decision(decision_type=None, wave=None, scope=None,
                               context=None, actor=None, source=None, **kwargs):
    """Build and return a governance decision dict.

    Parameters
    ----------
    decision_type : str, optional
    wave : str, optional
    scope : str, optional
    context : dict, optional
    actor : str, optional
    source : str, optional
    **kwargs : additional fields merged into the decision

    Returns
    -------
    dict
    """
    context_notes = ""
    if context and isinstance(context, dict):
        context_notes = "; ".join(f"{k}={v}" for k, v in context.items())
    elif context:
        context_notes = str(context)

    rationale = kwargs.pop("rationale", None) or (
        f"Governance review for {decision_type or 'decision'} "
        f"on {scope or 'portfolio'}" +
        (f" in {wave} wave." if wave else ".")
    )

    decision = {
        "id": str(uuid.uuid4())[:8],
        "decision_type": decision_type,
        "wave": wave,
        "scope": scope,
        "status": "Awaiting Approval",
        "created": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "context_notes": context_notes,
        "rationale": rationale,
    }
    decision.update(kwargs)
    return decision
