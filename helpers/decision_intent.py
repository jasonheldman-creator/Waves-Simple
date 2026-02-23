"""Decision intent generator - stub implementation."""


def generate_decision_intent(ctx):
    """Generate a natural-language intent description from a context dict.

    Parameters
    ----------
    ctx : dict
        May contain decision_type, scope, wave, trigger_count, etc.

    Returns
    -------
    str
        A descriptive intent string.
    """
    if not ctx:
        return "No decision context provided."

    decision_type = ctx.get("decision_type", "Decision")
    scope = ctx.get("scope", "portfolio")
    wave = ctx.get("wave")
    trigger_count = ctx.get("trigger_count", 0)

    parts = [f"Review {decision_type} for {scope}"]
    if wave:
        parts.append(f"in {wave} wave")
    if trigger_count:
        parts.append(f"based on {trigger_count} signal(s)")

    return " ".join(parts) + "."
