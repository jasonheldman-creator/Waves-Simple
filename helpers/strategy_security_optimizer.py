"""Strategy security optimizer - stub implementation."""


def evaluate_strategy_fit(holdings):
    """Return list of strategy fit evaluation dicts for each holding."""
    if not holdings:
        return []
    results = []
    for h in holdings:
        ticker = h.get("ticker", "")
        wave = h.get("wave", "")
        drift = h.get("drift", 0.0)
        if drift <= 3.0:
            fit_score, fit_label, reason = 90.0, "Optimal", "Within target allocation band."
        elif drift <= 7.0:
            fit_score, fit_label, reason = 65.0, "Acceptable", "Minor drift; monitor next cycle."
        else:
            fit_score, fit_label, reason = 40.0, "Review Recommended", "Drift exceeds tolerance threshold."
        results.append({
            "ticker": ticker,
            "wave": wave,
            "fit_score": fit_score,
            "fit_label": fit_label,
            "reason": reason,
            "alternatives": [],
        })
    return results


def evaluate_replacement_candidates(fit_results):
    """Return list of replacement candidate dicts for low-fit holdings."""
    candidates = []
    for r in fit_results:
        if r.get("fit_score", 100) < 50:
            candidates.append({
                "incumbent": r.get("ticker"),
                "replacement": None,
                "improvement_score": 0.0,
                "rationale": "No replacement candidate identified in stub mode.",
            })
    return candidates


def generate_strategy_observations(fit_results, upgrade_candidates):
    """Return list of observation strings."""
    obs = []
    low_fit = [r for r in fit_results if r.get("fit_score", 100) < 50]
    if low_fit:
        obs.append(f"{len(low_fit)} holding(s) have fit scores below 50 and warrant review.")
    if upgrade_candidates:
        obs.append(f"{len(upgrade_candidates)} potential replacement candidate(s) identified.")
    if not obs:
        obs.append("All holdings are within acceptable strategy fit parameters.")
    return obs


def get_strategy_fit_summary(fit_results):
    """Return aggregate strategy fit summary dict."""
    if not fit_results:
        return {"optimal": 0, "acceptable": 0, "review_recommended": 0,
                "data_pending": 0, "total": 0}
    optimal = sum(1 for r in fit_results if r.get("fit_label") == "Optimal")
    acceptable = sum(1 for r in fit_results if r.get("fit_label") == "Acceptable")
    review = sum(1 for r in fit_results if r.get("fit_label") == "Review Recommended")
    return {
        "optimal": optimal,
        "acceptable": acceptable,
        "review_recommended": review,
        "data_pending": 0,
        "total": len(fit_results),
    }


def check_governance_triggers(fit_results, upgrade_candidates):
    """Return list of governance trigger dicts."""
    triggers = []
    for r in fit_results:
        if r.get("fit_score", 100) < 40:
            triggers.append({
                "ticker": r.get("ticker"),
                "trigger_type": "Low Strategy Fit",
                "fit_score": r.get("fit_score"),
                "action_required": "Review holding alignment with wave mandate.",
            })
    return triggers


def execute_governance_proposals(triggers):
    """Execute governance proposals for given triggers (stub - no-op)."""
    pass
