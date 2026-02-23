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
            fit_score, classification, reason = 90.0, "Optimal Fit", "Within target allocation band."
        elif drift <= 7.0:
            fit_score, classification, reason = 65.0, "Acceptable Fit", "Minor drift; monitor next cycle."
        elif drift <= 12.0:
            fit_score, classification, reason = 45.0, "Weak Fit", "Moderate drift; review recommended."
        else:
            fit_score, classification, reason = 25.0, "Review Candidate", "Drift exceeds tolerance threshold."
        results.append({
            "ticker": ticker,
            "wave": wave,
            "fit_score": fit_score,
            "classification": classification,
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
                "current_security": r.get("ticker"),
                "candidate_security": None,
                "wave": r.get("wave", ""),
                "current_score": r.get("fit_score", 0.0),
                "candidate_score": 0.0,
                "relative_fit_improvement": 0.0,
            })
    return candidates


def generate_strategy_observations(fit_results, upgrade_candidates):
    """Return list of observation dicts with 'security' and 'observation' keys."""
    obs = []
    low_fit = [r for r in fit_results if r.get("fit_score", 100) < 50]
    for r in low_fit:
        obs.append({
            "security": r.get("ticker", ""),
            "observation": f"Fit score {r.get('fit_score', 0):.0f} — {r.get('reason', 'Review recommended.')}",
        })
    if upgrade_candidates:
        for uc in upgrade_candidates:
            obs.append({
                "security": uc.get("current_security", ""),
                "observation": "Potential replacement candidate identified.",
            })
    if not obs:
        for r in fit_results:
            obs.append({
                "security": r.get("ticker", ""),
                "observation": "Within acceptable strategy fit parameters.",
            })
    return obs


def get_strategy_fit_summary(fit_results):
    """Return aggregate strategy fit summary dict."""
    if not fit_results:
        return {
            "optimal": 0,
            "acceptable": 0,
            "weak_fit": 0,
            "review_candidate": 0,
            "data_pending": 0,
            "total": 0,
        }
    optimal = sum(1 for r in fit_results if r.get("classification") == "Optimal Fit")
    acceptable = sum(1 for r in fit_results if r.get("classification") == "Acceptable Fit")
    weak_fit = sum(1 for r in fit_results if r.get("classification") == "Weak Fit")
    review_candidate = sum(1 for r in fit_results if r.get("classification") == "Review Candidate")
    data_pending = sum(1 for r in fit_results if r.get("classification") == "Data Pending")
    return {
        "optimal": optimal,
        "acceptable": acceptable,
        "weak_fit": weak_fit,
        "review_candidate": review_candidate,
        "data_pending": data_pending,
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
