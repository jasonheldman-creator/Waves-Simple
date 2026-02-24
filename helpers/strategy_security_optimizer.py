"""Strategy security optimizer — dynamic fit scoring from market metrics."""


def evaluate_strategy_fit(holdings):
    """Return list of strategy fit evaluation dicts for each holding.

    Fit score is computed dynamically from actual market metrics
    (momentum_30d, vol_30d, drawdown_90d, trend_stability) so scores
    vary across securities and reflect current conditions.
    Score range: 0–100.
    """
    if not holdings:
        return []
    results = []
    for h in holdings:
        ticker = h.get("ticker", "")
        wave = h.get("wave", "")
        momentum = h.get("momentum_30d")
        vol = h.get("vol_30d")
        drawdown = h.get("drawdown_90d")
        trend = h.get("trend_stability")

        if momentum is None and vol is None:
            # Fall back to drift-based scoring when market metrics are unavailable
            drift = h.get("drift", 0.0)
            if drift <= 3.0:
                fit_score, classification, reason = 90.0, "Optimal Fit", "Within target allocation band."
            elif drift <= 7.0:
                fit_score, classification, reason = 65.0, "Acceptable Fit", "Minor drift; monitor next cycle."
            elif drift <= 12.0:
                fit_score, classification, reason = 45.0, "Weak Fit", "Moderate drift; review recommended."
            else:
                fit_score, classification, reason = 25.0, "Review Candidate", "Drift exceeds tolerance threshold."
        else:
            # Base score: 50. Adjust symmetrically from real market metrics.
            score = 50.0
            # Momentum contribution: ±25 pts (momentum of ±12.5% maps to ±25)
            if momentum is not None:
                score += max(-25.0, min(25.0, momentum * 200.0))
            # Volatility penalty: 0–15 pts (annualised vol of 100% → –15)
            if vol is not None:
                score -= min(15.0, vol * 15.0)
            # Drawdown penalty: 0–20 pts (90d drawdown of –20% → –20)
            if drawdown is not None:
                score -= max(0.0, min(20.0, abs(drawdown) * 100.0))
            # Trend stability bonus: ±10 pts
            if trend is not None:
                score += trend * 10.0

            fit_score = round(max(0.0, min(100.0, score)), 1)

            if fit_score >= 70:
                classification = "Optimal Fit"
                reason = "Strong momentum and controlled risk profile."
            elif fit_score >= 50:
                classification = "Acceptable Fit"
                reason = "Adequate performance within tolerance."
            elif fit_score >= 30:
                classification = "Weak Fit"
                reason = "Elevated drawdown or negative momentum detected."
            else:
                classification = "Review Candidate"
                reason = "Poor momentum and/or excessive drawdown."

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
        score = r.get("fit_score")
        if score is not None and score < 50:
            candidates.append({
                "current_security": r.get("ticker"),
                "candidate_security": None,
                "wave": r.get("wave", ""),
                "current_score": score,
                "candidate_score": 0.0,
                "relative_fit_improvement": 0.0,
            })
    return candidates


def generate_strategy_observations(fit_results, upgrade_candidates):
    """Return list of observation dicts with 'security' and 'observation' keys."""
    obs = []
    low_fit = [r for r in fit_results if (r.get("fit_score") or 100) < 50]
    for r in low_fit:
        score = r.get("fit_score")
        score_str = f"{score:.0f}" if score is not None else "—"
        obs.append({
            "security": r.get("ticker", ""),
            "observation": f"Fit score {score_str} — {r.get('reason', 'Review recommended.')}",
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
