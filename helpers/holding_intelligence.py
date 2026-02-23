"""Holding intelligence evaluator - stub implementation."""
import os


def _load_holdings(data_prefix="data"):
    """Load holdings from live_snapshot.csv; return list of dicts."""
    path = os.path.join(data_prefix, "live_snapshot.csv")
    try:
        import csv
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def evaluate_holdings(data_prefix="data"):
    """Return list of holding evaluation dicts."""
    raw = _load_holdings(data_prefix)
    results = []
    for row in raw:
        ticker = row.get("ticker") or row.get("Ticker", "")
        wave = row.get("wave") or row.get("Wave", "")
        try:
            weight = float(row.get("weight") or row.get("Weight") or 0)
        except (ValueError, TypeError):
            weight = 0.0
        try:
            target = float(row.get("target_weight") or row.get("Target_Weight") or weight)
        except (ValueError, TypeError):
            target = weight
        drift = round(abs(weight - target), 4)
        results.append({
            "ticker": ticker,
            "wave": wave,
            "weight": weight,
            "target_weight": target,
            "drift": drift,
            "drift_direction": "Over" if weight > target else ("Under" if weight < target else "Flat"),
            "review_cycles": 0,
            "status": "Stable",
            "observation": "No significant drift detected.",
            "vol_30d": None,
            "momentum_30d": None,
            "drawdown_90d": None,
            "trend_stability": None,
        })
    return results


def evaluate_secondary_candidates(holdings=None, wave_name=None):
    """Return list of secondary candidate evaluation dicts."""
    if not holdings:
        return []
    candidates = [h for h in holdings if h.get("drift", 0) > 2.0]
    if wave_name:
        candidates = [h for h in candidates if h.get("wave") == wave_name]
    return candidates


def generate_holding_observations(holdings, secondary=None):
    """Return list of observation dicts for current holdings."""
    if not holdings:
        return [{"security": "", "observation": "No holding data available."}]
    obs = []
    for h in holdings:
        ticker = h.get("ticker", "")
        if h.get("drift", 0) > 5.0:
            obs.append({
                "security": ticker,
                "observation": f"Drift of {h['drift']:.1f}% detected ({h['drift_direction']}).",
            })
    if not obs:
        for h in holdings:
            ticker = h.get("ticker", "")
            obs.append({
                "security": ticker,
                "observation": "Within acceptable drift thresholds.",
            })
    return obs


def get_holdings_summary(holdings):
    """Return aggregate holdings summary dict."""
    if not holdings:
        return {
            "total": 0,
            "with_data": 0,
            "coverage_pct": 0.0,
            "stable": 0,
            "monitoring": 0,
            "review_candidate": 0,
            "data_pending": 0,
        }
    total = len(holdings)
    with_data = sum(1 for h in holdings if h.get("ticker") or h.get("wave"))
    stable = sum(1 for h in holdings if h.get("status") == "Stable")
    monitoring = sum(1 for h in holdings if h.get("status") == "Monitoring")
    review_candidate = sum(1 for h in holdings if h.get("status") == "Review Candidate")
    data_pending = sum(1 for h in holdings if h.get("status") == "Data Pending")
    coverage_pct = round((with_data / total) * 100.0, 1) if total > 0 else 0.0
    return {
        "total": total,
        "with_data": with_data,
        "coverage_pct": coverage_pct,
        "stable": stable,
        "monitoring": monitoring,
        "review_candidate": review_candidate,
        "data_pending": data_pending,
    }


def get_governance_eligible_holdings(holdings):
    """Return holdings where drift > 5.0 and review_cycles > 2."""
    return [h for h in holdings if h.get("drift", 0) > 5.0 and h.get("review_cycles", 0) > 2]
