"""Holding intelligence evaluator."""
import os
import math
from collections import defaultdict


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


def _load_wave_weights(data_prefix="data"):
    """Return {wave_name: [(ticker, weight), ...]} from wave_weights.csv."""
    path = os.path.join(data_prefix, "wave_weights.csv")
    result = defaultdict(list)
    try:
        import csv
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wave = (row.get("wave_id") or row.get("wave_name", "")).strip()
                ticker = row.get("ticker", "").strip()
                try:
                    weight = float(row.get("weight") or 0)
                except (ValueError, TypeError):
                    weight = 0.0
                if wave and ticker and weight > 0:
                    result[wave].append((ticker, weight))
    except Exception:
        pass
    return dict(result)


def _load_prices(data_prefix="data", tickers=None):
    """Return {ticker: {date_str: close_price}} from prices.csv."""
    path = os.path.join(data_prefix, "prices.csv")
    result = defaultdict(dict)
    try:
        import csv
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("ticker", "").strip()
                if tickers is not None and t not in tickers:
                    continue
                d = row.get("date", "").strip()
                try:
                    c = float(row.get("close") or 0)
                except (ValueError, TypeError):
                    c = 0.0
                if t and d and c > 0:
                    result[t][d] = c
    except Exception:
        pass
    return dict(result)


def _compute_wave_metrics(wave_name, wave_weights_map, price_dict):
    """Compute vol_30d, momentum_30d, drawdown_90d, trend_stability for a wave."""
    tickers_weights = wave_weights_map.get(wave_name, [])
    if not tickers_weights:
        return None, None, None, None

    total_w = sum(w for _, w in tickers_weights)
    if total_w <= 0:
        return None, None, None, None

    # Collect all dates available for this wave's constituents
    all_dates = set()
    for ticker, _ in tickers_weights:
        if ticker in price_dict:
            all_dates.update(price_dict[ticker].keys())
    if not all_dates:
        return None, None, None, None

    sorted_dates = sorted(all_dates)
    if len(sorted_dates) < 2:
        return None, None, None, None

    # Compute daily portfolio returns for the last 92 price points (≈91 trading days)
    recent_dates = sorted_dates[-92:]
    daily_returns = []
    for i in range(1, len(recent_dates)):
        d_prev = recent_dates[i - 1]
        d_curr = recent_dates[i]
        day_ret = 0.0
        weight_used = 0.0
        for ticker, w in tickers_weights:
            p_prev = price_dict.get(ticker, {}).get(d_prev)
            p_curr = price_dict.get(ticker, {}).get(d_curr)
            if p_prev and p_curr and p_prev > 0:
                day_ret += (w / total_w) * ((p_curr - p_prev) / p_prev)
                weight_used += w / total_w
        if weight_used > 0.05:
            daily_returns.append(day_ret)

    if not daily_returns:
        return None, None, None, None

    last_30 = daily_returns[-30:] if len(daily_returns) >= 30 else daily_returns
    last_90 = daily_returns[-90:] if len(daily_returns) >= 90 else daily_returns

    # vol_30d: annualized standard deviation of recent 30-day returns
    if len(last_30) >= 5:
        mean_r = sum(last_30) / len(last_30)
        var_r = sum((r - mean_r) ** 2 for r in last_30) / len(last_30)
        vol_30d = round(math.sqrt(var_r * 252), 4)
    else:
        vol_30d = None

    # momentum_30d: cumulative return over last 30 trading days
    if last_30:
        cum = 1.0
        for r in last_30:
            cum *= (1 + r)
        momentum_30d = round(cum - 1.0, 4)
    else:
        momentum_30d = None

    # drawdown_90d: maximum drawdown over last 90 trading days
    if last_90:
        cum_vals = [1.0]
        for r in last_90:
            cum_vals.append(cum_vals[-1] * (1 + r))
        peak = cum_vals[0]
        drawdown_90d = 0.0
        for v in cum_vals:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (v - peak) / peak
                if dd < drawdown_90d:
                    drawdown_90d = dd
        drawdown_90d = round(drawdown_90d, 4)
    else:
        drawdown_90d = None

    # trend_stability: Sharpe-like ratio (momentum / vol), clipped to [-1, 1]
    if vol_30d is not None and vol_30d > 0 and momentum_30d is not None:
        trend_stability = round(max(-1.0, min(1.0, momentum_30d / vol_30d)), 4)
    elif momentum_30d is not None:
        trend_stability = 1.0 if momentum_30d > 0 else (-1.0 if momentum_30d < 0 else 0.0)
    else:
        trend_stability = None

    return vol_30d, momentum_30d, drawdown_90d, trend_stability


def _compute_security_metrics(ticker, price_dict):
    """Compute vol_30d, momentum_30d, drawdown_90d, trend_stability for a single ticker."""
    dates_prices = price_dict.get(ticker)
    if not dates_prices:
        return None, None, None, None

    sorted_dates = sorted(dates_prices.keys())
    if len(sorted_dates) < 2:
        return None, None, None, None

    recent_dates = sorted_dates[-92:]
    daily_returns = []
    for i in range(1, len(recent_dates)):
        p_prev = dates_prices.get(recent_dates[i - 1])
        p_curr = dates_prices.get(recent_dates[i])
        if p_prev and p_curr and p_prev > 0:
            daily_returns.append((p_curr - p_prev) / p_prev)

    if not daily_returns:
        return None, None, None, None

    last_30 = daily_returns[-30:] if len(daily_returns) >= 30 else daily_returns
    last_90 = daily_returns[-90:] if len(daily_returns) >= 90 else daily_returns

    if len(last_30) >= 5:
        mean_r = sum(last_30) / len(last_30)
        var_r = sum((r - mean_r) ** 2 for r in last_30) / len(last_30)
        vol_30d = round(math.sqrt(var_r * 252), 4)
    else:
        vol_30d = None

    if last_30:
        cum = 1.0
        for r in last_30:
            cum *= (1 + r)
        momentum_30d = round(cum - 1.0, 4)
    else:
        momentum_30d = None

    if last_90:
        cum_vals = [1.0]
        for r in last_90:
            cum_vals.append(cum_vals[-1] * (1 + r))
        peak = cum_vals[0]
        drawdown_90d = 0.0
        for v in cum_vals:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (v - peak) / peak
                if dd < drawdown_90d:
                    drawdown_90d = dd
        drawdown_90d = round(drawdown_90d, 4)
    else:
        drawdown_90d = None

    if vol_30d is not None and vol_30d > 0 and momentum_30d is not None:
        trend_stability = round(max(-1.0, min(1.0, momentum_30d / vol_30d)), 4)
    elif momentum_30d is not None:
        trend_stability = 1.0 if momentum_30d > 0 else (-1.0 if momentum_30d < 0 else 0.0)
    else:
        trend_stability = None

    return vol_30d, momentum_30d, drawdown_90d, trend_stability


def evaluate_holdings(data_prefix="data"):
    """Return list of holding evaluation dicts with computed metrics.

    Holdings are sourced canonically from wave_weights.csv (per-security, per-wave).
    Metrics (vol, momentum, drawdown, trend) are computed from prices.csv.
    Missing data logs a warning rather than silently defaulting.
    """
    # Canonical source: wave_weights.csv (not live_snapshot.csv)
    wave_weights_map = _load_wave_weights(data_prefix)
    if not wave_weights_map:
        import warnings
        warnings.warn("[HI] wave_weights.csv empty or missing — holdings unavailable")
        return []

    all_tickers = {t for tickers in wave_weights_map.values() for t, _ in tickers}
    price_dict = _load_prices(data_prefix, all_tickers)

    results = []
    for wave_name, tickers_weights in sorted(wave_weights_map.items()):
        for ticker, weight in tickers_weights:
            vol_30d, momentum_30d, drawdown_90d, trend_stability = _compute_security_metrics(
                ticker, price_dict
            )
            if vol_30d is None and momentum_30d is None:
                import warnings
                warnings.warn(f"[HI] No price data for {ticker} in {wave_name}")
                status = "Data Pending"
            elif (drawdown_90d is not None and drawdown_90d < -0.15) or (
                momentum_30d is not None and momentum_30d < -0.12
            ):
                status = "Review Candidate"
            elif (drawdown_90d is not None and drawdown_90d < -0.08) or (
                momentum_30d is not None and momentum_30d < -0.05
            ):
                status = "Monitoring"
            else:
                status = "Stable"

            if momentum_30d is not None and momentum_30d > 0.05:
                observation = f"Positive momentum ({momentum_30d * 100:.1f}% / 30d)."
            elif momentum_30d is not None and momentum_30d < -0.05:
                observation = f"Negative momentum ({momentum_30d * 100:.1f}% / 30d)."
            elif drawdown_90d is not None and drawdown_90d < -0.10:
                observation = f"Drawdown {drawdown_90d * 100:.1f}% over 90d."
            else:
                observation = "Within acceptable performance range."

            results.append({
                "ticker": ticker,
                "wave": wave_name,
                "weight": weight,
                "target_weight": weight,
                "drift": 0.0,
                "drift_direction": "Flat",
                "review_cycles": 0,
                "status": status,
                "observation": observation,
                "vol_30d": vol_30d,
                "momentum_30d": momentum_30d,
                "drawdown_90d": drawdown_90d,
                "trend_stability": trend_stability,
            })

    if not results:
        import warnings
        warnings.warn("[HI] No holdings evaluated — check wave_weights.csv and prices.csv")
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
