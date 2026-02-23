"""Market briefing computations - stub implementation (no live API calls)."""
from datetime import datetime, timezone
from helpers.market_data import (
    compute_returns as _compute_returns_series,
    compute_above_ma,
    compute_realized_vol,
    SECTOR_TICKERS,
)


def _safe_prices(prices, ticker):
    """Return price list for ticker or empty list."""
    if not prices:
        return []
    val = prices.get(ticker)
    return val if isinstance(val, list) and len(val) >= 2 else []


def compute_returns(prices, ticker):
    """Return period returns dict for a ticker using the prices dict."""
    series = _safe_prices(prices, ticker)
    return _compute_returns_series(series)


def compute_direction_label(returns_dict):
    """Return directional label based on returns dict."""
    if not returns_dict:
        return "Neutral"
    r1d = returns_dict.get("1d")
    r5d = returns_dict.get("5d")
    if r1d is None and r5d is None:
        return "Neutral"
    positives = sum(1 for v in [r1d, r5d] if v is not None and v > 0)
    negatives = sum(1 for v in [r1d, r5d] if v is not None and v < 0)
    if positives > negatives:
        return "Bullish"
    if negatives > positives:
        return "Bearish"
    return "Mixed"


def compute_strength_score(returns_dict):
    """Return a 0-100 strength score based on recent returns."""
    if not returns_dict:
        return 50.0
    vals = [v for v in returns_dict.values() if v is not None]
    if not vals:
        return 50.0
    avg = sum(vals) / len(vals)
    score = 50.0 + avg * 500.0
    return round(max(0.0, min(100.0, score)), 2)


def compute_regime_structure(prices):
    """Return regime structure dict."""
    if not prices:
        return {"regime": "Unknown", "description": "Insufficient data.", "confidence": "Low"}
    spy = _safe_prices(prices, "SPY")
    if not spy:
        return {"regime": "Unknown", "description": "SPY data unavailable.", "confidence": "Low"}
    above_ma = compute_above_ma(spy, 50)
    vol = compute_realized_vol(spy)
    if above_ma and vol < 0.20:
        return {"regime": "Bull/Low-Vol", "description": "Trending above 50-day MA with low volatility.", "confidence": "Medium"}
    if not above_ma and vol > 0.25:
        return {"regime": "Bear/High-Vol", "description": "Below 50-day MA with elevated volatility.", "confidence": "Medium"}
    return {"regime": "Transitional", "description": "Mixed signals across trend and volatility.", "confidence": "Low"}


def compute_directional_agreement(prices):
    """Return agreement score across tickers."""
    if not prices:
        return {"agreement_score": 0.0, "direction": "Neutral", "tickers_up": 0, "tickers_down": 0}
    up = down = 0
    for series in prices.values():
        if isinstance(series, list) and len(series) >= 2:
            if series[-1] > series[-2]:
                up += 1
            elif series[-1] < series[-2]:
                down += 1
    total = up + down or 1
    agreement_score = round(max(up, down) / total, 3)
    direction = "Bullish" if up > down else ("Bearish" if down > up else "Neutral")
    return {"agreement_score": agreement_score, "direction": direction,
            "tickers_up": up, "tickers_down": down}


def compute_decision_implications(regime_dict):
    """Return decision implication dict based on regime."""
    regime = (regime_dict or {}).get("regime", "Unknown")
    if "Bull" in regime:
        return {"implication": "Favour risk-on positioning.", "action_bias": "Accumulate", "risk_level": "Low"}
    if "Bear" in regime:
        return {"implication": "Reduce exposure; favour defensive assets.", "action_bias": "Reduce", "risk_level": "High"}
    return {"implication": "Hold current allocations pending clarity.", "action_bias": "Hold", "risk_level": "Medium"}


def compute_structural_signals(prices):
    """Return structural signals dict."""
    signals = []
    if prices:
        for ticker, series in prices.items():
            if isinstance(series, list) and len(series) >= 50:
                above = compute_above_ma(series)
                signals.append({"ticker": ticker, "above_ma50": above,
                                 "signal": "Trend +" if above else "Trend -"})
    summary = f"{sum(1 for s in signals if s.get('above_ma50'))} of {len(signals)} tickers above 50-day MA."
    return {"signals": signals, "summary": summary}


def compute_horizon_explanation(period):
    """Return description for a return horizon."""
    explanations = {
        "1d": "1-day return (most recent session)",
        "5d": "5-day return (~1 trading week)",
        "30d": "30-day return (~1 calendar month)",
        "90d": "90-day return (~1 quarter)",
        "365d": "365-day return (~1 year)",
    }
    return explanations.get(str(period), f"{period} return window")


def compute_volatility_stress_assessment(prices):
    """Return volatility stress assessment dict."""
    if not prices:
        return {"level": "Unknown", "score": 0.0, "description": "No price data available."}
    vols = []
    for series in prices.values():
        if isinstance(series, list) and len(series) >= 22:
            vols.append(compute_realized_vol(series))
    if not vols:
        return {"level": "Unknown", "score": 0.0, "description": "Insufficient data."}
    avg_vol = sum(vols) / len(vols)
    score = round(min(100.0, avg_vol * 300.0), 2)
    level = "High" if avg_vol > 0.30 else ("Medium" if avg_vol > 0.15 else "Low")
    return {"level": level, "score": score,
            "description": f"Average realised vol: {avg_vol:.1%}"}


def compute_breadth_assessment(prices):
    """Return market breadth assessment dict."""
    if not prices:
        return {"breadth_pct": 0.0, "level": "Unknown", "description": "No data."}
    agreement = compute_directional_agreement(prices)
    breadth = agreement["tickers_up"] / max(1, agreement["tickers_up"] + agreement["tickers_down"])
    level = "Strong" if breadth > 0.7 else ("Weak" if breadth < 0.3 else "Moderate")
    return {"breadth_pct": round(breadth, 3), "level": level,
            "description": f"{agreement['tickers_up']} of {agreement['tickers_up'] + agreement['tickers_down']} tickers advancing."}


def compute_rates_credit_assessment(prices):
    """Return rates and credit assessment dict."""
    if not prices:
        return {"rates_trend": "Unknown", "credit_stress": "Unknown", "description": "No data."}
    tlt = _safe_prices(prices, "TLT")
    hyg = _safe_prices(prices, "HYG")
    rates_trend = "Rising" if (tlt and tlt[-1] < tlt[-2]) else ("Falling" if (tlt and tlt[-1] > tlt[-2]) else "Flat")
    credit_stress = "Elevated" if (hyg and hyg[-1] < hyg[-2]) else "Normal"
    return {"rates_trend": rates_trend, "credit_stress": credit_stress,
            "description": f"Rates {rates_trend.lower()}, credit stress {credit_stress.lower()}."}


def compute_sector_assessment(prices):
    """Return sector rotation assessment dict."""
    if not prices:
        return {"top_sector": "N/A", "lagging_sector": "N/A", "rotation_signal": "No data"}
    sector_rets = {}
    for t in SECTOR_TICKERS:
        series = _safe_prices(prices, t)
        if series:
            r = _compute_returns_series(series)
            if r.get("5d") is not None:
                sector_rets[t] = r["5d"]
    if not sector_rets:
        return {"top_sector": "N/A", "lagging_sector": "N/A", "rotation_signal": "Insufficient data"}
    top = max(sector_rets, key=sector_rets.get)
    lag = min(sector_rets, key=sector_rets.get)
    return {"top_sector": top, "lagging_sector": lag,
            "rotation_signal": f"Rotation towards {top}, away from {lag}."}


def compute_regime_assessment(prices):
    """Return regime assessment dict."""
    if not prices:
        return {"regime": "Unknown", "stability": "Unknown", "description": "No data.", "data_available": False}
    structure = compute_regime_structure(prices)
    return {**structure, "stability": "Moderate", "data_available": True}


def compute_executive_chips(prices):
    """Return list of executive chip dicts for dashboard display."""
    chips = []
    if not prices:
        return chips
    spy = _safe_prices(prices, "SPY")
    if spy:
        rets = _compute_returns_series(spy)
        r1d = rets.get("1d")
        chips.append({
            "label": "SPY 1d",
            "value": f"{r1d:+.2%}" if r1d is not None else "N/A",
            "color": "green" if (r1d or 0) >= 0 else "red",
            "trend": "up" if (r1d or 0) >= 0 else "down",
        })
        vol = compute_realized_vol(spy)
        chips.append({
            "label": "Realised Vol",
            "value": f"{vol:.1%}",
            "color": "red" if vol > 0.25 else ("orange" if vol > 0.15 else "green"),
            "trend": "neutral",
        })
    return chips


def compute_orientation_sentence(prices):
    """Return a single sentence describing the current market orientation."""
    if not prices:
        return "Market data is not available."
    agreement = compute_directional_agreement(prices)
    regime = compute_regime_structure(prices)
    direction = agreement["direction"]
    reg = regime["regime"]
    return f"Markets are broadly {direction.lower()} within a {reg} regime."


def compute_what_changed(prices):
    """Return list of notable change strings."""
    if not prices:
        return ["No price data available to detect changes."]
    changes = []
    for ticker, series in prices.items():
        if isinstance(series, list) and len(series) >= 2:
            r = _compute_returns_series(series)
            r1d = r.get("1d")
            if r1d is not None and abs(r1d) > 0.02:
                direction = "gained" if r1d > 0 else "fell"
                changes.append(f"{ticker} {direction} {abs(r1d):.1%} yesterday.")
    return changes if changes else ["No significant moves detected."]
