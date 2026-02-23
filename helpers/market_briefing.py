"""Market briefing computations - stub implementation (no live API calls)."""
from datetime import datetime, timezone
from helpers.market_data import (
    compute_returns as _compute_returns_series,
    compute_above_ma,
    compute_realized_vol,
    compute_vol_of_vol,
    compute_drawdown,
    compute_pct_up_days,
    SECTOR_TICKERS,
)


def _safe_prices(prices, ticker):
    """Return price list for ticker or empty list."""
    if not prices:
        return []
    val = prices.get(ticker)
    return val if isinstance(val, list) and len(val) >= 2 else []


def compute_returns(prices, ticker):
    """Return period returns dict for a ticker using the prices dict.

    Also accepts (price_list, _days) for direct price-series usage;
    when a list is passed as the first argument, the second argument is ignored
    and all standard period returns are computed from the series.
    """
    if isinstance(prices, list):
        # Called with a price series directly; second arg (days/ticker) is ignored
        return _compute_returns_series(prices)
    series = _safe_prices(prices, ticker)
    return _compute_returns_series(series)


def compute_direction_label(returns_dict):
    """Return (direction_label, pct_str) tuple based on returns dict.

    direction_label is 'Up', 'Down', or 'Flat'.
    pct_str is a formatted percentage string of the most representative return.
    """
    if not returns_dict or not isinstance(returns_dict, dict):
        return ("Flat", "N/A")
    # Prefer 30d as the most balanced medium-term view; fall back to shorter/longer periods
    pct_val = None
    for key in ("30d", "90d", "5d", "1d", "365d"):
        v = returns_dict.get(key)
        if v is not None:
            pct_val = v
            break
    if pct_val is None:
        return ("Flat", "N/A")
    if pct_val > 0.01:
        direction = "Up"
    elif pct_val < -0.01:
        direction = "Down"
    else:
        direction = "Flat"
    pct_str = f"{pct_val:+.1%}"
    return (direction, pct_str)


def compute_strength_score(prices_or_dict, days=None):
    """Return a 0-100 strength score.

    Accepts either:
    - compute_strength_score(returns_dict): original signature
    - compute_strength_score(prices_list, days): prices list + look-back days

    When called with a prices list, the `days` parameter is used only to detect
    the calling convention; all standard period returns are computed from the list.
    """
    if days is not None and isinstance(prices_or_dict, list):
        returns_dict = _compute_returns_series(prices_or_dict)
    else:
        returns_dict = prices_or_dict
    if not returns_dict or not isinstance(returns_dict, dict):
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


def compute_horizon_explanation(period_or_prices, days=None, label=None):
    """Return description for a return horizon.

    Accepts either:
    - compute_horizon_explanation(period): original signature where period is a
      string like '30d' or an int.
    - compute_horizon_explanation(prices, days, label): richer context signature
      where the first argument (prices dict) is not used in the description; only
      `days` and `label` are used.
    """
    explanations = {
        "1d": "1-day return (most recent session)",
        "5d": "5-day return (~1 trading week)",
        "30d": "30-day return (~1 calendar month)",
        "90d": "90-day return (~1 quarter)",
        "365d": "365-day return (~1 year)",
        30: "30-day return (~1 calendar month)",
        90: "90-day return (~1 quarter)",
        365: "365-day return (~1 year)",
    }
    if days is not None:
        base = explanations.get(days, f"{days}-day return window")
        if label:
            return f"{label}: {base}"
        return base
    return explanations.get(str(period_or_prices), f"{period_or_prices} return window")


def compute_volatility_stress_assessment(prices):
    """Return volatility stress assessment dict."""
    if not prices:
        return {"level": "Unknown", "score": 0.0, "description": "No price data available.",
                "stress_level": "Low", "trend": "Stable", "regime": "Neutral", "opportunity_context": "Neutral",
                "avg_vol": 0.0, "avg_vov": 0.0, "worst_dd": 0.0, "cross_asset_agreement": False}
    vols = []
    vov_readings = []
    dd_readings = []
    for t in ["SPY", "QQQ", "IWM", "EFA", "HYG"]:
        series = _safe_prices(prices, t)
        if series:
            vols.append(compute_realized_vol(series, 21))
            vov_readings.append(compute_vol_of_vol(series, 63))
            dd_readings.append(compute_drawdown(series))
    if not vols:
        return {"level": "Unknown", "score": 0.0, "description": "Insufficient data.",
                "stress_level": "Low", "trend": "Stable", "regime": "Neutral", "opportunity_context": "Neutral",
                "avg_vol": 0.0, "avg_vov": 0.0, "worst_dd": 0.0, "cross_asset_agreement": False}
    avg_vol = (sum(vols) / len(vols)) if vols else 0.15
    avg_vov = (sum(vov_readings) / len(vov_readings)) if vov_readings else 0.03
    worst_dd = min(dd_readings) if dd_readings else -0.03

    stress_level = "Low"
    if avg_vol > 0.25 or worst_dd < -0.10 or avg_vov > 0.06:
        stress_level = "Elevated"
    elif avg_vol > 0.16 or worst_dd < -0.05 or avg_vov > 0.04:
        stress_level = "Moderate"

    regime = "Neutral"
    if avg_vol < 0.10:
        regime = "Compression"
    elif avg_vol > 0.28:
        regime = "Expansion"
    elif avg_vol > 0.20 and avg_vov > 0.04:
        regime = "Exhaustion"

    opp_context = "Neutral"
    if regime == "Compression":
        opp_context = "Tailwind"
    elif regime in ["Expansion", "Exhaustion"]:
        opp_context = "Headwind"

    spy = _safe_prices(prices, "SPY")
    trend = "Stable"
    if spy:
        spy_vol_short = compute_realized_vol(spy, 10)
        spy_vol_long = compute_realized_vol(spy, 42)
        if spy_vol_short and spy_vol_long:
            if spy_vol_short > spy_vol_long * 1.2:
                trend = "Rising"
            elif spy_vol_short < spy_vol_long * 0.8:
                trend = "Subsiding"

    cross_asset_agreement = len(vols) >= 3
    score = round(min(100.0, avg_vol * 300.0), 2)
    level = "High" if avg_vol > 0.30 else ("Medium" if avg_vol > 0.15 else "Low")
    return {
        "level": level, "score": score,
        "description": f"Average realised vol: {avg_vol:.1%}",
        "stress_level": stress_level,
        "trend": trend,
        "regime": regime,
        "opportunity_context": opp_context,
        "avg_vol": avg_vol,
        "avg_vov": avg_vov,
        "worst_dd": worst_dd,
        "cross_asset_agreement": cross_asset_agreement,
    }


def compute_breadth_assessment(prices):
    """Return market breadth assessment dict."""
    if not prices:
        return {"breadth_pct": 0.0, "level": "Unknown", "description": "No data.",
                "classification": "Mixed", "pct_above_50dma": 0.0, "pct_above_200dma": 0.0}
    equity_tickers = ["SPY", "QQQ", "IWM", "EFA"]
    above_50 = 0
    above_200 = 0
    total = 0
    for t in equity_tickers:
        series = _safe_prices(prices, t)
        if series and len(series) > 50:
            total += 1
            if compute_above_ma(series, 50):
                above_50 += 1
            if len(series) > 200 and compute_above_ma(series, 200):
                above_200 += 1

    pct_above_50 = (above_50 / total * 100) if total > 0 else 0.0
    pct_above_200 = (above_200 / total * 100) if total > 0 else 0.0

    classification = "Mixed"
    if pct_above_50 >= 75 and pct_above_200 >= 75:
        classification = "Broad"
    elif pct_above_50 <= 25 or pct_above_200 <= 25:
        classification = "Narrow"

    breadth = above_50 / max(1, total)
    level = "Strong" if classification == "Broad" else ("Weak" if classification == "Narrow" else "Moderate")
    agreement = compute_directional_agreement(prices)
    return {
        "breadth_pct": round(breadth, 3),
        "level": level,
        "description": f"{above_50} of {total} equity tickers above 50-day MA.",
        "classification": classification,
        "pct_above_50dma": pct_above_50,
        "pct_above_200dma": pct_above_200,
    }


def compute_rates_credit_assessment(prices):
    """Return rates and credit assessment dict."""
    if not prices:
        return {"rates_trend": "Unknown", "credit_stress": "Unknown", "description": "No data.",
                "credit_condition": "—", "curve_proxy": "—", "liquidity_proxy": "—", "dollar_trend": "—"}
    tlt = _safe_prices(prices, "TLT")
    hyg = _safe_prices(prices, "HYG")
    lqd = _safe_prices(prices, "LQD")
    shy = _safe_prices(prices, "SHY")

    rates_trend = "Flat"
    if tlt:
        tlt_rets = _compute_returns_series(tlt)
        tlt_ret30 = tlt_rets.get("30d")
        if tlt_ret30 is not None:
            if tlt_ret30 > 0.02:
                rates_trend = "Falling"
            elif tlt_ret30 < -0.02:
                rates_trend = "Rising"

    credit_condition = "—"
    if hyg and lqd:
        hyg_rets = _compute_returns_series(hyg)
        lqd_rets = _compute_returns_series(lqd)
        hyg_ret30 = hyg_rets.get("30d")
        lqd_ret30 = lqd_rets.get("30d")
        if hyg_ret30 is not None and lqd_ret30 is not None:
            spread = hyg_ret30 - lqd_ret30
            if spread > 0.01:
                credit_condition = "Tightening"
            elif spread < -0.01:
                credit_condition = "Widening"
            else:
                credit_condition = "Stable"
    elif hyg:
        hyg_rets = _compute_returns_series(hyg)
        hyg_ret30 = hyg_rets.get("30d")
        if hyg_ret30 is not None:
            credit_condition = "Stable" if hyg_ret30 > -0.01 else "Widening"

    credit_stress = "Elevated" if credit_condition == "Widening" else "Normal"

    curve_proxy = "—"
    if tlt and shy:
        tlt_rets = _compute_returns_series(tlt)
        shy_rets = _compute_returns_series(shy)
        tlt_ret30 = tlt_rets.get("30d")
        shy_ret30 = shy_rets.get("30d")
        if tlt_ret30 is not None and shy_ret30 is not None:
            spread_change = tlt_ret30 - shy_ret30
            if spread_change > 0.01:
                curve_proxy = "Steepening"
            elif spread_change < -0.01:
                curve_proxy = "Flattening"
            else:
                curve_proxy = "Stable"

    return {
        "rates_trend": rates_trend,
        "credit_stress": credit_stress,
        "description": f"Rates {rates_trend.lower()}, credit {credit_condition.lower()}.",
        "credit_condition": credit_condition,
        "curve_proxy": curve_proxy,
        "liquidity_proxy": "—",
        "dollar_trend": "—",
    }


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
    """Return regime label string: 'Risk-On', 'Risk-Off', or 'Transitional'."""
    if not prices:
        return "Transitional"
    spy = _safe_prices(prices, "SPY")
    hyg = _safe_prices(prices, "HYG")
    qqq = _safe_prices(prices, "QQQ")

    risk_on_signals = 0
    risk_off_signals = 0
    total_signals = 0

    if spy:
        spy_rets = _compute_returns_series(spy)
        spy_ret30 = spy_rets.get("30d")
        if spy_ret30 is not None:
            total_signals += 1
            if spy_ret30 > 0.02:
                risk_on_signals += 1
            elif spy_ret30 < -0.02:
                risk_off_signals += 1
        if compute_above_ma(spy, 50):
            risk_on_signals += 1
        else:
            risk_off_signals += 1
        total_signals += 1
        if len(spy) > 200:
            if compute_above_ma(spy, 200):
                risk_on_signals += 1
            else:
                risk_off_signals += 1
            total_signals += 1

    if hyg:
        hyg_rets = _compute_returns_series(hyg)
        hyg_ret30 = hyg_rets.get("30d")
        if hyg_ret30 is not None:
            total_signals += 1
            if hyg_ret30 > 0.005:
                risk_on_signals += 1
            elif hyg_ret30 < -0.01:
                risk_off_signals += 1

    breadth = compute_breadth_assessment(prices)
    total_signals += 1
    if breadth.get("classification") == "Broad":
        risk_on_signals += 1
    elif breadth.get("classification") == "Narrow":
        risk_off_signals += 1

    if total_signals == 0:
        return "Transitional"

    on_pct = risk_on_signals / total_signals
    off_pct = risk_off_signals / total_signals

    if on_pct >= 0.7:
        return "Risk-On"
    elif off_pct >= 0.7:
        return "Risk-Off"
    else:
        return "Transitional"


def compute_executive_chips(prices_dict, vol_assessment=None, breadth_assessment=None, rates_assessment=None, regime=None):
    """Return list of executive chip dicts for dashboard display."""
    chips = []
    if not prices_dict:
        return chips

    # Regime chip
    if regime is not None:
        regime_str = regime if isinstance(regime, str) else str(regime)
        regime_color = "green" if regime_str == "Risk-On" else ("red" if regime_str == "Risk-Off" else "orange")
        chips.append({"label": "Market Regime", "value": regime_str, "color": regime_color, "trend": "neutral"})

    # SPY direction chips
    spy = _safe_prices(prices_dict, "SPY")
    if spy:
        rets = _compute_returns_series(spy)
        for days, label in [("30d", "30D"), ("90d", "90D"), ("365d", "365D")]:
            r = rets.get(days)
            if r is not None:
                dir_label = "Up" if r > 0.02 else ("Down" if r < -0.02 else "Flat")
                color = "green" if dir_label == "Up" else ("red" if dir_label == "Down" else "orange")
                chips.append({"label": f"Direction ({label})", "value": f"{dir_label} ({r:+.1%})",
                               "color": color, "trend": "up" if dir_label == "Up" else ("down" if dir_label == "Down" else "neutral")})

    # Volatility stress chip
    if vol_assessment and isinstance(vol_assessment, dict):
        sl = vol_assessment.get("stress_level", vol_assessment.get("level", "Unknown"))
        tr = vol_assessment.get("trend", "Stable")
        sc = "green" if sl == "Low" else ("orange" if sl == "Moderate" else "red")
        chips.append({"label": "Vol Stress", "value": f"{sl} · {tr}", "color": sc, "trend": "neutral"})

    # Breadth chip
    if breadth_assessment and isinstance(breadth_assessment, dict):
        bc = breadth_assessment.get("classification", breadth_assessment.get("level", "Unknown"))
        bcolor = "green" if bc == "Broad" else ("orange" if bc == "Mixed" else "red")
        chips.append({"label": "Breadth", "value": bc, "color": bcolor,
                      "trend": "up" if bc == "Broad" else ("down" if bc == "Narrow" else "neutral")})

    # Rates chip
    if rates_assessment and isinstance(rates_assessment, dict):
        rt = rates_assessment.get("rates_trend", "Unknown")
        chips.append({"label": "Rates Regime", "value": rt, "color": "#60A5FA", "trend": "neutral"})

    # Credit conditions chip
    if rates_assessment and isinstance(rates_assessment, dict):
        cc = rates_assessment.get("credit_condition", "—")
        ccolor = "green" if cc in ["Stable", "Tightening"] else ("orange" if cc == "—" else "red")
        chips.append({"label": "Credit Conditions", "value": cc, "color": ccolor, "trend": "neutral"})

    return chips


def compute_orientation_sentence(regime_or_prices, vol_assessment=None, breadth_assessment=None, rates_assessment=None):
    """Return a single sentence describing the current market orientation.

    Can be called with a regime string (canonical) or a prices dict (legacy).
    """
    # Canonical call: first arg is regime string, rest are precomputed assessments
    if isinstance(regime_or_prices, str):
        regime_word = regime_or_prices.lower().replace("-", " ")
        vol_word = "unknown"
        vol_trend = "stable"
        breadth_word = "mixed"
        credit_word = "undetermined"
        rates_word = "neutral"
        if vol_assessment and isinstance(vol_assessment, dict):
            vol_word = vol_assessment.get("stress_level", vol_assessment.get("level", "unknown")).lower()
            vol_trend = vol_assessment.get("trend", "Stable").lower()
        if breadth_assessment and isinstance(breadth_assessment, dict):
            breadth_word = breadth_assessment.get("classification", breadth_assessment.get("level", "mixed")).lower()
        if rates_assessment and isinstance(rates_assessment, dict):
            cc = rates_assessment.get("credit_condition", "—")
            credit_word = cc.lower() if cc != "—" else "undetermined"
            rates_word = rates_assessment.get("rates_trend", "Neutral").lower()
        return (
            f"Market conditions are currently {regime_word}, with {breadth_word} cross-asset participation, "
            f"{vol_word} volatility stress ({vol_trend}), {rates_word} rate trajectory, "
            f"and {credit_word} credit conditions."
        )

    # Legacy call: first arg is prices dict
    prices = regime_or_prices
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
