# decision_engine.py — WAVES Intelligence™ Decision Intelligence (V1→V2)
# Converts analytics into:
#   • Actions
#   • Watch items
#   • Diligence notes
#   • Daily Movement / Volatility explanation ("what changed, why, results")
#
# This is NOT trading advice.
# It is an operating-system style “what to look at next” layer.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------
def _f(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _pct(x: Any, digits: int = 2) -> str:
    v = _f(x)
    if not math.isfinite(v):
        return "N/A"
    return f"{v*100:.{digits}f}%"


def _pp(x: Any, digits: int = 2) -> str:
    """Percent points formatting (assumes already in % points, e.g., -0.15)."""
    v = _f(x)
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.{digits}f} pts"


def _num(x: Any, digits: int = 2) -> str:
    v = _f(x)
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.{digits}f}"


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return max(0.0, min(1.0, x))


def _sign_word(x: Any) -> str:
    v = _f(x)
    if not math.isfinite(v):
        return "unknown"
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "flat"


def _safe_str(x: Any, default: str = "N/A") -> str:
    try:
        s = str(x).strip()
        return s if s else default
    except Exception:
        return default


def _listify(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _top_items(items: Any, k: int = 3) -> List[str]:
    """
    Accepts:
      - list[str]
      - list[dict] like {"ticker":"NVDA","contrib":0.012} or {"name":..., "move":...}
      - dict[str, float] mapping
    Returns: list[str] formatted.
    """
    out: List[str] = []
    if items is None:
        return out

    # dict mapping
    if isinstance(items, dict):
        # sort by absolute value desc if numeric
        pairs = []
        for kk, vv in items.items():
            fv = _f(vv)
            pairs.append((kk, fv))
        pairs.sort(key=lambda t: (abs(t[1]) if math.isfinite(t[1]) else -1), reverse=True)
        for kk, vv in pairs[:k]:
            if math.isfinite(vv):
                out.append(f"{kk} ({_pct(vv, 2)})")
            else:
                out.append(str(kk))
        return out

    # list variants
    lst = _listify(items)
    if not lst:
        return out

    if all(isinstance(z, str) for z in lst):
        return [z for z in lst[:k]]

    # list of dicts
    if all(isinstance(z, dict) for z in lst):
        scored: List[Tuple[str, float]] = []
        for d in lst:
            t = _safe_str(d.get("ticker") or d.get("symbol") or d.get("name") or d.get("asset"), "Unknown")
            # prefer contrib then move
            v = d.get("contrib", d.get("contribution", d.get("move", d.get("return", d.get("chg", None)))))
            fv = _f(v)
            scored.append((t, fv))
        scored.sort(key=lambda t: (abs(t[1]) if math.isfinite(t[1]) else -1), reverse=True)
        for t, fv in scored[:k]:
            if math.isfinite(fv):
                out.append(f"{t} ({_pct(fv, 2)})")
            else:
                out.append(t)
        return out

    # fallback stringification
    return [str(z) for z in lst[:k]]


# -----------------------------
# NEW: Daily Movement / Volatility Explanation
# -----------------------------
def build_daily_wave_activity(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a demo-ready explanation object the app can render.

    Expected ctx keys (any subset; this function is resilient):
      Identity:
        - wave_name: str
        - mode: str ("Standard"/"Alpha-Minus-Beta"/"Private Logic"...)
      Returns / Alpha:
        - r1d / r30 / r60 / r365: decimal returns (0.012 = 1.2%)
        - a1d / a30 / a60 / a365: decimal alpha (0.004 = 0.4%) OR
        - alpha_1d_pts / alpha_30d_pts ... (percent points already)
      Risk / Vol:
        - vix: float
        - regime: "risk-on"|"risk-off"|"neutral"
        - rv20: decimal realized vol (0.18 = 18%)
        - te: decimal tracking error
        - ir: float information ratio
        - mdd: decimal max drawdown (negative)
      Exposure / Adjustments:
        - exposure: 0..1
        - exposure_target: 0..1
        - exposure_prev: 0..1
        - smartsafe: 0..1   (portion in SmartSafe / cash-equivalent)
        - smartsafe_prev: 0..1
        - gating_reason: str
        - rebalance: bool
        - turnover: decimal (0.6 = 60% annualized or period-specific)
        - beta_real / beta_target: floats
      Drivers:
        - top_contributors: list/dict
        - top_detractors: list/dict
        - notes: str
    """
    wave_name = _safe_str(ctx.get("wave_name") or ctx.get("wave") or ctx.get("name"), "Wave")
    mode = _safe_str(ctx.get("mode"), "Standard")

    # returns
    r1d = _f(ctx.get("r1d", ctx.get("ret_1d", ctx.get("return_1d"))))
    r30 = _f(ctx.get("r30", ctx.get("ret_30d", ctx.get("return_30d"))))
    r60 = _f(ctx.get("r60", ctx.get("ret_60d", ctx.get("return_60d"))))
    r365 = _f(ctx.get("r365", ctx.get("ret_365d", ctx.get("return_365d"))))

    # alpha (either decimals or already in percent-points)
    a1d = _f(ctx.get("a1d", ctx.get("a_1d", ctx.get("alpha_1d"))))
    a30 = _f(ctx.get("a30", ctx.get("a_30d", ctx.get("alpha_30d"))))
    a60 = _f(ctx.get("a60", ctx.get("a_60d", ctx.get("alpha_60d"))))
    a365 = _f(ctx.get("a365", ctx.get("a_365d", ctx.get("alpha_365d"))))

    alpha_1d_pts = _f(ctx.get("alpha_1d_pts"))
    alpha_30d_pts = _f(ctx.get("alpha_30d_pts"))
    alpha_60d_pts = _f(ctx.get("alpha_60d_pts"))
    alpha_365d_pts = _f(ctx.get("alpha_365d_pts"))

    # choose display alpha (prefer explicit *_pts; else convert decimals to pts)
    def _alpha_pts(dec: float, pts: float) -> float:
        if math.isfinite(pts):
            return pts
        if math.isfinite(dec):
            return dec * 100.0
        return float("nan")

    a1d_pts = _alpha_pts(a1d, alpha_1d_pts)
    a30_pts = _alpha_pts(a30, alpha_30d_pts)
    a60_pts = _alpha_pts(a60, alpha_60d_pts)
    a365_pts = _alpha_pts(a365, alpha_365d_pts)

    # risk context
    vix = _f(ctx.get("vix"))
    regime = _safe_str(ctx.get("regime"), "neutral").lower()

    rv20 = _f(ctx.get("rv20", ctx.get("realized_vol_20d", ctx.get("vol_20d"))))
    te = _f(ctx.get("te"))
    ir = _f(ctx.get("ir"))
    mdd = _f(ctx.get("mdd"))

    # exposures / adjustments
    exposure = _f(ctx.get("exposure"))
    exposure_prev = _f(ctx.get("exposure_prev"))
    exposure_target = _f(ctx.get("exposure_target"))

    smartsafe = _f(ctx.get("smartsafe", ctx.get("smartsafe_share")))
    smartsafe_prev = _f(ctx.get("smartsafe_prev", ctx.get("smartsafe_share_prev")))

    gating_reason = _safe_str(ctx.get("gating_reason"), "")
    rebalance = bool(ctx.get("rebalance", False))
    turnover = _f(ctx.get("turnover"))

    beta_real = _f(ctx.get("beta_real"))
    beta_target = _f(ctx.get("beta_target"))

    # drivers
    top_contrib = _top_items(ctx.get("top_contributors"), 3)
    top_det = _top_items(ctx.get("top_detractors"), 3)

    # Derived “what changed”
    what_changed: List[str] = []

    # Exposure change
    if math.isfinite(exposure) and math.isfinite(exposure_prev):
        delta = exposure - exposure_prev
        if abs(delta) >= 0.03:
            what_changed.append(f"Exposure changed from {_pct(exposure_prev)} to {_pct(exposure)} ({_pct(delta, 2)}).")
        else:
            what_changed.append(f"Exposure held near {_pct(exposure)} (no meaningful change).")
    elif math.isfinite(exposure):
        what_changed.append(f"Exposure is {_pct(exposure)} (prior value unavailable).")

    # SmartSafe change
    if math.isfinite(smartsafe) and math.isfinite(smartsafe_prev):
        delta = smartsafe - smartsafe_prev
        if abs(delta) >= 0.03:
            what_changed.append(f"SmartSafe allocation moved from {_pct(smartsafe_prev)} to {_pct(smartsafe)}.")
    elif math.isfinite(smartsafe):
        what_changed.append(f"SmartSafe allocation is {_pct(smartsafe)}.")

    # Rebalance / turnover
    if rebalance:
        what_changed.append("A rebalance was triggered (weights adjusted).")
    if math.isfinite(turnover):
        if turnover >= 1.2:
            what_changed.append("Turnover is elevated (>=120%/yr) — expect higher implementation churn.")
        elif turnover >= 0.6:
            what_changed.append("Turnover is moderate-to-high — active changes contributed to movement.")

    # Beta discipline
    if math.isfinite(beta_real) and math.isfinite(beta_target):
        bd = abs(beta_real - beta_target)
        if bd >= 0.07:
            what_changed.append(f"Beta drift detected: β_real {_num(beta_real,2)} vs β_target {_num(beta_target,2)}.")
        else:
            what_changed.append(f"Beta discipline: β_real {_num(beta_real,2)} vs β_target {_num(beta_target,2)} (in-range).")

    # “Why” section (contextual narrative)
    why: List[str] = []
    if regime in ("risk-off", "riskon", "risk-on", "neutral"):
        pass

    if regime == "risk-off" or (math.isfinite(vix) and vix >= 25):
        why.append("Macro conditions are risk-off (high volatility / defensive regime), so exposure controls matter more.")
    elif regime == "risk-on" or (math.isfinite(vix) and vix <= 16):
        why.append("Macro conditions are risk-on (lower volatility), so alpha capture tends to be more opportunity-driven.")
    else:
        why.append("Macro conditions are neutral/mixed, so results likely reflect idiosyncratic holdings and timing.")

    if gating_reason:
        why.append(f"Gating rationale: {gating_reason}")

    if math.isfinite(rv20):
        if rv20 >= 0.28:
            why.append(f"Realized vol is high (20D ~ {_pct(rv20)}), which can amplify day-to-day movement.")
        elif rv20 <= 0.14:
            why.append(f"Realized vol is low (20D ~ {_pct(rv20)}), so movement is more likely signal-driven than noise.")

    if math.isfinite(te):
        if te >= 0.20:
            why.append("Active risk is high (TE elevated) — divergence vs benchmark is expected.")
        elif te <= 0.08:
            why.append("Active risk is low (TE tight) — alpha is harder; small differences matter more.")

    # Results (what happened)
    results: List[str] = []

    # Prefer showing 1D, then 30D/60D, then 365D if available
    if math.isfinite(r1d):
        results.append(f"1D return: {_pct(r1d)} ({_sign_word(r1d)}).")
    if math.isfinite(a1d_pts):
        results.append(f"1D alpha: {_pp(a1d_pts)} ({_sign_word(a1d_pts)}).")

    if math.isfinite(r30):
        results.append(f"30D return: {_pct(r30)}.")
    if math.isfinite(a30_pts):
        results.append(f"30D alpha: {_pp(a30_pts)}.")

    if math.isfinite(r60):
        results.append(f"60D return: {_pct(r60)}.")
    if math.isfinite(a60_pts):
        results.append(f"60D alpha: {_pp(a60_pts)}.")

    if math.isfinite(r365):
        results.append(f"365D return: {_pct(r365)}.")
    if math.isfinite(a365_pts):
        results.append(f"365D alpha: {_pp(a365_pts)}.")

    # Add driver bullets if provided
    if top_contrib:
        results.append("Top contributors: " + ", ".join(top_contrib) + ".")
    if top_det:
        results.append("Top detractors: " + ", ".join(top_det) + ".")

    # Confidence / checks
    checks: List[str] = []
    if math.isfinite(ir):
        if ir >= 1.0:
            checks.append("Risk-adjusted alpha quality looks strong (IR >= 1.0).")
        elif ir < 0:
            checks.append("Negative IR — sanity-check benchmark truth, drift, and regime alignment.")

    if math.isfinite(mdd):
        if mdd <= -0.25:
            checks.append("Drawdown is deep — highlight downside controls and/or tighter exposure caps.")
        else:
            checks.append("Drawdown is within typical bounds for active risk level (no red flag).")

    if math.isfinite(vix):
        checks.append(f"VIX: {_num(vix,1)} ({regime}).")

    # Headline (one-liner for demos)
    # Keep it short and readable.
    headline_parts: List[str] = [wave_name, f"({mode})"]
    if math.isfinite(r1d):
        headline_parts.append(f"1D {_pct(r1d)}")
    if math.isfinite(a1d_pts):
        headline_parts.append(f"α {_pp(a1d_pts)}")
    headline = " — ".join(headline_parts)

    # Guarantee non-empty sections
    if not what_changed:
        what_changed = ["No material adjustments detected on this window."]
    if not why:
        why = ["No clear regime signal detected; treat movement as holdings-driven."]
    if not results:
        results = ["Not enough return/alpha data to summarize results for this window."]
    if not checks:
        checks = ["No additional checks."]

    return {
        "headline": headline,
        "wave_name": wave_name,
        "mode": mode,
        "what_changed": what_changed,
        "why": why,
        "results": results,
        "checks": checks,
    }


# -----------------------------
# Existing: Actions / Watch / Notes
# -----------------------------
def generate_decisions(ctx: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    ctx expected keys (any subset):
      - bm_drift: "stable"|"drift"
      - completeness_score: 0..100
      - age_days: int
      - vix: float
      - regime: "risk-on"|"risk-off"|"neutral"
      - te: decimal (0.20 = 20%)
      - ir: float
      - a30: decimal alpha
      - a60: decimal alpha
      - a365: decimal alpha
      - ahi_1d / ahi_30d / ahi_60d / ahi_365d: 0..100
      - mdd: decimal (negative)
    """
    actions: List[str] = []
    watch: List[str] = []
    notes: List[str] = []

    bm_drift = str(ctx.get("bm_drift", "stable")).lower().strip()
    completeness = _f(ctx.get("completeness_score"))
    age_days = _f(ctx.get("age_days"))
    vix = _f(ctx.get("vix"))
    regime = str(ctx.get("regime", "neutral")).lower().strip()

    te = _f(ctx.get("te"))
    ir = _f(ctx.get("ir"))

    a30 = _f(ctx.get("a30"))
    a60 = _f(ctx.get("a60"))
    a365 = _f(ctx.get("a365"))
    mdd = _f(ctx.get("mdd"))

    ahi_30 = _f(ctx.get("ahi_30d"))
    ahi_60 = _f(ctx.get("ahi_60d"))

    # --- Data integrity / demo stability
    if bm_drift != "stable":
        actions.append("Benchmark drift detected — freeze benchmark snapshot for demos / diligence.")
    if math.isfinite(completeness) and completeness < 80:
        actions.append("Coverage is below 80 — verify history source, missing business days, and engine writes.")
    elif math.isfinite(completeness) and completeness < 90:
        watch.append("Coverage is good but not perfect — watch missing days / staleness for investor demos.")

    if math.isfinite(age_days) and age_days >= 5:
        actions.append("History is stale (>=5 days) — refresh or validate data feed before external sharing.")
    elif math.isfinite(age_days) and age_days >= 3:
        watch.append("History is slightly stale (>=3 days) — consider refresh before key meetings.")

    # --- Regime posture
    if regime == "risk-off" or (math.isfinite(vix) and vix >= 25):
        actions.append("Macro regime is risk-off — emphasize SmartSafe posture and downside discipline in narrative.")
    elif regime == "risk-on" or (math.isfinite(vix) and vix <= 16):
        watch.append("Macro regime is risk-on — confirm exposure caps match mode intent.")

    # --- Active risk posture
    if math.isfinite(te) and te >= 0.20:
        watch.append("Tracking error is high — ensure stakeholders understand active risk vs benchmark.")
    elif math.isfinite(te) and te <= 0.08:
        watch.append("Tracking error is low — alpha may be harder to generate; interpret IR carefully.")

    # --- Quality / persistence
    if math.isfinite(ir) and ir >= 1.0:
        notes.append("Strong risk-adjusted excess return (IR >= 1.0).")
    elif math.isfinite(ir) and ir < 0:
        watch.append("Negative IR — investigate benchmark truth, recent regime shift, or alpha decay.")

    # --- Alpha condition
    if math.isfinite(a30) and abs(a30) >= 0.08:
        watch.append("Large 30D alpha — verify benchmark mix, missing days, and attribution assumptions.")
    if math.isfinite(a30) and math.isfinite(a60) and (a30 < 0 and a60 > 0):
        watch.append("Short-term alpha weak but medium-term positive — potential drawdown or timing effects.")
    if math.isfinite(a365) and a365 > 0 and math.isfinite(a30) and a30 < 0:
        watch.append("Long-term alpha positive but 30D weak — monitor for alpha decay vs temporary noise.")

    # --- Drawdown
    if math.isfinite(mdd) and mdd <= -0.25:
        watch.append("Deep drawdown — consider stronger downside narrative and/or tighter exposure caps.")

    # --- AHI decision layer
    if math.isfinite(ahi_30) and ahi_30 >= 80:
        notes.append("AHI indicates elite 30D relative alpha vs peers (>=80).")
    elif math.isfinite(ahi_30) and ahi_30 <= 30:
        watch.append("AHI shows weak 30D relative alpha (<=30) — investigate contributors and regime alignment.")

    if math.isfinite(ahi_60) and ahi_60 >= 80 and math.isfinite(ahi_30) and ahi_30 >= 60:
        notes.append("AHI strength persists across 30D/60D — higher confidence in relative alpha.")
    if math.isfinite(ahi_60) and ahi_60 <= 30 and math.isfinite(ahi_30) and ahi_30 <= 40:
        actions.append("Sustained weak AHI across 30D/60D — run Wave Doctor review (contributors, drift, regime).")

    # --- Always have something
    if not actions:
        actions.append("No urgent actions — system appears stable on this window.")
    if not watch:
        watch.append("No major watch items detected.")
    if not notes:
        notes.append("No additional notes.")

    return {"actions": actions, "watch": watch, "notes": notes}