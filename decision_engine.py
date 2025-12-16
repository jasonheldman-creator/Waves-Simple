# decision_engine.py — WAVES Intelligence™ Decision Intelligence (V1→V2)
# Converts analytics into:
#   • Actions
#   • Watch items
#   • Diligence notes
#
# This is NOT trading advice.
# It is an operating-system style “what to look at next” layer.

from __future__ import annotations

import math
from typing import Any, Dict, List


def _f(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


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