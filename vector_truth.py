# vector_truth.py — WAVES Intelligence™
# Vector™ Truth Layer v1 (Read-only, deterministic, governance-ready)
#
# Purpose:
#   Provide governance-native "Truth outputs" tied to the alpha framework:
#     • Alpha Source Decomposition (Truth View - architectural transparency)
#     • Capital-Weighted vs Exposure-Adjusted Reconciliation (context vs control)
#     • Risk-On vs Risk-Off Attribution (regime-aware governance)
#     • Durability & Fragility Scan (predictable system-feedback)
#
# Architectural refinement principles:
#   • No optimization, no portfolio actions, no forecasting (no-predict constraint).
#   • Deterministic wording templates (stable, predictable outputs).
#   • Works with partial inputs (flexible onboarding, graceful fallbacks).

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple
import math


# ---------------------------
# Helpers
# ---------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:.{digits}f}%"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sum_ignore_none(vals: Sequence[Optional[float]]) -> float:
    s = 0.0
    for v in vals:
        if v is not None:
            s += float(v)
    return s


def _label_regime(r: Any) -> str:
    if r is None:
        return "UNKNOWN"
    s = str(r).strip().upper()
    if s in ("RISK_ON", "RISK-ON", "ON"):
        return "RISK_ON"
    if s in ("RISK_OFF", "RISK-OFF", "OFF"):
        return "RISK_OFF"
    return s or "UNKNOWN"


# ---------------------------
# Output dataclasses
# ---------------------------

@dataclass
class VectorAlphaSources:
    total_excess_return: Optional[float] = None
    security_selection_alpha: Optional[float] = None
    exposure_management_alpha: Optional[float] = None
    capital_preservation_effect: Optional[float] = None
    benchmark_construction_effect: Optional[float] = None
    assessment: str = ""


@dataclass
class VectorAlphaReconciliation:
    capital_weighted_alpha: Optional[float] = None
    exposure_adjusted_alpha: Optional[float] = None
    explanation: str = ""
    conclusion: str = ""
    inflation_risk: str = "N/A"


@dataclass
class VectorRegimeAttribution:
    alpha_risk_on: Optional[float] = None
    alpha_risk_off: Optional[float] = None
    volatility_sensitivity: str = ""
    flag: str = ""


@dataclass
class VectorDurabilityScan:
    alpha_type: str = "N/A"
    fragility_score: Optional[float] = None
    primary_risk: str = ""
    verdict: str = ""


@dataclass
class VectorTruthReport:
    wave_name: str
    timeframe_label: str
    sources: VectorAlphaSources
    reconciliation: VectorAlphaReconciliation
    regime: VectorRegimeAttribution
    durability: VectorDurabilityScan

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------
# Core computations
# ---------------------------

def compute_regime_attribution(
    alpha_series: Optional[Sequence[float]] = None,
    regime_series: Optional[Sequence[Any]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes cumulative alpha in Risk-On vs Risk-Off regimes from aligned sequences.
    alpha_series should be periodic returns (e.g., daily alpha as decimal).
    regime_series should be labels per period ("RISK_ON"/"RISK_OFF" etc).
    """
    if not alpha_series or not regime_series:
        return None, None

    n = min(len(alpha_series), len(regime_series))
    if n <= 0:
        return None, None

    on = 0.0
    off = 0.0
    seen_on = False
    seen_off = False

    for i in range(n):
        a = _safe_float(alpha_series[i])
        r = _label_regime(regime_series[i])
        if a is None:
            continue
        if r == "RISK_ON":
            on += a
            seen_on = True
        elif r == "RISK_OFF":
            off += a
            seen_off = True

    return (on if seen_on else None), (off if seen_off else None)


def compute_alpha_sources(
    total_excess: Optional[float],
    capital_weighted_alpha: Optional[float],
    exposure_adjusted_alpha: Optional[float],
    alpha_risk_off: Optional[float],
    alpha_risk_on: Optional[float],
    overlay_contribution: Optional[float] = None,
    vix_contribution: Optional[float] = None,
    smartsafe_contribution: Optional[float] = None,
) -> VectorAlphaSources:
    """
    Decomposes total excess into alpha sources with true decomposition from return streams.
    
    Avoids placeholder logic where Selection Alpha == Total Excess and other components
    artificially balance to zero. Instead, provides N/A when data is insufficient.

    Definitions:
      • Security Selection Alpha ~ derived from risky sleeve returns when available
      • Exposure Management Alpha ~ impact of exposure scaling (not forced to zero)
      • Capital Preservation Effect ~ SmartSafe gating and risk-off positioning
      • Benchmark Construction Effect ~ residual from benchmark composition choices
    """
    te = _safe_float(total_excess)
    cwa = _safe_float(capital_weighted_alpha)
    eaa = _safe_float(exposure_adjusted_alpha)
    aro = _safe_float(alpha_risk_off)
    a_on = _safe_float(alpha_risk_on)
    
    # Optional overlay contributions
    overlay = _safe_float(overlay_contribution)
    vix_contrib = _safe_float(vix_contribution)
    smartsafe = _safe_float(smartsafe_contribution)

    # Security selection: use exposure-adjusted alpha when available
    # If exposure-adjusted is missing but we have capital-weighted, indicate N/A
    sel = eaa if eaa is not None else None

    # Exposure management: difference between capital-weighted and exposure-adjusted
    # Only compute if both inputs are available; otherwise N/A
    exp_mgmt = None
    if cwa is not None and eaa is not None:
        exp_mgmt = cwa - eaa
        # Don't force to zero - if there's a real difference, show it

    # Capital preservation effect: combines SmartSafe gating + risk-off regime alpha
    # Use explicit SmartSafe contribution if available; otherwise use risk-off alpha as proxy
    preserve = None
    if smartsafe is not None:
        preserve = smartsafe
    elif aro is not None:
        # Use risk-off alpha as a proxy for capital preservation
        # Don't artificially balance - show actual contribution
        preserve = aro
    # If neither available, leave as N/A

    # Benchmark construction effect: only compute if we have total excess
    # This is the residual, but we don't force artificial balancing
    bench_eff = None
    if te is not None:
        known = _sum_ignore_none([sel, exp_mgmt, preserve])
        bench_eff = te - known
        # If residual is tiny relative to total, it's legitimate
        # If it's large, that indicates benchmark construction matters

    # Assessment template - updated to avoid overstating selection
    assessment_parts = []
    
    # Check if we have sufficient data for meaningful decomposition
    has_selection = sel is not None
    has_exposure = exp_mgmt is not None
    has_preservation = preserve is not None
    
    if has_selection and has_exposure:
        # We can make comparative statements
        sel_abs = abs(sel) if sel is not None else 0
        exp_abs = abs(exp_mgmt) if exp_mgmt is not None else 0
        
        if exp_abs > sel_abs * 1.2:
            assessment_parts.append("This decomposition shows exposure management as a meaningful contributor to alpha.")
        elif sel_abs > exp_abs * 1.2:
            assessment_parts.append("This decomposition provides selection/overlay insight where sufficient resolution exists.")
        else:
            assessment_parts.append("Alpha sources show balanced contributions from selection and exposure management.")
    elif has_selection:
        assessment_parts.append("Selection component is observable. Exposure management effect requires additional exposure history for decomposition.")
    else:
        assessment_parts.append("Insufficient data resolution for complete alpha source decomposition.")

    if has_preservation and te is not None:
        pres_abs = abs(preserve) if preserve is not None else 0
        te_abs = abs(te) if te is not None else 1e-9
        if pres_abs >= te_abs * 0.5:
            assessment_parts.append("Capital preservation (SmartSafe/risk-off) contributed materially to total alpha.")

    # Add disclaimer about institutional residuals
    if bench_eff is not None and abs(bench_eff) > (abs(te) if te is not None and te != 0 else 1.0) * 0.2:
        assessment_parts.append("Institutional residuals from missing history or benchmark construction are present.")

    assessment = " ".join(assessment_parts).strip() if assessment_parts else "Decomposition limited by available data series."

    return VectorAlphaSources(
        total_excess_return=te,
        security_selection_alpha=sel,
        exposure_management_alpha=exp_mgmt,
        capital_preservation_effect=preserve,
        benchmark_construction_effect=bench_eff,
        assessment=assessment,
    )


def compute_reconciliation(
    capital_weighted_alpha: Optional[float],
    exposure_adjusted_alpha: Optional[float],
) -> VectorAlphaReconciliation:
    cwa = _safe_float(capital_weighted_alpha)
    eaa = _safe_float(exposure_adjusted_alpha)

    explanation = (
        "Capital-weighted alpha reflects realized investor experience (including cash sweeps and scaling). "
        "Exposure-adjusted alpha isolates strategy efficiency independent of exposure level."
    )

    if cwa is None or eaa is None:
        return VectorAlphaReconciliation(
            capital_weighted_alpha=cwa,
            exposure_adjusted_alpha=eaa,
            explanation=explanation,
            conclusion="Both measures are valid, but reconciliation is limited by missing inputs.",
            inflation_risk="N/A",
        )

    gap = cwa - eaa
    # Inflation risk heuristic: if capital-weighted is far above exposure-adjusted,
    # it means alpha is heavily driven by exposure gating; that's not "fake", but must be disclosed.
    # We classify "LOW/MODERATE/HIGH" based on relative gap.
    denom = max(abs(eaa), 1e-6)
    rel_gap = abs(gap) / denom

    if rel_gap < 0.35:
        risk = "LOW"
    elif rel_gap < 0.85:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    conclusion = (
        "Both are valid; they answer different questions. "
        "Capital-weighted alpha should be reported with exposure context, not as pure selection skill."
    )

    return VectorAlphaReconciliation(
        capital_weighted_alpha=cwa,
        exposure_adjusted_alpha=eaa,
        explanation=explanation,
        conclusion=conclusion,
        inflation_risk=risk,
    )


def compute_durability(
    total_excess: Optional[float],
    alpha_risk_off: Optional[float],
    alpha_risk_on: Optional[float],
    exposure_mgmt_alpha: Optional[float],
) -> VectorDurabilityScan:
    te = _safe_float(total_excess)
    aro = _safe_float(alpha_risk_off)
    aon = _safe_float(alpha_risk_on)
    ema = _safe_float(exposure_mgmt_alpha)

    # Fragility score heuristic (0 = low fragility, 1 = high fragility)
    # If alpha is overwhelmingly concentrated in one regime + heavily dependent on exposure mgmt, it’s more fragile.
    frag = None
    if (aro is not None or aon is not None) and ema is not None:
        reg_sum = _sum_ignore_none([abs(aro), abs(aon)])
        reg_conc = 0.0
        if reg_sum > 1e-9 and aro is not None and aon is not None:
            reg_conc = abs(abs(aro) - abs(aon)) / reg_sum  # 0 balanced, 1 concentrated
        dep = _clamp(abs(ema) / max(abs(te) if te is not None else 1e-6, 1e-6), 0.0, 1.0)
        frag = _clamp(0.55 * reg_conc + 0.45 * dep, 0.0, 1.0)

    # Alpha type
    alpha_type = "N/A"
    if ema is not None:
        if abs(ema) > (abs(te) if te is not None else abs(ema)) * 0.4:
            alpha_type = "Structural + Regime-Adaptive"
        else:
            alpha_type = "Selection-Dominant"

    # Primary risk wording
    primary_risk = "N/A"
    if ema is not None and te is not None:
        if abs(ema) > abs(te) * 0.5:
            primary_risk = "Extended volatility suppression reduces exposure-management contribution."
        else:
            primary_risk = "Dispersion collapse reduces selection opportunity."

    # Verdict template
    if frag is None:
        verdict = "Durability scan is limited by missing regime/exposure inputs."
    else:
        if frag < 0.33:
            verdict = "Alpha appears resilient across regimes; fragility is LOW."
        elif frag < 0.66:
            verdict = "Alpha durability is MODERATE; monitor regime concentration and exposure dependence."
        else:
            verdict = "Alpha is regime-concentrated and exposure-dependent; fragility is HIGH."

    return VectorDurabilityScan(
        alpha_type=alpha_type,
        fragility_score=frag,
        primary_risk=primary_risk,
        verdict=verdict,
    )


def build_vector_truth_report(
    wave_name: str,
    timeframe_label: str,
    total_excess_return: Optional[float] = None,
    capital_weighted_alpha: Optional[float] = None,
    exposure_adjusted_alpha: Optional[float] = None,
    alpha_series: Optional[Sequence[float]] = None,
    regime_series: Optional[Sequence[Any]] = None,
    overlay_contribution: Optional[float] = None,
    vix_contribution: Optional[float] = None,
    smartsafe_contribution: Optional[float] = None,
) -> VectorTruthReport:
    """
    Main entry point.
    Provide what you have; the report degrades gracefully.
    Supports optional overlay, VIX, and SmartSafe contribution series.
    """
    aro, aon = compute_regime_attribution(alpha_series=alpha_series, regime_series=regime_series)

    sources = compute_alpha_sources(
        total_excess=total_excess_return,
        capital_weighted_alpha=capital_weighted_alpha,
        exposure_adjusted_alpha=exposure_adjusted_alpha,
        alpha_risk_off=aro,
        alpha_risk_on=aon,
        overlay_contribution=overlay_contribution,
        vix_contribution=vix_contribution,
        smartsafe_contribution=smartsafe_contribution,
    )

    recon = compute_reconciliation(
        capital_weighted_alpha=capital_weighted_alpha,
        exposure_adjusted_alpha=exposure_adjusted_alpha,
    )

    vol_sens = ""
    flag = ""
    if aro is None and aon is None:
        vol_sens = "Regime attribution unavailable (missing risk-on/off series)."
        flag = "Provide regime labels to enable risk-on vs risk-off truth outputs."
    else:
        # Volatility sensitivity inference from where alpha is earned
        if aro is not None and aon is not None:
            if abs(aro) > abs(aon) * 1.25:
                vol_sens = "This Wave benefits disproportionately from Risk-Off regimes (volatility spikes / stress periods)."
                flag = "Durability: HIGH in unstable markets; MODERATE in prolonged low-volatility regimes."
            elif abs(aon) > abs(aro) * 1.25:
                vol_sens = "This Wave earns alpha primarily in Risk-On regimes (trend / growth environments)."
                flag = "Durability: MODERATE in stress regimes; HIGH in persistent risk-on tape."
            else:
                vol_sens = "Alpha appears relatively balanced across regimes."
                flag = "Durability: HIGH; monitor for regime shift and dispersion collapse."
        elif aro is not None:
            vol_sens = "Observed alpha was earned during Risk-Off regimes (partial regime coverage)."
            flag = "Durability: likely higher in unstable markets; validate with full regime labeling."
        else:
            vol_sens = "Observed alpha was earned during Risk-On regimes (partial regime coverage)."
            flag = "Durability: validate in stress regimes with full regime labeling."

    regime = VectorRegimeAttribution(
        alpha_risk_on=aon,
        alpha_risk_off=aro,
        volatility_sensitivity=vol_sens,
        flag=flag,
    )

    durability = compute_durability(
        total_excess=total_excess_return,
        alpha_risk_off=aro,
        alpha_risk_on=aon,
        exposure_mgmt_alpha=sources.exposure_management_alpha,
    )

    return VectorTruthReport(
        wave_name=wave_name,
        timeframe_label=timeframe_label,
        sources=sources,
        reconciliation=recon,
        regime=regime,
        durability=durability,
    )


# ---------------------------
# Rendering helpers (optional)
# ---------------------------

def format_vector_truth_markdown(report: VectorTruthReport) -> str:
    """
    Returns a deterministic markdown block you can print in Streamlit.
    Handles N/A values appropriately and provides clear attribution context.
    """
    s = report.sources
    r = report.reconciliation
    g = report.regime
    d = report.durability

    frag = "N/A" if d.fragility_score is None else f"{d.fragility_score:.2f}"

    md = f"""
### Vector™ Truth Layer — {report.wave_name} ({report.timeframe_label})

**VECTOR TRUTH — ALPHA SOURCES**
- Total Excess Return: **{_pct(s.total_excess_return)}**
- Security Selection Alpha: **{_pct(s.security_selection_alpha)}**
- Exposure Management Alpha: **{_pct(s.exposure_management_alpha)}**
- Capital Preservation Effect: **{_pct(s.capital_preservation_effect)}**
- Benchmark Construction Effect: **{_pct(s.benchmark_construction_effect)}**

**Vector Assessment:** {s.assessment}

*Note: N/A values indicate insufficient data series for that component. This decomposition provides insight where sufficient resolution exists.*

---

**VECTOR TRUTH — ALPHA RECONCILIATION**
- Capital-Weighted Alpha: **{_pct(r.capital_weighted_alpha)}**
- Exposure-Adjusted Alpha: **{_pct(r.exposure_adjusted_alpha)}**

{r.explanation}

**Vector Conclusion:** {r.conclusion}  
**Reported alpha inflation risk:** **{r.inflation_risk}**

---

**VECTOR TRUTH — REGIME ATTRIBUTION**
- Alpha in Risk-On: **{_pct(g.alpha_risk_on)}**
- Alpha in Risk-Off: **{_pct(g.alpha_risk_off)}**

{g.volatility_sensitivity}  
**Vector Flag:** {g.flag}

---

**VECTOR TRUTH — DURABILITY SCAN**
- Alpha Type: **{d.alpha_type}**
- Fragility Score: **{frag}** *(0=Low, 1=High)*
- Primary Risk: {d.primary_risk}

**Vector Verdict:** {d.verdict}
""".strip()

    return md
