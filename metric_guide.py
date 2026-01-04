# metric_guide.py — WAVES Intelligence™ Metric Interpretation Layer (V1→V2)
# Central registry for:
#   • Grade bands (A+/A/B/C/D)
#   • Plain-English interpretations
#   • Consistent rendering helpers
#
# Notes:
#   • No matplotlib dependence
#   • Safe for Streamlit Cloud
#   • Keeps UI institutional (no gimmicks)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# ----------------------------
# Core banding utilities
# ----------------------------
Band = Tuple[float, float, str, str]  # lo, hi, grade, label


def band_lookup(value: float, bands: List[Band]) -> Tuple[str, str]:
    try:
        v = float(value)
        if not math.isfinite(v):
            return ("N/A", "No data")
        for lo, hi, g, label in bands:
            if v >= lo and v < hi:
                return (g, label)
        # clamp
        if v < bands[0][0]:
            return (bands[0][2], bands[0][3])
        return (bands[-1][2], bands[-1][3])
    except Exception:
        return ("N/A", "No data")


def grade_badge(grade: str, label: str) -> str:
    return f"{grade} — {label}"


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


# ----------------------------
# Metric specs
# ----------------------------
@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    unit: str
    better: str  # "higher", "lower", "context"
    bands: Optional[List[Band]] = None
    explainer: str = ""


def get_specs() -> Dict[str, MetricSpec]:
    """
    Single source of truth for all grading bands + explainers.
    Add new metrics here once; the whole console stays consistent.
    """
    return {
        # Difficulty proxy is clipped to roughly [-25, +25] in your app
        "difficulty_vs_spy": MetricSpec(
            key="difficulty_vs_spy",
            title="Difficulty vs SPY (proxy)",
            unit="pts",
            better="context",
            bands=[
                (-25.0, -15.0, "A+", "Much easier than SPY"),
                (-15.0,  -5.0, "A",  "Easier than SPY"),
                ( -5.0,   5.0, "B",  "SPY-like baseline"),
                (  5.0,  15.0, "C",  "Harder than SPY"),
                ( 15.0,  25.1, "D",  "Much harder than SPY"),
            ],
            explainer=(
                "Degree-of-difficulty proxy based on benchmark concentration/diversification structure. "
                "Negative means more diversified (SPY-like or easier). Positive means more idiosyncratic "
                "(harder to beat consistently)."
            ),
        ),

        "hhi": MetricSpec(
            key="hhi",
            title="HHI (Concentration)",
            unit="0–1",
            better="lower",
            bands=[
                (0.00, 0.05, "A+", "Very diversified"),
                (0.05, 0.10, "A",  "Diversified"),
                (0.10, 0.20, "B",  "Moderately concentrated"),
                (0.20, 0.35, "C",  "Concentrated"),
                (0.35, 1.01, "D",  "Highly concentrated"),
            ],
            explainer=(
                "Herfindahl-Hirschman Index (HHI). Lower implies more diversified benchmark composition. "
                "Higher implies concentration and single-name / sector dominance."
            ),
        ),

        "entropy": MetricSpec(
            key="entropy",
            title="Entropy (Diversification)",
            unit="nats",
            better="higher",
            bands=[
                (0.00, 1.00, "D",  "Very low diversification"),
                (1.00, 1.80, "C",  "Low diversification"),
                (1.80, 2.50, "B",  "Moderate diversification"),
                (2.50, 3.20, "A",  "High diversification"),
                (3.20, 9.99, "A+", "Very high diversification"),
            ],
            explainer=(
                "Entropy proxy for diversification. Higher generally indicates broader spread among constituents. "
                "Interpret relative to benchmark size."
            ),
        ),

        "top_weight": MetricSpec(
            key="top_weight",
            title="Top Weight",
            unit="%",
            better="lower",
            bands=[
                (0.00, 0.08, "A+", "No single-name dominance"),
                (0.08, 0.15, "A",  "Healthy concentration"),
                (0.15, 0.25, "B",  "Moderate single-name risk"),
                (0.25, 0.40, "C",  "High single-name risk"),
                (0.40, 1.01, "D",  "Extreme single-name risk"),
            ],
            explainer=(
                "Largest single constituent weight. Lower reduces concentration risk and benchmark fragility."
            ),
        ),

        "ahi": MetricSpec(
            key="ahi",
            title="Alpha Heat Index (0–100)",
            unit="score",
            better="higher",
            bands=[
                (0.0,  20.0,  "D",  "Very weak relative alpha"),
                (20.0, 40.0,  "C",  "Weak relative alpha"),
                (40.0, 60.0,  "B",  "Around median"),
                (60.0, 80.0,  "A",  "Strong relative alpha"),
                (80.0, 101.0, "A+", "Elite relative alpha"),
            ],
            explainer=(
                "Relative alpha surface normalized to 0–100, where 50 is median by design. "
                "This is observational intelligence (not a trading signal)."
            ),
        ),

        "wavescore": MetricSpec(
            key="wavescore",
            title="WaveScore (0–100)",
            unit="score",
            better="higher",
            bands=[
                (0.0,  60.0,  "D",  "Needs improvement"),
                (60.0, 70.0,  "C",  "Acceptable"),
                (70.0, 80.0,  "B",  "Good"),
                (80.0, 90.0,  "A",  "Strong"),
                (90.0, 101.0, "A+", "Elite"),
            ],
            explainer=(
                "Console-side approximation of your locked WaveScore framework. "
                "Use for directional ranking and monitoring."
            ),
        ),

        "ir": MetricSpec(
            key="ir",
            title="Information Ratio",
            unit="ratio",
            better="higher",
            bands=[
                (-9.0, -0.25, "D",  "Negative / weak"),
                (-0.25, 0.25, "C",  "Low"),
                (0.25, 0.75,  "B",  "Moderate"),
                (0.75, 1.25,  "A",  "Strong"),
                (1.25, 9.0,   "A+", "Elite"),
            ],
            explainer="Risk-adjusted excess return vs benchmark using tracking error. Higher is better.",
        ),

        "te": MetricSpec(
            key="te",
            title="Tracking Error (ann.)",
            unit="%",
            better="context",
            bands=[
                (0.00, 0.05, "A+", "Very low active risk"),
                (0.05, 0.10, "A",  "Low active risk"),
                (0.10, 0.20, "B",  "Moderate active risk"),
                (0.20, 0.35, "C",  "High active risk"),
                (0.35, 9.99, "D",  "Very high active risk"),
            ],
            explainer=(
                "Annualized volatility of active returns vs benchmark. Not inherently good/bad—"
                "interpret alongside alpha and risk posture."
            ),
        ),

        "completeness": MetricSpec(
            key="completeness",
            title="Data Completeness",
            unit="/100",
            better="higher",
            bands=[
                (0.0,  60.0,  "D",  "Unreliable coverage"),
                (60.0, 75.0,  "C",  "Limited coverage"),
                (75.0, 90.0,  "B",  "Good coverage"),
                (90.0, 97.0,  "A",  "Strong coverage"),
                (97.0, 101.0, "A+", "Excellent coverage"),
            ],
            explainer=(
                "Coverage score based on missing business days and staleness. Higher is better for diligence."
            ),
        ),

        "age_days": MetricSpec(
            key="age_days",
            title="Data Age (days)",
            unit="days",
            better="lower",
            bands=[
                (0.0,  2.0,  "A+", "Fresh"),
                (2.0,  4.0,  "A",  "Recent"),
                (4.0,  7.0,  "B",  "Slightly stale"),
                (7.0,  14.0, "C",  "Stale"),
                (14.0, 9999.0, "D", "Very stale"),
            ],
            explainer="How old the last datapoint is. Lower is better for live demos and diligence.",
        ),
    }


def grade(metric_key: str, value: float) -> Tuple[str, str]:
    specs = get_specs()
    spec = specs.get(metric_key)
    if spec is None or not spec.bands:
        return ("N/A", "No grade bands")
    return band_lookup(value, spec.bands)


def explain(metric_key: str) -> str:
    specs = get_specs()
    spec = specs.get(metric_key)
    return spec.explainer if spec else ""


def table_for(metric_key: str) -> List[Tuple[str, str]]:
    """
    Returns list of (grade, meaning) in band order for display.
    """
    specs = get_specs()
    spec = specs.get(metric_key)
    if spec is None or not spec.bands:
        return []
    rows = []
    for lo, hi, g, label in spec.bands:
        rows.append((g, f"{label}  [{lo:g} to {hi:g})"))
    return rows