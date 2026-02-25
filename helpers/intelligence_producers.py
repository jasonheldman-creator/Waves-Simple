"""
Deterministic Intelligence Producers (Session-State Only)

Purpose
-------
Populate required session_state keys expected by existing panels.

Rules
-----
- NO UI output
- NO rendering logic
- Fail-open (never crash app)
- Runs automatically via intelligence_boot registry
"""

from __future__ import annotations

import streamlit as st
from helpers.intelligence_boot import register_intelligence


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_setdefault(key: str, value):
    """Safely set a default session_state value."""
    try:
        if key not in st.session_state:
            st.session_state[key] = value
    except Exception:
        pass


# ============================================================
# INTELLIGENCE PRODUCERS
# ============================================================

@register_intelligence
def build_alpha_state():
    """Baseline Alpha Intelligence state."""
    _safe_setdefault("alpha_state", "Mixed")


@register_intelligence
def build_market_context():
    """Baseline market context."""
    _safe_setdefault("market_context", "Neutral")


@register_intelligence
def build_directional_signal():
    """Directional signal placeholder."""
    _safe_setdefault("directional_signal", "Neutral")


@register_intelligence
def build_confidence_state():
    """Confidence classification baseline."""
    _safe_setdefault("confidence_state", "Low")


@register_intelligence
def build_executive_summary():
    """Executive Snapshot fallback text."""
    _safe_setdefault(
        "executive_summary",
        "System initialized. Intelligence pipelines active and awaiting live computation."
    )


@register_intelligence
def ensure_intelligence_container():
    """Guarantee intelligence root container exists."""
    _safe_setdefault("intelligence", {})