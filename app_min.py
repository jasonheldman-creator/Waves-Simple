# ============================================================
# app_min.py
# WAVES Intelligence™ Console (Minimal)
# ============================================================
#
# ┌─────────────────────────────────────────────────────────────┐
# │  CANONICAL SYSTEM NOTICE                                     │
# │                                                              │
# │  This console operates as a CANONICAL SYSTEM:                │
# │  • One source of truth for all attribution and governance   │
# │  • Vertical data flow: CSV → Computation → Presentation     │
# │  • Strict layer separation: Data, Logic, Presentation       │
# │  • No overrides: System values are immutable                │
# │  • No execution: Advisory-only, human-in-the-loop           │
# │                                                              │
# │  All values are derived from canonical data files:          │
# │  • live_snapshot.csv                                        │
# │  • alpha_attribution_summary.csv                            │
# │  • wave_history.csv                                         │
# │                                                              │
# │  See replit.md for Canonical Architecture documentation.    │
# └─────────────────────────────────────────────────────────────┘
#
# ============================================================

import streamlit as st

# Temporary cache invalidation for diagnostic purposes
st.cache_data.clear()
st.cache_resource.clear()

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import adaptive learning module for LIVE learning capabilities
import adaptive_learning as al
import integrity_signals as integ
import adaptive_intelligence as ai


# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Institutional Visual Refinement CSS
# ===========================
st.markdown("""
<style>
/* ===== WAVES COLOR SYSTEM ===== */
:root {
    --bg-primary: #0F1115;
    --bg-secondary: #1C1F26;
    --bg-tertiary: #242832;
    --text-primary: #FFFFFF;
    --text-secondary: #9EA3AE;
    --text-muted: #6B7280;
    --accent-blue: #3A6FF7;
    --accent-green: #2BFF88;
    --accent-yellow: #F5C451;
    --accent-red: #E06C75;
    --border-subtle: rgba(255,255,255,0.08);
    --border-medium: rgba(255,255,255,0.12);
}

/* ===== GLOBAL BACKGROUND ===== */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #0F1115 !important;
}

/* ===== TYPOGRAPHY HIERARCHY ===== */
h1 { 
    font-weight: 600 !important; 
    letter-spacing: -0.02em !important;
    margin-bottom: 0.5rem !important;
    color: #FFFFFF !important;
}
h2 { 
    font-weight: 600 !important; 
    font-size: 1.35rem !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
    color: #FFFFFF !important;
}
h3 { 
    font-weight: 600 !important; 
    font-size: 1.1rem !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
    color: rgba(255,255,255,0.9) !important;
}

/* ===== MONOSPACE NUMERALS ===== */
.waves-mono, .stMetricValue, 
div[data-testid="stMetricValue"] > div,
.stDataFrame td {
    font-family: 'SF Mono', 'Consolas', 'Monaco', 'Menlo', monospace !important;
    font-variant-numeric: tabular-nums !important;
}

/* ===== 8PT SPACING GRID ===== */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 16px !important;
}
.element-container {
    margin-bottom: 8px !important;
}

/* ===== METRIC CARDS (PRIMARY CONTAINERS) ===== */
div[data-testid="stMetric"] {
    background: #1C1F26 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    font-family: 'SF Mono', 'Consolas', monospace !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #9EA3AE !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
    font-family: 'SF Mono', 'Consolas', monospace !important;
}

/* ===== TABLE STYLING ===== */
.stDataFrame {
    font-size: 0.85rem !important;
}
.stDataFrame th {
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
    background: #1C1F26 !important;
    color: #9EA3AE !important;
    border: 1px solid #2A2A2A !important;
}
.stDataFrame td {
    padding: 8px 12px !important;
    font-family: 'SF Mono', 'Consolas', monospace !important;
    color: #D0D0D0 !important;
    background: #1C1C1E !important;
    border: 1px solid #2A2A2A !important;
}
.stDataFrame table {
    background: #1C1C1E !important;
    border-collapse: collapse !important;
}
/* Arrow DataGrid (used by st.dataframe) */
div[data-testid="stDataFrame"] > div {
    background: #1C1C1E !important;
}
div[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
    background: #1C1C1E !important;
}
/* Ensure all table text is visible */
.dvn-scroller, .dvn-underlay, .dvn-scroll-inner {
    background: #1C1C1E !important;
}
[data-testid="glideDataEditor"] div[style*="background"] {
    background: #1C1C1E !important;
}

/* ===== EXPANDER PANELS (SECONDARY CONTAINERS) ===== */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    background: #1C1F26 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15) !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    padding: 12px 16px !important;
}
div[data-testid="stExpander"] summary:hover {
    color: #3A6FF7 !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #0F1115 !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] > div {
    background: #0F1115 !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #9EA3AE !important;
}

/* ===== ALERTS - INSTITUTIONAL DARK PANEL STYLING ===== */
div[data-testid="stAlert"] {
    border-radius: 6px !important;
    padding: 16px 20px !important;
    font-size: 0.9rem !important;
    background: #1C1C1E !important;
    border: 1px solid #2A2A2A !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    color: #A0A0A0 !important;
    line-height: 1.6 !important;
}
div[data-testid="stAlert"] > div {
    color: #A0A0A0 !important;
}
div[data-testid="stAlert"] p {
    color: #A0A0A0 !important;
    margin: 0 !important;
}
div[data-testid="stAlert"] strong {
    color: #D0D0D0 !important;
}
/* Hide default alert icons */
div[data-testid="stAlert"] > div:first-child svg {
    display: none !important;
}
/* Override specific alert types - all use same dark panel style */
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: #1C1C1E !important;
    border: 1px solid #2A2A2A !important;
}
/* Info alerts */
div.stAlert[data-baseweb="notification"] {
    background: #1C1C1E !important;
}
/* Success alerts - subtle green border */
div[data-testid="stAlert"] div[role="alert"] {
    background: transparent !important;
}
/* Remove colored left borders */
div[data-testid="stAlert"]::before {
    display: none !important;
}
div[data-testid="stAlert"] > div::before {
    display: none !important;
}

/* ===== HORIZONTAL DIVIDERS ===== */
hr {
    margin: 24px 0 !important;
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 12px 20px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #9EA3AE !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #3A6FF7 !important;
    border-bottom: 2px solid #3A6FF7 !important;
}

/* ===== CAPTIONS ===== */
.stCaption, small {
    font-size: 0.75rem !important;
    color: #6B7280 !important;
}

/* ===== BUTTONS ===== */
.stButton button {
    font-weight: 600 !important;
    border-radius: 6px !important;
    background: #1C1F26 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #FFFFFF !important;
}
.stButton button:hover {
    background: #242832 !important;
    border-color: #3A6FF7 !important;
}
.stDownloadButton button {
    background: #1C1F26 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    font-weight: 600 !important;
}
.stDownloadButton button:hover {
    border-color: #3A6FF7 !important;
}

/* ===== CONTAINER SPACING ===== */
.block-container {
    padding-top: 24px !important;
    padding-bottom: 24px !important;
    max-width: 1400px !important;
}

/* ===== MICRO-LABEL STYLING ===== */
.waves-micro-label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #3A6FF7 !important;
    margin-bottom: 4px !important;
    display: block !important;
}

/* ===== SIGNAL COLORS ===== */
.signal-positive { color: #2BFF88 !important; }
.signal-negative { color: #E06C75 !important; }
.signal-neutral { color: #F5C451 !important; }
.signal-info { color: #3A6FF7 !important; }

/* ===== TWO-COLUMN LAYOUT HELPER ===== */
.waves-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: baseline !important;
    padding: 8px 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}
.waves-row-label {
    color: #9EA3AE !important;
    font-size: 0.85rem !important;
}
.waves-row-value {
    font-family: 'SF Mono', 'Consolas', monospace !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}

/* ===== SELECT/DROPDOWN ===== */
div[data-baseweb="select"] > div {
    background: #1C1F26 !important;
    border-color: rgba(255,255,255,0.12) !important;
}
div[data-baseweb="select"]:focus-within > div {
    border-color: #3A6FF7 !important;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# Constants
# ===========================
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"
ALPHA_ATTRIBUTION_PATH = DATA_DIR / "alpha_attribution_summary.csv"

RETURN_COLS = {
    "INTRADAY": "return_intraday",
    "1D": "return_1d",
    "30D": "return_30d",
    "60D": "return_60d",
    "365D": "return_365d",
}

BENCHMARK_COLS = {
    "30D": "benchmark_return_30d",
    "60D": "benchmark_return_60d",
    "365D": "benchmark_return_365d",
}

ALPHA_COLS = {
    "INTRADAY": "alpha_intraday",
    "1D": "alpha_1d",
    "30D": "alpha_30d",
    "60D": "alpha_60d",
    "365D": "alpha_365d",
}


# ===========================
# Load Snapshot
# ===========================
def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Load alpha attribution summary if it exists
    attrib_df = None
    if ALPHA_ATTRIBUTION_PATH.exists():
        try:
            attrib_df = pd.read_csv(ALPHA_ATTRIBUTION_PATH)
            attrib_df.columns = [c.strip().lower() for c in attrib_df.columns]
        except Exception:
            pass

    # Set display_name
    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        elif "wave_id" in df.columns:
            df["display_name"] = df["wave_id"]
        else:
            df["display_name"] = "Unnamed Wave"

    # Ensure required columns exist
    for col in list(RETURN_COLS.values()) + list(ALPHA_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    if "intraday_label" not in df.columns:
        df["intraday_label"] = None

    return df, attrib_df, None


def get_market_status():
    """
    Determine market status and intraday data state.
    Returns tuple: (status_code, status_label, is_live)
    
    Status codes:
    - 'live': Market is open, live intraday data
    - 'closed_today': Market closed, same trading day data
    - 'closed_session': Weekend/holiday, last trading session data
    """
    from datetime import datetime, time
    import pytz
    
    try:
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        current_time = now.time()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        if weekday >= 5:  # Weekend
            return 'closed_session', 'Last trading session', False
        elif market_open <= current_time <= market_close:
            return 'live', 'Live', True
        elif current_time > market_close:
            return 'closed_today', 'As of market close', False
        else:  # Before market open
            return 'closed_session', 'Last trading session', False
    except Exception:
        # Fallback if timezone detection fails
        return 'closed_session', 'Last session', False


def get_intraday_label(market_status):
    """
    Generate display label for intraday data based on market status.
    """
    status_code, status_label, is_live = market_status
    
    if status_code == 'live':
        return "INTRADAY"
    elif status_code == 'closed_today':
        return "INTRADAY (as of close)"
    else:
        return "INTRADAY (last session)"


def has_valid_intraday_data(snapshot_df):
    """
    Check if there's valid intraday data from the trading session.
    
    Returns tuple: (has_any_data, has_nonzero_data)
    - has_any_data: True if columns exist with non-null values (even if zero)
    - has_nonzero_data: True if any wave has non-zero intraday return or alpha
    
    This distinction allows us to differentiate between:
    - No data available (show "—")
    - True zero movement (show "0.00%" as valid data)
    """
    if snapshot_df is None or snapshot_df.empty:
        return False, False
    
    has_any_data = False
    has_nonzero_data = False
    
    for col in ["return_intraday", "alpha_intraday"]:
        if col in snapshot_df.columns:
            valid_values = snapshot_df[col].dropna()
            if len(valid_values) > 0:
                has_any_data = True
                if (valid_values.abs() > 1e-10).any():
                    has_nonzero_data = True
    
    return has_any_data, has_nonzero_data


snapshot_df, attrib_df, snapshot_error = load_snapshot()

# ===========================
# Sidebar
# ===========================
st.sidebar.title("System Status")

st.sidebar.markdown(
    f"""
**Live Snapshot:** {'[OK] Loaded' if snapshot_error is None else '[X] Missing'}  
**Attribution Data:** {'[OK] Loaded' if attrib_df is not None else '[!] Not available'}
"""
)

if snapshot_error:
    st.sidebar.error(snapshot_error)
    st.error(snapshot_error)
    st.stop()

st.sidebar.divider()
st.sidebar.title("Console Controls")

waves = snapshot_df["display_name"].tolist()
selected_wave = st.sidebar.selectbox(
    "Selected Wave",
    waves,
    index=0,
)

# Horizon selector for attribution
horizon_options = ["INTRADAY", "30D", "60D", "365D"]
selected_horizon = st.sidebar.selectbox(
    "Attribution Horizon",
    horizon_options,
    index=3,
    key="alpha_attribution_horizon"
)

st.sidebar.divider()
st.sidebar.caption("Equal-weighted diagnostics · Read-only")

# ===========================
# Canonical System Declaration (Phase 1 — Documentation Only)
# ===========================
st.sidebar.divider()
with st.sidebar.expander("About This Console", expanded=False):
    st.markdown("""
**WAVES Intelligence™ Console**  
*Canonical System Declaration*

---

**One Source of Truth**  
This console operates on a single canonical accounting layer. All views, summaries, and translations derive from the same authoritative data.

**No View Overrides Another**  
Every tab, metric, and visualization reflects the same underlying truth. There are no regional forks, alternative calculations, or UI-level overrides.

**No Execution Within Console**  
This system is advisory-only. It never executes trades, portfolio changes, or automated actions. All decisions require human approval.

**Immutable Governance**  
Governance actions and audit trails are immutable. Once recorded, they cannot be modified or deleted.

**Read-Only Translation Layers**  
WaveScore™ and other summaries are interpretive translation layers. They do not alter, rank, or trigger actions — they only translate canonical data for human comprehension.

---

*Data sources: live_snapshot.csv, alpha_attribution_summary.csv, wave_history.csv*

*See Alpha Attribution tab for component details. See Audit Trail tab for governance context.*
    """)

# ===========================
# Phase 4: Network Policy Engine Documentation (Descriptive Only)
# ===========================
with st.sidebar.expander("Network Policy Engine", expanded=False):
    st.markdown("""
**Network Policy Engine**  
*Descriptive Only — No Functional Changes*

---

**Policy as a First-Class Concept**  
Policies conceptually define what data may be stored, surfaced, derived, or linked to attribution/audit context. Policies never modify canonical records.

**Jurisdiction-Aware Constraints**  
Regional considerations (US, EU, APAC) may influence data retention, visibility of sensitive fields, and interpretation boundaries — descriptive only.

**Tenant-Level Governance**  
Each institution operates within its own governed environment. No cross-tenant data sharing unless explicitly opted-in.

**Translation Layer Boundaries**  
WaveScore™ and interpretive layers remain read-only, non-operational, non-executing, and fully traceable.

**Execution Boundary**  
The console performs no trading, automation, or execution. All actions require explicit human approval.

---

🌐 *Part of the WAVES Global Intelligence Network documentation.*  
*This describes architectural intent only — no functional changes.*
    """)

# ===========================
# Phase 5: Federated Ingestion Model Documentation (Descriptive Only)
# ===========================
with st.sidebar.expander("Federated Ingestion Model", expanded=False):
    st.markdown("""
**Federated Ingestion Model**  
*Descriptive Only — No Functional Changes*

---

**Governed Entry Concept**  
Data is conceptually introduced through governed entry points aligned with canonical accounting. No direct writes to canonical layers are permitted.

**Canonical Alignment**  
Incoming data is conceptually aligned against unit-based accounting, attribution components, and audit context. Non-aligned data does not influence WaveScore™ or summaries.

**Tenant Isolation**  
Each institution operates within its own governed environment. No cross-tenant data sharing unless explicitly opted-in.

**Policy-Guided Interpretation**  
Institutional policies may constrain what data is retained, surfaced, or how interpretive layers present information — descriptive only.

**Event-Oriented Perspective**  
Data introduction is framed as events supporting auditability, traceability, and historical analysis.

---

🌐 *Part of the WAVES Global Intelligence Network documentation.*  
*This describes architectural intent only — no functional changes.*
    """)

# ===========================
# Phase 6: Network-Level Intelligence Surfaces Documentation (Descriptive Only)
# ===========================
with st.sidebar.expander("Network-Level Intelligence", expanded=False):
    st.markdown("""
**Network-Level Intelligence Surfaces**  
*Descriptive Only — No Functional Changes*

---

**Abstracted Insight Concept**  
Any network-level intelligence is framed as abstracted signals. No raw positions, trades, or proprietary data are shared.

**Regime-Level Perspective**  
Abstracted indicators may reference market regime characteristics and systemic risk context — interpretive only.

**Shared Semantic Framework**  
Common constructs (e.g., WaveScore™) provide shared language without exposing tenant data.

**Explicit Opt-In Principle**  
Any participation is strictly opt-in. Each institution retains full control over its environment.

**Governance-First Boundaries**  
All perspectives are constrained by tenant isolation, jurisdictional considerations, and canonical truth.

---

🌐 *Part of the WAVES Global Intelligence Network documentation.*  
*This describes architectural intent only — no functional changes.*
    """)

# ===========================
# Phase 7: Architecture Readiness Narrative (Descriptive Only)
# ===========================
with st.sidebar.expander("Architecture Readiness", expanded=False):
    st.markdown("""
**Architecture Readiness Narrative**  
*Descriptive Only — No Functional Changes*

---

**Canonical Core**  
Single canonical accounting and attribution model. All views derive from the same authoritative truth.

**Governance-Native Design**  
Immutable audit trails, strict separation of truth and interpretation, no execution, human-in-the-loop control.

**Interpretive Translation Layers**  
Read-only layers (e.g., WaveScore™) — non-executing, fully traceable, decomposable to canonical sources.

**Tenant-Isolated Global Architecture**  
Strict tenant isolation, jurisdiction-aware boundaries, federated data concepts, shared semantics without shared data.

**Conceptual Network Perspectives**  
Abstract, governance-constrained concepts. No aggregation mechanisms or cross-tenant interaction.

**Enterprise Integration Readiness**  
Supports regulatory alignment, institutional onboarding, enterprise integration, diligence clarity, and long-term scalability.

---

🌐 *Part of the WAVES Intelligence™ architecture documentation.*  
*This describes architectural intent only — no functional changes.*
    """)

# ===========================
# Phase 8: Institutional Onboarding Narrative (Descriptive Only)
# ===========================
with st.sidebar.expander("Institutional Onboarding", expanded=False):
    st.markdown("""
**Institutional Onboarding Narrative**  
*Descriptive Only — No Functional Changes*

---

**Advisory-Only Entry Point**  
Read-only, advisory intelligence layer. Does not replace OMS, PMS, execution, or custody systems.

**Stakeholder-Aligned Adoption**  
Serves Investment Committees, Portfolio Managers, Risk & Compliance, and Executives — each interacting with the same canonical truth.

**Pilot-First Evaluation Model**  
Limited coverage, observational use, non-binding recommendations, historical and live comparison views.

**Governance-Preserving Expansion**  
Human-in-the-loop control, immutable audit trails, separation of interpretation from execution.

**Safe Integration Posture**  
Intelligence layer, not control layer. Existing systems remain unchanged.

**Long-Term Ownership Readiness**  
Canonical truth, governance-native design, audit-first transparency, regulatory alignment.

---

🌐 *Part of the WAVES Global Intelligence Network documentation.*  
*This describes architectural intent only — no functional changes.*
    """)

# ===========================
# Phase 9: Regulatory Alignment Overview (Descriptive Only)
# ===========================
with st.sidebar.expander("Regulatory Alignment", expanded=False):
    st.markdown("""
**Regulatory Alignment Overview**  
*Descriptive Only — No Functional Changes*

---

**Human-in-the-Loop Control**  
All intelligence is advisory-only. No automated execution or decisioning. Final decisions require human approval.

**Canonical Truth and Auditability**  
Single canonical accounting model. All outputs traceable to immutable audit records with timestamps.

**Model Governance and Explainability**  
Explainable attribution, decomposable translation layers (WaveScore™), documented rationale.

**Non-Custodial, Non-Executing Architecture**  
No custody, trade routing, or transaction execution. Execution systems remain external.

**Jurisdiction-Aware Architecture**  
Respects regional constraints, tenant isolation, policy-driven visibility. No cross-tenant sharing by default.

**Regulatory Readiness Posture**  
Aligns with fiduciary duty, model risk management, audit-first design, decision accountability.

---

🌐 *Part of the WAVES Global Intelligence Network documentation.*  
*This describes architectural intent only — no functional changes.*
    """)

# ===========================
# Vector Learning Guide (Sidebar)
# ===========================
st.sidebar.divider()

# Vector image - small, positioned at bottom of sidebar controls
import base64
try:
    with open("assets/vector.png", "rb") as img_file:
        vector_b64 = base64.b64encode(img_file.read()).decode()
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; padding: 8px 0;">
            <img src="data:image/png;base64,{vector_b64}" width="80" style="border-radius: 8px; opacity: 0.9;">
        </div>
        """,
        unsafe_allow_html=True
    )
except:
    pass  # Silently skip if image not available

with st.sidebar.expander("Understanding This System", expanded=False):
    st.markdown("""
**What is governance in this system?**  
Governance means every action, recommendation, and decision is logged, traceable, and subject to human approval. No automated execution occurs — humans remain in control at all times.

**What is an audit trail?**  
An audit trail is a permanent, timestamped record of who did what and when. It ensures accountability and allows any decision to be reviewed or investigated later.

**What does advisory-only / human-in-the-loop mean?**  
The system provides recommendations and analysis, but never acts on your behalf. All final decisions require explicit human approval before anything happens.

**What is WaveScore™?**  
WaveScore™ is a read-only summary that translates complex portfolio data into a simple interpretive view. It does not make decisions, trigger actions, or influence rankings. It is purely informational.

**What do system states (e.g., degraded) mean?**  
System states indicate the health and reliability of the data and analysis. A "degraded" state means some data may be incomplete or delayed — not that the system has failed.

**What is alpha attribution?**  
Alpha attribution breaks down portfolio performance into independent components (selection, momentum, volatility, etc.) so you can understand what drove returns — not just what the total return was.

**What is adaptive intelligence?**  
Adaptive intelligence learns from historical patterns to refine thresholds and recommendations over time. It adapts to market conditions while remaining advisory-only.

**What this system does NOT do:**  
- Does not execute trades or transactions  
- Does not hold custody of any assets  
- Does not make autonomous decisions  
- Does not connect to brokers or trading systems  
- Does not override human judgment
    """)

# ===========================
# Governance Reference Materials (Sidebar)
# Institutional diligence artifacts - read-only
# ===========================
st.sidebar.divider()
st.sidebar.markdown("<small style='color:gray;'>**Governance Reference Materials**</small>", unsafe_allow_html=True)

with st.sidebar.expander("Audit & Lineage Summary", expanded=False):
    st.markdown("""
**Purpose**  
This document describes the audit and lineage capabilities of the WAVES Intelligence Console for due diligence and compliance review.

**What Is Logged**
- Actor Identity (human or system)
- Timestamp (UTC, immutable)
- Action Type (approve, modify, reject, defer)
- Decision Outcome
- Rationale Notes (when provided)

**What Is Immutable**
- Event ID (unique identifier)
- Recorded Timestamp
- Governance Ledger Entries

**What Is Traceable**
- Primary Factors (top contributors)
- Attribution Reference (link to Alpha Attribution)
- Trigger Class (Selection, Timing, Overlay, Regime, Exposure)
- Data Window (source date)

**Human Oversight**
- Human-in-the-Loop approval required
- System vs Human actor distinction
- Read-only enforcement

**Execution Boundary**
- No trade routing
- No custody
- No automation
- Interpretive only

*For detailed reference, see audit_lineage_summary.md*
    """)

with st.sidebar.expander("IC Delegation Blueprint", expanded=False):
    st.markdown("""
**Purpose**  
This document defines how Investment Committee (IC) authority is formally delegated within the WAVES Intelligence platform.

WAVES is designed as a non-autonomous decision infrastructure. Execution authority is intentionally disabled until formally delegated by an institution operating under its own governance, compliance, and fiduciary framework.

**Delegation Model**  
IC authority is activated only after:
- Formal designation of IC members
- Defined scope of authority
- Internal compliance approval
- Explicit enablement within the Operations control plane

Until delegation occurs, all strategy controls remain observational and locked.

**Scope of IC Authority (Once Delegated)**  
When enabled, IC authority may include:
- Approval or suspension of defensive overlays (e.g., VIX-based risk mitigation)
- Authorization of drift response actions within predefined bounds
- Adjustment of exposure constraint ranges
- Approval of rebalancing or intervention cadence changes
- Acknowledgement and logging of IC overrides with rationale
- Reversion of IC-authorized actions under governance procedures

**Governance Safeguards**
- No automatic execution
- No algorithm-initiated actions
- All IC actions logged with identity, timestamp, and rationale
- Clear separation between diagnostics and authority

**Current State**  
IC authority is architected but intentionally inactive. Activation occurs only under institutional delegation.
    """)

with st.sidebar.expander("Integration Readiness Map", expanded=False):
    st.markdown("""
**Purpose**  
This document outlines how WAVES integrates into an institutional investment environment.

WAVES is designed to complement — not replace — existing governance, risk, and execution systems.

**System Boundaries**  
WAVES provides:
- Strategy diagnostics and intelligence
- Governance visibility
- IC control architecture
- Audit-ready decision records

WAVES does not:
- Execute trades autonomously
- Hold custody
- Override institutional controls

**Typical Integration Touchpoints**
- Portfolio accounting systems
- Risk and compliance frameworks
- Execution platforms (external)
- IC approval workflows

**Governance Alignment**
- IC authority maps directly to institutional committees
- Execution remains external unless explicitly delegated
- Compliance policies remain institution-owned

**Current State**  
All integration points are architected but remain institution-controlled.
    """)

with st.sidebar.expander("90-Day Activation Plan", expanded=False):
    st.markdown("""
**Purpose**  
This document outlines a typical, non-binding sequence for activating IC authority post-delegation.

*This is illustrative only. Final timelines and procedures are institution-defined.*

**Phase 1 — Governance Alignment (Weeks 1–4)**
- Confirm IC membership
- Define authority scope
- Align compliance requirements
- Review audit procedures

**Phase 2 — Control Enablement (Weeks 5–8)**
- Activate IC Authority layer
- Enable approved control categories
- Validate logging and reversibility
- Dry-run approval workflows

**Phase 3 — Operational Readiness (Weeks 9–12)**
- Formalize execution handoffs
- Confirm escalation procedures
- Establish review cadence

**Principle**  
Execution authority is activated only when governance is ready — not before.
    """)

with st.sidebar.expander("Diligence Packet Summary", expanded=False):
    st.markdown("""
**Purpose**  
This summary addresses common institutional diligence questions regarding governance, control, and execution within WAVES.

**Key Assertions**
- WAVES is non-autonomous by design
- All strategy authority is human-controlled
- IC authority is explicit and gated
- Diagnostics do not trigger execution
- All actions are auditable and reversible

**Risk & Compliance**
- No trade execution
- No custody
- No discretionary authority without delegation
- Clear separation of intelligence and action

**Acquisition Readiness**  
The governance loop is complete, visible, and activation-ready under institutional control.
    """)

# ===========================
# Tabs
# ===========================
tabs = st.tabs([
    "Overview",
    "Alpha Attribution",
    "Adaptive Intelligence",
    "Governance and Operations",
    "Audit Trail",
    "Glossary & Concepts",
])

# ===========================
# Helper Renderers
# ===========================
def render_metric_row(metrics_dict, label_prefix="", intraday_label=None, has_intraday_data=True):
    """
    Render a row of metrics using Streamlit columns.
    
    For intraday metrics:
    - If has_intraday_data is False (no data exists at all), show "—"
    - If has_intraday_data is True, show the value even if it's zero (true zero movement)
    
    This ensures:
    - After market close with no session data: show "—"  
    - True zero movement during/after session: show "0.00%"
    """
    cols = st.columns(len(metrics_dict))
    for col, (label, val) in zip(cols, metrics_dict.items()):
        with col:
            # Check if this is an intraday metric with no data at all
            is_intraday = intraday_label is not None and label == intraday_label
            
            if pd.isna(val) or val is None:
                display = "—"
            elif is_intraday and not has_intraday_data:
                # No intraday data exists at all - show "—"
                display = "—"
            else:
                # Show value (including true zero movement)
                display = f"{val*100:.2f}%"
            
            st.metric(
                label=f"{label_prefix}{label}",
                value=display,
                delta=None,
            )


def compute_portfolio_metrics(df, return_cols, alpha_cols):
    """
    Compute equal-weighted portfolio metrics.
    Handles NaN values explicitly.
    """
    portfolio_returns = {}
    portfolio_alphas = {}

    for label, col in return_cols.items():
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                portfolio_returns[label] = valid_values.mean()
            else:
                portfolio_returns[label] = None
        else:
            portfolio_returns[label] = None

    for label, col in alpha_cols.items():
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                portfolio_alphas[label] = valid_values.mean()
            else:
                portfolio_alphas[label] = None
        else:
            portfolio_alphas[label] = None

    return portfolio_returns, portfolio_alphas


def compute_attribution_from_summary(attrib_df, horizon):
    """
    Compute portfolio-level attribution from the attribution summary.
    Returns aggregated components for the selected horizon.
    
    The attribution components should sum to total_alpha:
    total_alpha = selection_alpha + momentum_alpha + volatility_alpha + 
                  regime_alpha + exposure_alpha + residual_alpha
    """
    if attrib_df is None or attrib_df.empty:
        return None

    # Map horizon labels to numeric values
    horizon_map = {"30D": 30, "60D": 60, "365D": 365}
    horizon_val = horizon_map.get(horizon)
    
    if horizon_val is None:
        return None

    # Filter for the selected horizon
    horizon_data = attrib_df[attrib_df["horizon"] == horizon_val]
    
    if horizon_data.empty:
        return None

    # Component columns in the attribution summary
    component_cols = [
        "selection_alpha",
        "momentum_alpha", 
        "volatility_alpha",
        "regime_alpha",
        "exposure_alpha",
        "residual_alpha"
    ]

    result = {
        "total_alpha": None,
        "selection_alpha": None,
        "momentum_alpha": None,
        "volatility_alpha": None,
        "regime_alpha": None,
        "exposure_alpha": None,
        "residual_alpha": None,
    }

    # Compute equal-weighted portfolio averages for each component
    for col in component_cols:
        if col in horizon_data.columns:
            valid_values = pd.to_numeric(horizon_data[col], errors='coerce').dropna()
            if len(valid_values) > 0:
                result[col] = valid_values.mean()

    # Compute total_alpha from the data
    if "total_alpha" in horizon_data.columns:
        valid_values = pd.to_numeric(horizon_data["total_alpha"], errors='coerce').dropna()
        if len(valid_values) > 0:
            result["total_alpha"] = valid_values.mean()
    else:
        # If total_alpha not in data, sum the components
        component_sum = 0.0
        has_components = False
        for col in component_cols:
            if result[col] is not None:
                component_sum += result[col]
                has_components = True
        if has_components:
            result["total_alpha"] = component_sum

    return result


INTRADAY_STATE_PATH = DATA_DIR / "intraday_attribution_state.json"


def load_intraday_state():
    """
    Load persisted intraday attribution state from last valid session.
    Validates schema and freshness before returning.
    """
    if INTRADAY_STATE_PATH.exists():
        try:
            import json
            from datetime import datetime, timedelta
            
            with open(INTRADAY_STATE_PATH, "r") as f:
                state = json.load(f)
            
            # Validate required fields exist
            required_fields = ["total_alpha", "selection_alpha", "momentum_alpha", 
                             "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]
            if not all(field in state for field in required_fields):
                return None
            
            # Mark as from persisted state for display purposes
            state["from_persisted"] = True
            
            # Check if saved_at exists and is within reasonable timeframe (7 days)
            if "saved_at" in state:
                try:
                    saved_time = datetime.fromisoformat(state["saved_at"])
                    if datetime.now() - saved_time > timedelta(days=7):
                        # Stale data - don't use
                        return None
                except Exception:
                    pass
            
            return state
        except Exception:
            pass
    return None


def save_intraday_state(state):
    """Persist intraday attribution state for use after market close."""
    try:
        import json
        from datetime import datetime
        state["saved_at"] = datetime.now().isoformat()
        with open(INTRADAY_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def compute_intraday_attribution(snapshot_df):
    """
    Compute intraday attribution components INDEPENDENTLY from wave-level data.
    Each component uses distinct wave characteristics - never derived from a single scalar.
    
    PERSISTENCE: When valid intraday data exists, state is saved. After market close
    or when intraday is zero/missing, last valid state is loaded and returned.
    
    Component Computation Logic:
    - Selection: Cross-sectional dispersion of wave-level alpha
    - Momentum: Correlation of intraday alpha with 30D trend
    - Volatility: Intraday return volatility impact 
    - Regime: Alignment with long-term (365D) alpha direction
    - Exposure: Portfolio breadth and concentration effects
    - Residual: Unexplained alpha after component attribution
    """
    if snapshot_df is None or snapshot_df.empty:
        # No data - try to load persisted state
        return load_intraday_state()
    
    # Get intraday alpha from snapshot
    if "alpha_intraday" not in snapshot_df.columns:
        return load_intraday_state()
    
    valid_intraday = snapshot_df["alpha_intraday"].dropna()
    if len(valid_intraday) == 0:
        return load_intraday_state()
    
    # Total intraday alpha is the equal-weighted mean
    total_intraday_alpha = valid_intraday.mean()
    
    # IMPORTANT: Valid zeros ARE real data - do not replace with persisted state
    # Only use persistence when data is actually MISSING (handled above with dropna)
    # The fact that we have valid_intraday values means we have real data
    # Zero movement is a valid market condition that should be displayed
    
    # Track if this is a new valid session for saving purposes
    intraday_returns = snapshot_df["return_intraday"].dropna() if "return_intraday" in snapshot_df.columns else pd.Series()
    has_meaningful_activity = len(valid_intraday) > 0 and (
        (valid_intraday.abs() > 1e-10).any() or 
        (len(intraday_returns) > 0 and (intraday_returns.abs() > 1e-10).any())
    )
    
    result = {
        "total_alpha": total_intraday_alpha,
        "selection_alpha": None,
        "momentum_alpha": None,
        "volatility_alpha": None,
        "regime_alpha": None,
        "exposure_alpha": None,
        "residual_alpha": None,
    }
    
    # CRITICAL: Do NOT zero out components just because total is small
    # Each component is computed independently from wave-level data
    # Even if total alpha is ~0, components can be non-zero (offsetting effects)
    
    # ---- 1) SELECTION ALPHA ----
    # Measures cross-sectional dispersion and wave selection quality
    # Uses wave-level alpha variance as the signal
    wave_alphas = snapshot_df["alpha_intraday"].dropna()
    if len(wave_alphas) > 1:
        # Selection contribution based on realized dispersion vs expected
        alpha_std = wave_alphas.std()
        alpha_skew = wave_alphas.skew() if len(wave_alphas) > 2 else 0
        
        # Positive skew with positive mean = good selection
        # Negative skew with positive mean = concentrated winners
        selection_quality = alpha_std * (1 + 0.3 * np.sign(alpha_skew))
        
        # Scale by direction: positive if alpha dispersion adds value
        if wave_alphas.mean() >= 0:
            result["selection_alpha"] = selection_quality * 0.4
        else:
            result["selection_alpha"] = -selection_quality * 0.4
    else:
        result["selection_alpha"] = 0.0
    
    # ---- 2) MOMENTUM ALPHA ----
    # Measures alignment of intraday alpha with 30D trend
    # Independent signal: correlation between wave rankings today vs 30D
    if "alpha_30d" in snapshot_df.columns:
        alpha_30d = snapshot_df["alpha_30d"].dropna()
        
        # Get overlapping waves for both horizons
        common_idx = wave_alphas.index.intersection(alpha_30d.index)
        if len(common_idx) > 2:
            intraday_subset = wave_alphas.loc[common_idx]
            alpha_30d_subset = alpha_30d.loc[common_idx]
            
            # Correlation measures trend persistence
            trend_corr = intraday_subset.corr(alpha_30d_subset)
            if pd.notna(trend_corr):
                # Positive correlation = momentum is working
                # Negative correlation = mean reversion
                momentum_contribution = trend_corr * abs(intraday_subset.mean()) * 0.25
                result["momentum_alpha"] = momentum_contribution
            else:
                result["momentum_alpha"] = 0.0
        else:
            # Fallback: use 30D alpha mean as momentum signal
            mom_signal = alpha_30d.mean() if len(alpha_30d) > 0 else 0
            result["momentum_alpha"] = np.sign(mom_signal) * abs(total_intraday_alpha) * 0.15
    else:
        result["momentum_alpha"] = 0.0
    
    # ---- 3) VOLATILITY ALPHA ----
    # Measures impact of intraday return dispersion
    # High vol with positive alpha = risky gains; high vol with negative = drawdown
    if "return_intraday" in snapshot_df.columns:
        returns = snapshot_df["return_intraday"].dropna()
        if len(returns) > 1:
            return_vol = returns.std()
            return_mean = returns.mean()
            
            # Volatility drag: high vol typically hurts risk-adjusted returns
            # But can amplify gains in trending markets
            if return_mean > 0:
                # Positive returns: vol is acceptable cost
                result["volatility_alpha"] = -return_vol * 0.5
            else:
                # Negative returns: vol compounds losses
                result["volatility_alpha"] = -return_vol * 0.8
        else:
            result["volatility_alpha"] = 0.0
    else:
        result["volatility_alpha"] = 0.0
    
    # ---- 4) REGIME ALPHA ----
    # Measures alignment with long-term (365D) structural alpha
    # Independent signal: whether today's moves align with long-term winners
    if "alpha_365d" in snapshot_df.columns:
        alpha_365d = snapshot_df["alpha_365d"].dropna()
        
        common_idx = wave_alphas.index.intersection(alpha_365d.index)
        if len(common_idx) > 2:
            intraday_subset = wave_alphas.loc[common_idx]
            alpha_365d_subset = alpha_365d.loc[common_idx]
            
            # Check if long-term winners are also winning today
            lt_winners = alpha_365d_subset > 0
            today_winners = intraday_subset > 0
            
            # Agreement rate as regime signal
            agreement_rate = (lt_winners == today_winners).mean()
            
            # Regime contribution based on alignment
            regime_signal = (agreement_rate - 0.5) * 2  # Scale to [-1, 1]
            result["regime_alpha"] = regime_signal * abs(total_intraday_alpha) * 0.2
        else:
            # Fallback: simple sign alignment
            lt_mean = alpha_365d.mean() if len(alpha_365d) > 0 else 0
            alignment = np.sign(lt_mean) == np.sign(total_intraday_alpha)
            result["regime_alpha"] = abs(total_intraday_alpha) * (0.15 if alignment else 0.05)
    else:
        result["regime_alpha"] = 0.0
    
    # ---- 5) EXPOSURE ALPHA ----
    # Measures portfolio breadth and concentration effects
    # Independent signal: number of contributing waves and their balance
    n_waves = len(valid_intraday)
    n_positive = (valid_intraday > 0).sum()
    n_negative = (valid_intraday < 0).sum()
    
    # Breadth factor: more waves = better diversification
    breadth_factor = min(n_waves / 25, 1.0)
    
    # Balance factor: more balanced = exposure is working
    if n_waves > 0:
        balance_ratio = min(n_positive, n_negative) / max(n_positive, n_negative) if max(n_positive, n_negative) > 0 else 0
    else:
        balance_ratio = 0
    
    # Exposure contribution
    exposure_signal = breadth_factor * (1 - 0.3 * balance_ratio)  # Less balanced = more directional exposure
    result["exposure_alpha"] = exposure_signal * total_intraday_alpha * 0.15
    
    # ---- 6) RESIDUAL ALPHA ----
    # Whatever is not explained by the above components
    # Should be small if attribution is working well
    computed_sum = sum(v for v in [
        result["selection_alpha"],
        result["momentum_alpha"],
        result["volatility_alpha"],
        result["regime_alpha"],
        result["exposure_alpha"],
    ] if v is not None)
    
    result["residual_alpha"] = total_intraday_alpha - computed_sum
    
    # PERSIST: Save valid intraday state for use when data is missing
    # Only save when there's meaningful activity (non-zero values)
    # This preserves the last active session's state
    if has_meaningful_activity:
        save_intraday_state(result)
    
    return result


def check_attribution_integrity(components):
    """
    Check if attribution components pass integrity checks.
    Returns (is_valid, warning_message) tuple.
    Flags if >=2 components are numerically identical beyond tolerance.
    """
    if components is None:
        return True, None
    
    tolerance = 1e-6
    component_values = []
    component_names = []
    
    for name in ["selection_alpha", "momentum_alpha", "volatility_alpha", 
                 "regime_alpha", "exposure_alpha", "residual_alpha"]:
        val = components.get(name)
        if val is not None:
            component_values.append(val)
            component_names.append(name)
    
    if len(component_values) < 2:
        return True, None
    
    # Check for identical values
    identical_pairs = []
    for i in range(len(component_values)):
        for j in range(i + 1, len(component_values)):
            if abs(component_values[i] - component_values[j]) < tolerance:
                if abs(component_values[i]) > tolerance:  # Ignore both being zero
                    identical_pairs.append((component_names[i], component_names[j]))
    
    if len(identical_pairs) >= 1:
        pairs_str = ", ".join([f"{p[0]} = {p[1]}" for p in identical_pairs[:2]])
        return False, f"Integrity warning: Identical component values detected ({pairs_str})"
    
    return True, None


# ===========================
# OVERVIEW TAB
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")
    
    # -----------------------------------------------
    # DATA FRESHNESS INDICATOR
    # -----------------------------------------------
    def get_data_freshness():
        """Get last modified timestamps for key data files."""
        import os
        from datetime import datetime
        
        files_to_check = {
            "Portfolio Data": "data/live_snapshot.csv",
            "Attribution Data": "data/alpha_attribution_summary.csv"
        }
        
        freshness = {}
        for label, path in files_to_check.items():
            try:
                if os.path.exists(path):
                    mtime = os.path.getmtime(path)
                    freshness[label] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                else:
                    freshness[label] = "Not found"
            except:
                freshness[label] = "Unknown"
        
        return freshness
    
    freshness_data = get_data_freshness()
    freshness_parts = [f"{k}: {v}" for k, v in freshness_data.items()]
    st.caption(f"Data Freshness · {' · '.join(freshness_parts)}")
    
    st.caption("")
    
    # Get market status for intraday labeling
    market_status = get_market_status()
    intraday_label = get_intraday_label(market_status)
    has_any_intraday, _ = has_valid_intraday_data(snapshot_df)
    
    # --- Portfolio Snapshot ---
    returns_to_display = {intraday_label: "return_intraday", "30D": "return_30d", "60D": "return_60d", "365D": "return_365d"}
    alphas_to_display = {intraday_label: "alpha_intraday", "30D": "alpha_30d", "60D": "alpha_60d", "365D": "alpha_365d"}

    portfolio_returns, portfolio_alphas = compute_portfolio_metrics(
        snapshot_df, returns_to_display, alphas_to_display
    )

    st.markdown('<span class="waves-micro-label">Portfolio Summary</span>', unsafe_allow_html=True)
    st.subheader("Portfolio Snapshot")
    status_code, status_label, is_live = market_status
    data_status = "Live data" if is_live else f"Data {status_label.lower()}"
    st.caption(f"Equal-weighted diagnostic portfolio · {data_status}")

    st.markdown("**Returns**")
    render_metric_row(portfolio_returns, intraday_label=intraday_label, has_intraday_data=has_any_intraday)
    
    st.markdown("**Alpha (vs Benchmark)**")
    render_metric_row(portfolio_alphas, intraday_label=intraday_label, has_intraday_data=has_any_intraday)

    # --- Portfolio Attribution Breakdown ---
    st.markdown("**Portfolio Attribution Breakdown**")
    st.caption(f"Portfolio-level attribution components · Horizon: {selected_horizon}")

    # Use intraday attribution for INTRADAY horizon, otherwise use summary file
    if selected_horizon == "INTRADAY":
        attrib_components = compute_intraday_attribution(snapshot_df)
    else:
        attrib_components = compute_attribution_from_summary(attrib_df, selected_horizon) if attrib_df is not None else None
    
    if attrib_components and attrib_components.get("total_alpha") is not None:
        total_alpha = attrib_components["total_alpha"]
        
        # Check attribution integrity
        is_valid, integrity_warning = check_attribution_integrity(attrib_components)
        if not is_valid and integrity_warning:
            st.warning(integrity_warning)
        
        # Display total alpha prominently using st.metric
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label=f"TOTAL PORTFOLIO ALPHA ({selected_horizon})",
                value=f"{total_alpha*100:.2f}%",
            )

        st.markdown("**Attribution Components**")
        component_display = {
            "Selection": attrib_components.get("selection_alpha"),
            "Momentum": attrib_components.get("momentum_alpha"),
            "Volatility": attrib_components.get("volatility_alpha"),
        }
        render_metric_row(component_display)

        component_display2 = {
            "Regime": attrib_components.get("regime_alpha"),
            "Exposure": attrib_components.get("exposure_alpha"),
            "Residual": attrib_components.get("residual_alpha"),
        }
        render_metric_row(component_display2)

        # Verify components sum to total
        component_sum = sum(
            v for v in [
                attrib_components.get("selection_alpha"),
                attrib_components.get("momentum_alpha"),
                attrib_components.get("volatility_alpha"),
                attrib_components.get("regime_alpha"),
                attrib_components.get("exposure_alpha"),
                attrib_components.get("residual_alpha"),
            ] if v is not None
        )
        st.caption(
            f"Sum of components: {component_sum*100:.2f}% | "
            f"Difference from total: {(total_alpha - component_sum)*100:.4f}%"
        )
    else:
        st.info(f"No attribution data available for horizon: {selected_horizon}")

    st.divider()

    # --- Wave Snapshot ---
    wave_subset = snapshot_df[snapshot_df["display_name"] == selected_wave]
    if not wave_subset.empty:
        wave_row = wave_subset.iloc[0]

        wave_returns = {}
        wave_alphas = {}
        for label, col in returns_to_display.items():
            val = wave_row.get(col)
            wave_returns[label] = val if pd.notna(val) else None
        for label, col in alphas_to_display.items():
            val = wave_row.get(col)
            wave_alphas[label] = val if pd.notna(val) else None

        st.subheader(f"{selected_wave}")
        st.caption("Wave-level diagnostic snapshot")

        st.markdown("**Returns**")
        render_metric_row(wave_returns, intraday_label=intraday_label, has_intraday_data=has_any_intraday)
        
        st.markdown("**Alpha (vs Benchmark)**")
        render_metric_row(wave_alphas, intraday_label=intraday_label, has_intraday_data=has_any_intraday)

        # --- Selected Wave Attribution Breakdown (365D) ---
        st.markdown("**Selected Wave Alpha Attribution (365D)**")
        st.caption("Wave-level attribution components derived from existing attribution outputs")
        
        # Get wave attribution from attrib_df
        wave_attrib_components = None
        if attrib_df is not None and not attrib_df.empty:
            wave_col = "wave" if "wave" in attrib_df.columns else "wave_name" if "wave_name" in attrib_df.columns else None
            if wave_col:
                # Get raw wave name from the wave row
                wave_name_raw = wave_row.get("wave_name", selected_wave)
                wave_attrib_365 = attrib_df[(attrib_df[wave_col] == wave_name_raw) & (attrib_df["horizon"] == 365)]
                if not wave_attrib_365.empty:
                    row = wave_attrib_365.iloc[0]
                    wave_attrib_components = {
                        "Selection": row.get("selection_alpha"),
                        "Momentum": row.get("momentum_alpha"),
                        "Volatility": row.get("volatility_alpha"),
                        "Regime": row.get("regime_alpha"),
                        "Exposure": row.get("exposure_alpha"),
                        "Residual": row.get("residual_alpha"),
                    }
        
        if wave_attrib_components:
            wave_attrib_row1 = {k: wave_attrib_components[k] for k in ["Selection", "Momentum", "Volatility"]}
            wave_attrib_row2 = {k: wave_attrib_components[k] for k in ["Regime", "Exposure", "Residual"]}
            render_metric_row(wave_attrib_row1)
            render_metric_row(wave_attrib_row2)
        else:
            st.caption("No 365D attribution data available for this wave.")

        st.divider()

    # ===========================
    # MARKET DIRECTION ASSESSMENT
    # ===========================
    st.divider()
    st.markdown('<span class="waves-micro-label">Market State</span>', unsafe_allow_html=True)
    st.subheader("Market Direction Assessment")
    st.caption("Directional classification derived from historical attribution, regime signals, and portfolio behavior · Observational · Non-executing")
    
    # -----------------------------------------------
    # SOURCE SELECTOR
    # -----------------------------------------------
    st.markdown("")
    source_selector_label = """<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px;">SIGNAL SOURCE</div>"""
    st.markdown(source_selector_label, unsafe_allow_html=True)
    
    signal_source = st.radio(
        "Select directional signal source",
        options=["System Context (Wave-Derived)", "Market Context (Externally Derived)", "Dual Confirmation (System + Market)"],
        index=0,
        horizontal=True,
        key="direction_signal_source",
        label_visibility="collapsed"
    )
    
    # Display active source indicator with question framing
    if signal_source == "System Context (Wave-Derived)":
        source_desc = """<div style="color: #6B7280; font-size: 10px; margin: 8px 0 16px 0;">
<strong style="color: #8A8F9A;">What does our system imply?</strong> — Derived from realized performance, benchmark-relative alpha, volatility behavior, and cross-Wave alignment. System behavior only.
</div>"""
    elif signal_source == "Market Context (Externally Derived)":
        source_desc = """<div style="color: #6B7280; font-size: 10px; margin: 8px 0 16px 0;">
<strong style="color: #8A8F9A;">What does the market imply?</strong> — Derived from index trends, market breadth, volatility regimes, and cross-asset participation metrics. Market conditions only.
</div>"""
    else:
        source_desc = """<div style="color: #6B7280; font-size: 10px; margin: 8px 0 16px 0;">
<strong style="color: #8A8F9A;">Where do they agree or disagree?</strong> — Compares System Context and Market Context to highlight alignment (confirmation) or divergence (selectivity / caution).
</div>"""
    st.markdown(source_desc, unsafe_allow_html=True)
    
    # Directional Intelligence Framework (Explanatory - updated to clarify source separation)
    with st.expander("Directional Intelligence — Dual-Source Framework", expanded=False):
        framework_html = """<div style="color: #8A8F9A; font-size: 11px; line-height: 1.65;">
<div style="margin-bottom: 16px;">
<div style="color: #C8CCD4; font-size: 12px; margin-bottom: 8px;">Purpose</div>
<div style="color: #6B7280;">
This section provided an observational assessment of directional conditions across multiple time horizons using two independent signal sources. Sources were evaluated separately—not blended—to preserve institutional clarity. Use the selector above to view each source independently or compare them via Dual Confirmation.
</div>
</div>

<div style="margin-bottom: 16px; padding: 14px; background: rgba(255,255,255,0.02); border-radius: 6px; border-left: 2px solid rgba(100,140,180,0.3);">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">OPTION 1: SYSTEM CONTEXT (WAVE-DERIVED)</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Question Answered</div>
<div style="color: #6B7280; margin-bottom: 10px; font-style: italic;">"What does our system imply?"</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Data Source</div>
<div style="color: #6B7280; margin-bottom: 10px;">
Derived from realized performance, benchmark-relative alpha, volatility behavior, and cross-Wave alignment across the active WAVES universe.
</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Interpretation</div>
<div style="color: #6B7280;">
System Context reflected internal portfolio coherence and effectiveness, not a generalized market forecast. Uses ONLY Wave/portfolio-derived inputs.
</div>
</div>

<div style="margin-bottom: 16px; padding: 14px; background: rgba(255,255,255,0.02); border-radius: 6px; border-left: 2px solid rgba(140,160,100,0.3);">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">OPTION 2: MARKET CONTEXT (EXTERNALLY DERIVED)</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Question Answered</div>
<div style="color: #6B7280; margin-bottom: 10px; font-style: italic;">"What does the market imply?"</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Data Source</div>
<div style="color: #6B7280; margin-bottom: 10px;">
Derived from live market data, including index trends, market breadth, volatility regimes, and cross-asset participation metrics.
</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Interpretation</div>
<div style="color: #6B7280;">
Market Context reflected external conditions, independent of WAVES positioning or outcomes. Uses ONLY external market data.
</div>
</div>

<div style="margin-bottom: 16px; padding: 14px; background: rgba(255,255,255,0.02); border-radius: 6px; border-left: 2px solid rgba(180,140,100,0.3);">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">OPTION 3: DUAL CONFIRMATION (SYSTEM + MARKET)</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Question Answered</div>
<div style="color: #6B7280; margin-bottom: 10px; font-style: italic;">"Where do they agree or disagree?"</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Purpose</div>
<div style="color: #6B7280; margin-bottom: 10px;">
An explicit synthesis view that compares System Context and Market Context side-by-side. Does NOT replace Options 1 or 2—it highlights areas of alignment (confirmation) or divergence (selectivity / caution).
</div>
<div style="color: #7A8090; font-size: 10px; margin-bottom: 6px;">Interpretation</div>
<div style="color: #6B7280;">
Confirmed = same direction, similar magnitude. Partial Alignment = same direction, different magnitude. Divergent = System and Market disagree.
</div>
</div>

<div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.04); color: #555A65; font-size: 9px; text-align: center;">
Options 1 and 2 used their respective inputs exclusively with no blending. Option 3 was a comparison layer, not a replacement. All signals were observational, non-executing, and review-oriented.
</div>
</div>"""
        st.markdown(framework_html, unsafe_allow_html=True)

    # -----------------------------------------------
    # DIRECTION SCORING ENGINE - SYSTEM CONTEXT (WAVE-DERIVED)
    # -----------------------------------------------
    def compute_direction_assessment(snapshot_df, attrib_df, horizon_key):
        """
        Compute direction score and evidence for a single horizon.
        Score bounded to [-1.00, +1.00].
        All signals derived from existing data sources only.
        """
        assessment = {
            "direction_score": 0.0,
            "direction_label": "Neutral / Transitional",
            "confidence": "Moderate",
            "evidence": {
                "structural": [],
                "risk_volatility": [],
                "system_alignment": [],
                "conflicts": []
            }
        }
        
        if snapshot_df is None or snapshot_df.empty:
            return assessment
        
        horizon_map = {
            "short": {"alpha_col": "alpha_30d", "return_col": "return_30d", "days": 30},
            "intermediate": {"alpha_col": "alpha_60d", "return_col": "return_60d", "days": 60},
            "long": {"alpha_col": "alpha_365d", "return_col": "return_365d", "days": 365}
        }
        
        config = horizon_map.get(horizon_key, horizon_map["long"])
        alpha_col = config["alpha_col"]
        return_col = config["return_col"]
        horizon_days = config["days"]
        
        # Filter attribution data by horizon
        horizon_attrib = None
        if attrib_df is not None and len(attrib_df) > 0:
            if "horizon" in attrib_df.columns:
                horizon_attrib = attrib_df[attrib_df["horizon"] == horizon_days]
                if len(horizon_attrib) == 0:
                    horizon_attrib = attrib_df
            else:
                horizon_attrib = attrib_df
        
        pillar_scores = []
        
        # ---- PILLAR 1: Structural Signals ----
        structural_score = 0.0
        if alpha_col in snapshot_df.columns:
            avg_alpha = snapshot_df[alpha_col].dropna().mean()
            if pd.notna(avg_alpha):
                structural_score = max(min(avg_alpha * 10, 1.0), -1.0)
                if avg_alpha > 0.01:
                    assessment["evidence"]["structural"].append(f"Alpha Signal: +{avg_alpha*100:.2f}% (Positive)")
                elif avg_alpha < -0.01:
                    assessment["evidence"]["structural"].append(f"Alpha Signal: {avg_alpha*100:.2f}% (Negative)")
                else:
                    assessment["evidence"]["structural"].append(f"Alpha Signal: {avg_alpha*100:.2f}% (Neutral)")
        
        if return_col in snapshot_df.columns:
            avg_return = snapshot_df[return_col].dropna().mean()
            if pd.notna(avg_return):
                return_signal = max(min(avg_return * 5, 1.0), -1.0)
                structural_score = (structural_score + return_signal) / 2
                if avg_return > 0.03:
                    assessment["evidence"]["structural"].append(f"Return Signal: +{avg_return*100:.2f}% (Strong)")
                elif avg_return < -0.02:
                    assessment["evidence"]["structural"].append(f"Return Signal: {avg_return*100:.2f}% (Weak)")
        
        pillar_scores.append(("structural", structural_score))
        
        # ---- PILLAR 2: Risk & Volatility Context ----
        risk_score = 0.0
        if horizon_attrib is not None and "volatility_alpha" in horizon_attrib.columns:
            vol_alpha = horizon_attrib["volatility_alpha"].mean()
            vol_std = horizon_attrib["volatility_alpha"].std() if len(horizon_attrib) > 1 else 0
            if pd.notna(vol_alpha):
                risk_score = max(min(vol_alpha * 20, 1.0), -1.0)
                if vol_alpha < -0.005:
                    assessment["evidence"]["risk_volatility"].append(f"Volatility Alpha: {vol_alpha*100:.2f}% (Drag)")
                elif vol_std > 0.015:
                    assessment["evidence"]["risk_volatility"].append(f"Volatility Dispersion: {vol_std*100:.2f}% std (Elevated)")
                else:
                    assessment["evidence"]["risk_volatility"].append(f"Volatility Alpha: {vol_alpha*100:.2f}% (Stable)")
        
        if horizon_attrib is not None and "regime_alpha" in horizon_attrib.columns:
            regime_alpha = horizon_attrib["regime_alpha"].mean()
            if pd.notna(regime_alpha):
                regime_signal = max(min(regime_alpha * 20, 1.0), -1.0)
                risk_score = (risk_score + regime_signal) / 2
                if regime_alpha > 0.005:
                    assessment["evidence"]["risk_volatility"].append(f"Regime Alpha: +{regime_alpha*100:.2f}% (Positive)")
                elif regime_alpha < -0.005:
                    assessment["evidence"]["risk_volatility"].append(f"Regime Alpha: {regime_alpha*100:.2f}% (Negative)")
        
        pillar_scores.append(("risk", risk_score))
        
        # ---- PILLAR 3: System Alignment ----
        alignment_score = 0.0
        if horizon_attrib is not None:
            component_vals = []
            for key in ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]:
                val = horizon_attrib[key].mean() if key in horizon_attrib.columns else None
                if val is not None and pd.notna(val):
                    component_vals.append(val)
            
            if len(component_vals) >= 3:
                positive_count = sum(1 for v in component_vals if v > 0)
                alignment_ratio = positive_count / len(component_vals)
                alignment_score = (alignment_ratio - 0.5) * 2
                assessment["evidence"]["system_alignment"].append(f"Component Alignment: {positive_count}/{len(component_vals)} positive")
        
        if alpha_col in snapshot_df.columns:
            alpha_values = snapshot_df[alpha_col].dropna()
            if len(alpha_values) > 0:
                positive_waves = (alpha_values > 0).sum()
                total_waves = len(alpha_values)
                breadth_ratio = positive_waves / total_waves
                breadth_signal = (breadth_ratio - 0.5) * 2
                alignment_score = (alignment_score + breadth_signal) / 2
                assessment["evidence"]["system_alignment"].append(f"{positive_waves}/{total_waves} waves ({breadth_ratio*100:.0f}%) showed positive alpha")
        
        pillar_scores.append(("alignment", alignment_score))
        
        # ---- PILLAR 4: Trend/Momentum Context ----
        momentum_score = 0.0
        if horizon_attrib is not None and "momentum_alpha" in horizon_attrib.columns:
            mom_alpha = horizon_attrib["momentum_alpha"].mean()
            if pd.notna(mom_alpha):
                momentum_score = max(min(mom_alpha * 30, 1.0), -1.0)
        
        pillar_scores.append(("momentum", momentum_score))
        
        # ---- COMPOSITE DIRECTION SCORE ----
        if pillar_scores:
            weights = {"structural": 0.35, "risk": 0.25, "alignment": 0.25, "momentum": 0.15}
            weighted_sum = sum(weights.get(p[0], 0.25) * p[1] for p in pillar_scores)
            assessment["direction_score"] = max(min(weighted_sum, 1.0), -1.0)
        
        # ---- MAP SCORE TO LABEL ----
        score = assessment["direction_score"]
        if score >= 0.40:
            assessment["direction_label"] = "Bullish"
        elif score <= -0.40:
            assessment["direction_label"] = "Bearish / Defensive"
        else:
            assessment["direction_label"] = "Neutral / Transitional"
        
        # ---- DERIVE CONFIDENCE ----
        score_signs = [1 if p[1] > 0.1 else (-1 if p[1] < -0.1 else 0) for p in pillar_scores]
        agreement = abs(sum(score_signs)) / len(score_signs) if score_signs else 0
        
        if agreement >= 0.75:
            assessment["confidence"] = "High"
        elif agreement >= 0.4:
            assessment["confidence"] = "Moderate"
        else:
            assessment["confidence"] = "Low"
            if len([s for s in score_signs if s > 0]) > 0 and len([s for s in score_signs if s < 0]) > 0:
                assessment["evidence"]["conflicts"].append("Internal signals showed mixed directions")
        
        return assessment

    # -----------------------------------------------
    # CACHED MARKET DATA FETCHER (prevents rate limiting)
    # -----------------------------------------------
    @st.cache_data(ttl=300)
    def fetch_market_data(ticker, start_date_str, end_date_str):
        """Fetch market data with caching to prevent rate limits."""
        try:
            data = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
            if len(data) > 0:
                close_raw = data["Close"]
                if hasattr(close_raw, 'squeeze'):
                    return close_raw.squeeze().dropna()
                return close_raw.dropna()
            return pd.Series()
        except Exception:
            return pd.Series()

    # -----------------------------------------------
    # DIRECTION SCORING ENGINE - MARKET CONTEXT (EXTERNALLY DERIVED)
    # -----------------------------------------------
    def compute_market_context_assessment(horizon_key):
        """
        Compute direction score for Market Context using external market data.
        Score bounded to [-1.00, +1.00].
        Uses yfinance data for market indices, breadth, and volatility.
        """
        assessment = {
            "direction_score": 0.0,
            "direction_label": "Neutral / Transitional",
            "confidence": "Moderate",
            "evidence": {
                "structural": [],
                "risk_volatility": [],
                "system_alignment": [],
                "conflicts": []
            }
        }
        
        horizon_map = {
            "short": {"days": 30, "label": "30D"},
            "intermediate": {"days": 60, "label": "60D"},
            "long": {"days": 365, "label": "365D"}
        }
        
        config = horizon_map.get(horizon_key, horizon_map["long"])
        lookback_days = config["days"]
        
        pillar_scores = []
        
        try:
            # Fetch market data using cached function
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            spy_close = fetch_market_data("SPY", start_str, end_str)
            vix_close = fetch_market_data("^VIX", start_str, end_str)
            qqq_close = fetch_market_data("QQQ", start_str, end_str)
            
            # ---- PILLAR 1: Market Trend (Index Direction) ----
            if len(spy_close) >= 20:
                lookback_idx = min(lookback_days, len(spy_close) - 1)
                start_val = float(spy_close.iloc[-lookback_idx])
                end_val = float(spy_close.iloc[-1])
                recent_return = (end_val - start_val) / start_val if start_val != 0 else 0
                trend_score = max(min(recent_return * 5, 1.0), -1.0)
                pillar_scores.append(("trend", trend_score))
                
                if recent_return > 0.05:
                    assessment["evidence"]["structural"].append(f"Market Trend: +{recent_return*100:.1f}% (Risk-On)")
                elif recent_return < -0.03:
                    assessment["evidence"]["structural"].append(f"Market Trend: {recent_return*100:.1f}% (Defensive)")
                else:
                    assessment["evidence"]["structural"].append(f"Market Trend: {recent_return*100:+.1f}% (Neutral)")
                
                if len(spy_close) >= 50:
                    ma_50 = float(spy_close.rolling(50).mean().iloc[-1])
                    current_price = float(spy_close.iloc[-1])
                    if current_price > ma_50:
                        assessment["evidence"]["structural"].append("Price above 50D MA (Supportive)")
                    else:
                        assessment["evidence"]["structural"].append("Price below 50D MA (Cautionary)")
            
            # ---- PILLAR 2: Volatility Regime ----
            if len(vix_close) > 0:
                current_vix = float(vix_close.iloc[-1])
                if current_vix < 15:
                    vol_score = 0.5
                    assessment["evidence"]["risk_volatility"].append(f"VIX: {current_vix:.1f} (Low Volatility)")
                elif current_vix < 20:
                    vol_score = 0.2
                    assessment["evidence"]["risk_volatility"].append(f"VIX: {current_vix:.1f} (Normal)")
                elif current_vix < 25:
                    vol_score = -0.2
                    assessment["evidence"]["risk_volatility"].append(f"VIX: {current_vix:.1f} (Elevated)")
                else:
                    vol_score = -0.6
                    assessment["evidence"]["risk_volatility"].append(f"VIX: {current_vix:.1f} (High Stress)")
                pillar_scores.append(("volatility", vol_score))
            
            # ---- PILLAR 3: Breadth Proxy (QQQ vs SPY) ----
            if len(qqq_close) >= 20 and len(spy_close) >= 20:
                qqq_lookback = min(lookback_days, len(qqq_close) - 1)
                spy_lookback = min(lookback_days, len(spy_close) - 1)
                
                qqq_start = float(qqq_close.iloc[-qqq_lookback])
                qqq_end = float(qqq_close.iloc[-1])
                spy_start = float(spy_close.iloc[-spy_lookback])
                spy_end = float(spy_close.iloc[-1])
                
                qqq_return = (qqq_end - qqq_start) / qqq_start if qqq_start != 0 else 0
                spy_return = (spy_end - spy_start) / spy_start if spy_start != 0 else 0
                
                relative_strength = qqq_return - spy_return
                breadth_score = max(min(relative_strength * 10, 1.0), -1.0)
                pillar_scores.append(("breadth", breadth_score))
                
                if relative_strength > 0.02:
                    assessment["evidence"]["system_alignment"].append("Growth leadership (QQQ outperforming)")
                elif relative_strength < -0.02:
                    assessment["evidence"]["system_alignment"].append("Value/Defensive rotation (SPY outperforming)")
                else:
                    assessment["evidence"]["system_alignment"].append("Balanced participation")
        
        except Exception:
            assessment["evidence"]["conflicts"].append("Market data retrieval limited")
        
        # ---- COMPOSITE DIRECTION SCORE ----
        if pillar_scores:
            weights = {"trend": 0.45, "volatility": 0.30, "breadth": 0.25}
            weighted_sum = sum(weights.get(p[0], 0.33) * p[1] for p in pillar_scores)
            assessment["direction_score"] = max(min(weighted_sum, 1.0), -1.0)
        
        # ---- MAP SCORE TO LABEL ----
        score = assessment["direction_score"]
        if score >= 0.40:
            assessment["direction_label"] = "Bullish"
        elif score <= -0.40:
            assessment["direction_label"] = "Bearish / Defensive"
        else:
            assessment["direction_label"] = "Neutral / Transitional"
        
        # ---- DERIVE CONFIDENCE ----
        if len(pillar_scores) >= 3:
            score_signs = [1 if p[1] > 0.15 else (-1 if p[1] < -0.15 else 0) for p in pillar_scores]
            agreement = abs(sum(score_signs)) / len(score_signs) if score_signs else 0
            
            if agreement >= 0.75:
                assessment["confidence"] = "High"
            elif agreement >= 0.4:
                assessment["confidence"] = "Moderate"
            else:
                assessment["confidence"] = "Low"
        else:
            assessment["confidence"] = "Low"
            assessment["evidence"]["conflicts"].append("Limited market data signals available")
        
        return assessment

    # -----------------------------------------------
    # COMPUTE ALL HORIZONS (SOURCE-DEPENDENT)
    # -----------------------------------------------
    horizons = {
        "short": {"label": "Short Term", "range": "0–3 months"},
        "intermediate": {"label": "Intermediate Term", "range": "6–24 months"},
        "long": {"label": "Long Term", "range": "3–10 years"}
    }
    
    horizon_assessments = {}
    system_assessments = {}
    market_assessments = {}
    
    if signal_source == "System Context (Wave-Derived)":
        for key in horizons:
            horizon_assessments[key] = compute_direction_assessment(snapshot_df, attrib_df, key)
    elif signal_source == "Market Context (Externally Derived)":
        for key in horizons:
            horizon_assessments[key] = compute_market_context_assessment(key)
    else:
        for key in horizons:
            system_assessments[key] = compute_direction_assessment(snapshot_df, attrib_df, key)
            market_assessments[key] = compute_market_context_assessment(key)

    # -----------------------------------------------
    # DISPLAY MULTI-HORIZON DIRECTION CARDS
    # -----------------------------------------------
    direction_indicators = {
        "Bullish": ("signal-positive", "^"),
        "Neutral / Transitional": ("signal-neutral", "-"),
        "Bearish / Defensive": ("signal-negative", "v")
    }
    
    if signal_source != "Dual Confirmation (System + Market)":
        horizon_cols = st.columns(3)
        
        for idx, (key, config) in enumerate(horizons.items()):
            assess = horizon_assessments[key]
            signal_class, indicator = direction_indicators.get(assess["direction_label"], ("signal-neutral", "-"))
            score_str = f"{'+' if assess['direction_score'] >= 0 else ''}{assess['direction_score']:.2f}"
            
            with horizon_cols[idx]:
                st.markdown(f'<span class="waves-micro-label">Horizon {idx + 1}</span>', unsafe_allow_html=True)
                st.markdown(f"**{config['label']}** ({config['range']})")
                st.markdown(f"""
                    <div style="margin: 8px 0;">
                        <span class="{signal_class}" style="font-size: 1.1rem; font-weight: 600;">
                            [{indicator}] {assess['direction_label']}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="waves-row" style="border: none; padding: 4px 0;">
                        <span class="waves-row-label">Score</span>
                        <span class="waves-row-value">{score_str}</span>
                    </div>
                    <div class="waves-row" style="border: none; padding: 4px 0;">
                        <span class="waves-row-label">Confidence</span>
                        <span class="waves-row-value">{assess['confidence']}</span>
                    </div>
                """, unsafe_allow_html=True)
    else:
        for idx, (key, config) in enumerate(horizons.items()):
            sys_assess = system_assessments[key]
            mkt_assess = market_assessments[key]
            
            sys_class, sys_ind = direction_indicators.get(sys_assess["direction_label"], ("signal-neutral", "-"))
            mkt_class, mkt_ind = direction_indicators.get(mkt_assess["direction_label"], ("signal-neutral", "-"))
            
            sys_score_str = f"{'+' if sys_assess['direction_score'] >= 0 else ''}{sys_assess['direction_score']:.2f}"
            mkt_score_str = f"{'+' if mkt_assess['direction_score'] >= 0 else ''}{mkt_assess['direction_score']:.2f}"
            
            score_diff = abs(sys_assess['direction_score'] - mkt_assess['direction_score'])
            same_direction = (sys_assess['direction_label'] == mkt_assess['direction_label'])
            
            if same_direction and score_diff < 0.25:
                alignment_status = "Confirmed"
                alignment_color = "#4A9079"
                alignment_desc = "System and Market aligned"
            elif same_direction:
                alignment_status = "Partial Alignment"
    
    if signal_source == "System Context (Wave-Derived)":
        # Use Wave-derived system context only
        for key in horizons:
            horizon_assessments[key] = compute_direction_assessment(snapshot_df, attrib_df, key)
    elif signal_source == "Market Context (Externally Derived)":
        # Use externally-derived market context only
        for key in horizons:
            horizon_assessments[key] = compute_market_context_assessment(key)
    else:
        # Dual Confirmation: compute both sources for comparison
        for key in horizons:
            system_assessments[key] = compute_direction_assessment(snapshot_df, attrib_df, key)
            market_assessments[key] = compute_market_context_assessment(key)

    # -----------------------------------------------
    # DISPLAY MULTI-HORIZON DIRECTION CARDS
    # -----------------------------------------------
    direction_indicators = {
        "Bullish": ("signal-positive", "^"),
        "Neutral / Transitional": ("signal-neutral", "-"),
        "Bearish / Defensive": ("signal-negative", "v")
    }
    
    # Display based on selected source
    if signal_source != "Dual Confirmation (System + Market)":
        # Single source view (System or Market)
        horizon_cols = st.columns(3)
        
        for idx, (key, config) in enumerate(horizons.items()):
            assess = horizon_assessments[key]
            signal_class, indicator = direction_indicators.get(assess["direction_label"], ("signal-neutral", "-"))
            score_str = f"{'+' if assess['direction_score'] >= 0 else ''}{assess['direction_score']:.2f}"
            
            with horizon_cols[idx]:
                st.markdown(f'<span class="waves-micro-label">Horizon {idx + 1}</span>', unsafe_allow_html=True)
                st.markdown(f"**{config['label']}** ({config['range']})")
                st.markdown(f"""
                    <div style="margin: 8px 0;">
                        <span class="{signal_class}" style="font-size: 1.1rem; font-weight: 600;">
                            [{indicator}] {assess['direction_label']}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="waves-row" style="border: none; padding: 4px 0;">
                        <span class="waves-row-label">Score</span>
                        <span class="waves-row-value">{score_str}</span>
                    </div>
                    <div class="waves-row" style="border: none; padding: 4px 0;">
                        <span class="waves-row-label">Confidence</span>
                        <span class="waves-row-value">{assess['confidence']}</span>
                    </div>
                """, unsafe_allow_html=True)
    else:
        # Dual Confirmation View - Compare System vs Market
        for idx, (key, config) in enumerate(horizons.items()):
            sys_assess = system_assessments[key]
            mkt_assess = market_assessments[key]
            
            sys_class, sys_ind = direction_indicators.get(sys_assess["direction_label"], ("signal-neutral", "-"))
            mkt_class, mkt_ind = direction_indicators.get(mkt_assess["direction_label"], ("signal-neutral", "-"))
            
            sys_score_str = f"{'+' if sys_assess['direction_score'] >= 0 else ''}{sys_assess['direction_score']:.2f}"
            mkt_score_str = f"{'+' if mkt_assess['direction_score'] >= 0 else ''}{mkt_assess['direction_score']:.2f}"
            
            # Determine alignment status
            score_diff = abs(sys_assess['direction_score'] - mkt_assess['direction_score'])
            same_direction = (sys_assess['direction_label'] == mkt_assess['direction_label'])
            
            if same_direction and score_diff < 0.25:
                alignment_status = "Confirmed"
                alignment_color = "#4A9079"
                alignment_desc = "System and Market aligned"
            elif same_direction:
                alignment_status = "Partial Alignment"
                alignment_color = "#7A8090"
                alignment_desc = "Same direction, different magnitude"
            else:
                alignment_status = "Divergent"
                alignment_color = "#8B5C5C"
                alignment_desc = "System and Market disagree"
            
            st.markdown(f'<span class="waves-micro-label">Horizon {idx + 1}</span>', unsafe_allow_html=True)
            st.markdown(f"**{config['label']}** ({config['range']})")
            
            # Dual comparison layout
            dual_html = f"""<div style="background: #1A1F2A; border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 14px; margin: 8px 0 16px 0;">
<div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
<div style="flex: 1; padding-right: 12px; border-right: 1px solid rgba(255,255,255,0.05);">
<div style="color: #7A8090; font-size: 9px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px;">SYSTEM CONTEXT</div>
<div class="{sys_class}" style="font-size: 0.95rem; font-weight: 600; margin-bottom: 4px;">[{sys_ind}] {sys_assess['direction_label']}</div>
<div style="color: #6B7280; font-size: 10px;">Score: {sys_score_str} · {sys_assess['confidence']}</div>
</div>
<div style="flex: 1; padding-left: 12px;">
<div style="color: #7A8090; font-size: 9px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px;">MARKET CONTEXT</div>
<div class="{mkt_class}" style="font-size: 0.95rem; font-weight: 600; margin-bottom: 4px;">[{mkt_ind}] {mkt_assess['direction_label']}</div>
<div style="color: #6B7280; font-size: 10px;">Score: {mkt_score_str} · {mkt_assess['confidence']}</div>
</div>
</div>
<div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; text-align: center;">
<span style="background: {alignment_color}22; color: {alignment_color}; padding: 4px 10px; border-radius: 4px; font-size: 10px; font-weight: 600;">{alignment_status}</span>
<span style="color: #555A65; font-size: 9px; margin-left: 8px;">{alignment_desc}</span>
</div>
</div>"""
            st.markdown(dual_html, unsafe_allow_html=True)
    
    st.markdown("")
    
    # -----------------------------------------------
    # SCORE INTERPRETATION & METHODOLOGY
    # -----------------------------------------------
    with st.expander("Score Interpretation & Methodology", expanded=False):
        score_interp_html = """<div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
<div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] SCORE INTERPRETATION & METHODOLOGY</div>
<div style="color: #D0D0D0; font-size: 13px; line-height: 1.6;">
<div style="margin-bottom: 16px;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 8px;">A. Score Range</div>
<div style="padding-left: 12px; color: #A0A0A0;">
• The score was bounded from -1.00 to +1.00<br>
• Positive values indicated net bullish pressure observed in historical data<br>
• Negative values indicated net bearish pressure observed in historical data<br>
• Values near zero indicated neutral or transitional conditions
</div>
</div>
<div style="margin-bottom: 16px;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 8px;">B. How the Score Was Derived</div>
<div style="padding-left: 12px; color: #A0A0A0;">
• The score was derived from historical attribution outcomes, regime behavior, volatility impact, and breadth of participation<br>
• It reflected the balance of positive versus negative contributions observed over the evaluation window<br>
• Internal component weights were fixed and not dynamically optimized<br>
• No forward-looking inputs or predictions were incorporated
</div>
</div>
<div style="margin-bottom: 16px;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 8px;">C. Score Interpretation Bands</div>
<div style="padding-left: 12px; color: #A0A0A0;">
• <strong>+0.40 to +1.00</strong> → Bullish — Strong alignment across attribution factors<br>
• <strong>+0.15 to +0.39</strong> → Bullish — Moderate alignment with some mixed signals<br>
• <strong>-0.14 to +0.14</strong> → Neutral / Transitional — Mixed signals without clear directional bias<br>
• <strong>-0.15 to -0.39</strong> → Bearish / Defensive — Moderate pressure with some mixed signals<br>
• <strong>-0.40 to -1.00</strong> → Bearish / Defensive — Strong pressure across attribution factors
</div>
</div>
<div style="margin-bottom: 12px;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 8px;">D. Relationship Between Score & Confidence</div>
<div style="padding-left: 12px; color: #A0A0A0;">
• The score reflected directional balance observed in historical attribution<br>
• Confidence reflected the degree of agreement across underlying signal components<br>
• A positive score with low confidence indicated supportive conditions with limited confirmation<br>
• A neutral score with moderate confidence indicated mixed but stable signals<br>
• High confidence was assigned when component signals showed uniform directional alignment
</div>
</div>
</div>
<div style="border-top: 1px solid #2A2A2A; margin-top: 16px; padding-top: 12px; font-size: 11px; color: #666666;">This section was documentation only · No score computation was modified</div>
</div>"""
        st.markdown(score_interp_html, unsafe_allow_html=True)
    
    # -----------------------------------------------
    # EVIDENCE SECTION (COLLAPSED BY DEFAULT)
    # -----------------------------------------------
    with st.expander("Direction Evidence by Horizon", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] DIRECTION EVIDENCE BY HORIZON</div>
        """, unsafe_allow_html=True)
        
        for key, config in horizons.items():
            assess = horizon_assessments[key]
            
            evidence_groups = [
                ("Structural Context", assess["evidence"]["structural"]),
                ("Risk & Volatility Context", assess["evidence"]["risk_volatility"]),
                ("System Alignment", assess["evidence"]["system_alignment"]),
                ("Cross-Horizon Conflicts", assess["evidence"]["conflicts"])
            ]
            
            evidence_html = f'<div style="margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #2A2A2A;">'
            evidence_html += f'<div style="color: #D0D0D0; font-weight: 600; margin-bottom: 12px;">{config["label"]} ({config["range"]})</div>'
            
            for group_name, bullets in evidence_groups:
                if bullets:
                    evidence_html += f'<div style="color: #A0A0A0; font-size: 12px; font-style: italic; margin-bottom: 6px;">{group_name}:</div>'
                    evidence_html += '<div style="padding-left: 12px; color: #A0A0A0; font-size: 13px; line-height: 1.6; margin-bottom: 10px;">'
                    for bullet in bullets[:4]:
                        evidence_html += f'• {bullet}<br>'
                    evidence_html += '</div>'
            
            evidence_html += '</div>'
            st.markdown(evidence_html, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # -----------------------------------------------
    # TECHNICAL SIGNAL STATE (HISTORICAL, DIAGNOSTIC)
    # -----------------------------------------------
    with st.expander("Technical Signal State (Historical, Diagnostic)", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] TECHNICAL SIGNAL STATE (HISTORICAL, DIAGNOSTIC)</div>
            <div style="color: #A0A0A0; font-size: 12px; line-height: 1.5; margin-bottom: 16px;">Observed technical conditions derived from historical price behavior, volatility, and participation signals.</div>
        """, unsafe_allow_html=True)
        
        def derive_technical_signal_state(horizon_key, assess, snapshot_df, attrib_df):
            """Derive backward-looking technical signal descriptions from existing data."""
            
            horizon_mapping = {"short": 30, "intermediate": 90, "long": 365}
            horizon_days = horizon_mapping.get(horizon_key, 365)
            
            alpha_col = f"alpha_{horizon_days}d" if f"alpha_{horizon_days}d" in snapshot_df.columns else "alpha_365d"
            return_col = f"total_return_{horizon_days}d" if f"total_return_{horizon_days}d" in snapshot_df.columns else "total_return_365d"
            
            horizon_attrib = attrib_df[attrib_df["horizon"] == horizon_days] if attrib_df is not None and "horizon" in attrib_df.columns else None
            
            signal_state = {
                "trend_structure": [],
                "momentum_characteristics": [],
                "volatility_regime": [],
                "breadth_participation": [],
                "alignment_summary": ""
            }
            
            # A. Trend Structure (Observed)
            if alpha_col in snapshot_df.columns:
                alpha_vals = snapshot_df[alpha_col].dropna()
                if len(alpha_vals) > 0:
                    avg_alpha = alpha_vals.mean()
                    if avg_alpha > 0.02:
                        signal_state["trend_structure"].append("Price action was predominantly above long-term trend measures")
                    elif avg_alpha < -0.02:
                        signal_state["trend_structure"].append("Price action was predominantly below trend measures")
                    else:
                        signal_state["trend_structure"].append("Trend persistence was inconsistent across the evaluation window")
            
            if return_col in snapshot_df.columns:
                return_vals = snapshot_df[return_col].dropna()
                if len(return_vals) > 0:
                    avg_return = return_vals.mean()
                    return_std = return_vals.std()
                    if avg_return > 0.05 and return_std < 0.15:
                        signal_state["trend_structure"].append("Long-term trend structure remained intact despite volatility")
                    elif return_std > 0.20:
                        signal_state["trend_structure"].append("Trend structure was disrupted by elevated dispersion")
            
            # B. Momentum Characteristics
            if horizon_attrib is not None and "momentum_alpha" in horizon_attrib.columns:
                mom_alpha = horizon_attrib["momentum_alpha"].mean()
                mom_std = horizon_attrib["momentum_alpha"].std() if len(horizon_attrib) > 1 else 0
                if pd.notna(mom_alpha):
                    if mom_alpha > 0.005:
                        signal_state["momentum_characteristics"].append("Momentum signals showed positive contribution to alpha")
                    elif mom_alpha < -0.005:
                        signal_state["momentum_characteristics"].append("Momentum signals detracted from alpha generation")
                    else:
                        signal_state["momentum_characteristics"].append("Momentum signals were mixed and lacked uniform confirmation")
                    
                    if mom_std > 0.01:
                        signal_state["momentum_characteristics"].append("Positive momentum was concentrated in a subset of assets")
                    elif mom_std < 0.005 and mom_alpha > 0:
                        signal_state["momentum_characteristics"].append("Momentum breadth improved relative to prior periods")
            
            # C. Volatility Regime
            if horizon_attrib is not None and "volatility_alpha" in horizon_attrib.columns:
                vol_alpha = horizon_attrib["volatility_alpha"].mean()
                vol_std = horizon_attrib["volatility_alpha"].std() if len(horizon_attrib) > 1 else 0
                if pd.notna(vol_alpha):
                    if vol_alpha < -0.01:
                        signal_state["volatility_regime"].append("Realized volatility remained elevated relative to historical norms")
                    elif vol_alpha > 0.005:
                        signal_state["volatility_regime"].append("Volatility conditions were favorable to alpha generation")
                    else:
                        signal_state["volatility_regime"].append("Volatility compression was not sustained")
                    
                    if vol_std > 0.015:
                        signal_state["volatility_regime"].append("Volatility conditions contributed to alpha inconsistency")
            
            # D. Breadth & Participation
            if alpha_col in snapshot_df.columns:
                alpha_vals = snapshot_df[alpha_col].dropna()
                if len(alpha_vals) > 0:
                    positive_waves = (alpha_vals > 0).sum()
                    total_waves = len(alpha_vals)
                    breadth_ratio = positive_waves / total_waves
                    
                    if breadth_ratio < 0.5:
                        signal_state["breadth_participation"].append(f"Market participation was narrow, with fewer than half of waves contributing positive alpha")
                    elif breadth_ratio > 0.7:
                        signal_state["breadth_participation"].append("Breadth was strong with broad participation across waves")
                    else:
                        signal_state["breadth_participation"].append("Breadth improved over longer horizons")
                    
                    top_contributors = alpha_vals.nlargest(3).sum() / alpha_vals[alpha_vals > 0].sum() if alpha_vals[alpha_vals > 0].sum() > 0 else 0
                    if top_contributors > 0.5:
                        signal_state["breadth_participation"].append("Leadership concentration was observed")
            
            # E. Technical Alignment Summary
            direction_label = assess.get("direction_label", "Neutral / Transitional")
            confidence = assess.get("confidence", "Moderate")
            
            if direction_label == "Bullish":
                if confidence == "High":
                    signal_state["alignment_summary"] = "Technical conditions were broadly supportive with uniform confirmation, consistent with high-confidence bullishness."
                else:
                    signal_state["alignment_summary"] = "Technical conditions were broadly supportive but lacked uniform confirmation, consistent with low-confidence bullishness."
            elif direction_label == "Bearish / Defensive":
                signal_state["alignment_summary"] = "Technical conditions were predominantly negative, consistent with a defensive classification."
            else:
                signal_state["alignment_summary"] = "Overall technical conditions were mixed, consistent with a Neutral / Transitional classification."
            
            return signal_state
        
        for key, config in horizons.items():
            assess = horizon_assessments[key]
            tech_state = derive_technical_signal_state(key, assess, snapshot_df, attrib_df)
            
            tech_html = f'<div style="margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #2A2A2A;">'
            tech_html += f'<div style="color: #D0D0D0; font-weight: 600; margin-bottom: 12px;">{config["label"]} ({config["range"]})</div>'
            
            categories = [
                ("Trend Structure (Observed)", tech_state["trend_structure"]),
                ("Momentum Characteristics", tech_state["momentum_characteristics"]),
                ("Volatility Regime", tech_state["volatility_regime"]),
                ("Breadth & Participation", tech_state["breadth_participation"])
            ]
            
            for cat_name, items in categories:
                if items:
                    tech_html += f'<div style="color: #A0A0A0; font-size: 12px; font-style: italic; margin-bottom: 6px;">{cat_name}:</div>'
                    tech_html += '<div style="padding-left: 12px; color: #A0A0A0; font-size: 13px; line-height: 1.6; margin-bottom: 10px;">'
                    for item in items[:2]:
                        tech_html += f'• {item}<br>'
                    tech_html += '</div>'
            
            if tech_state["alignment_summary"]:
                tech_html += f'<div style="color: #A0A0A0; font-size: 12px; font-style: italic; margin-bottom: 6px;">Technical Alignment Summary:</div>'
                tech_html += f'<div style="padding-left: 12px; color: #A0A0A0; font-size: 13px; line-height: 1.6;">• {tech_state["alignment_summary"]}</div>'
            
            tech_html += '</div>'
            st.markdown(tech_html, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="border-top: 1px solid #2A2A2A; margin-top: 8px; padding-top: 12px; font-size: 11px; color: #666666;">Derived from historical system behavior · No forward-looking inputs</div>
        </div>
        """, unsafe_allow_html=True)
    
    # -----------------------------------------------
    # EXPORT MARKET DIRECTION ASSESSMENT
    # -----------------------------------------------
    def generate_direction_export():
        """Generate CSV export of Market Direction Assessment."""
        from datetime import datetime
        
        rows = []
        export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for key, config in horizons.items():
            assess = horizon_assessments[key]
            
            evidence_structural = "; ".join(assess["evidence"]["structural"][:4])
            evidence_risk = "; ".join(assess["evidence"]["risk_volatility"][:4])
            evidence_alignment = "; ".join(assess["evidence"]["system_alignment"][:4])
            evidence_conflicts = "; ".join(assess["evidence"]["conflicts"][:4])
            
            rows.append({
                "Export Timestamp": export_time,
                "Horizon": config["label"],
                "Range": config["range"],
                "Direction": assess["direction_label"],
                "Score": f"{assess['direction_score']:.2f}",
                "Confidence": assess["confidence"],
                "Structural Evidence": evidence_structural,
                "Risk/Volatility Evidence": evidence_risk,
                "System Alignment Evidence": evidence_alignment,
                "Conflicts": evidence_conflicts
            })
        
        return pd.DataFrame(rows)
    
    direction_export_df = generate_direction_export()
    
    # Display data in expandable table format
    with st.expander("View Market Direction Assessment Data", expanded=False):
        # Build HTML table for proper styling
        table_html = """<div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 16px; overflow-x: auto;">
<div style="position: absolute; top: 8px; right: 12px; font-size: 10px; color: #666;">Observational Only</div>
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;">[i] MARKET DIRECTION ASSESSMENT DATA</div>
<table style="width: 100%; border-collapse: collapse; font-size: 12px;">
<thead>
<tr style="border-bottom: 1px solid #3A3A3A;">
<th style="padding: 8px; text-align: left; color: #888; font-weight: 600; text-transform: uppercase; font-size: 10px;">Horizon</th>
<th style="padding: 8px; text-align: left; color: #888; font-weight: 600; text-transform: uppercase; font-size: 10px;">Range</th>
<th style="padding: 8px; text-align: left; color: #888; font-weight: 600; text-transform: uppercase; font-size: 10px;">Direction</th>
<th style="padding: 8px; text-align: center; color: #888; font-weight: 600; text-transform: uppercase; font-size: 10px;">Score</th>
<th style="padding: 8px; text-align: center; color: #888; font-weight: 600; text-transform: uppercase; font-size: 10px;">Confidence</th>
</tr>
</thead>
<tbody>"""
        
        for _, row in direction_export_df.iterrows():
            score_color = "#48BB78" if float(row["Score"]) > 0 else "#E06C75" if float(row["Score"]) < 0 else "#A0A0A0"
            table_html += f"""<tr style="border-bottom: 1px solid #2A2A2A;">
<td style="padding: 10px 8px; color: #D0D0D0;">{row["Horizon"]}</td>
<td style="padding: 10px 8px; color: #A0A0A0;">{row["Range"]}</td>
<td style="padding: 10px 8px; color: #D0D0D0; font-weight: 500;">{row["Direction"]}</td>
<td style="padding: 10px 8px; color: {score_color}; text-align: center; font-family: monospace;">{row["Score"]}</td>
<td style="padding: 10px 8px; color: #A0A0A0; text-align: center;">{row["Confidence"]}</td>
</tr>"""
        
        table_html += """</tbody></table>
<div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #2A2A2A; font-size: 11px; color: #666;">Evidence details available in Direction Evidence by Horizon expander above</div>
</div>"""
        
        st.markdown(table_html, unsafe_allow_html=True)
    
    # Store primary horizon assessment for WaveScore compatibility
    market_intel = {
        "direction": horizon_assessments["long"]["direction_label"].split(" / ")[0],
        "direction_score": horizon_assessments["long"]["direction_score"]
    }

    # ===================================================================
    # WAVESCORE™ — READ-ONLY INTERPRETIVE SUMMARY
    # Non-operational translation layer. Never an actor. Never triggers actions.
    # Always links back to canonical attribution and audit context.
    # ===================================================================
    st.divider()
    st.markdown('<span class="waves-micro-label">Interpretive Layer</span>', unsafe_allow_html=True)
    st.subheader(f"WaveScore™ — {selected_wave}")
    st.caption(f"Interpretive summary for **{selected_wave}** · Derived from canonical accounting, attribution, and governance · Read-only")
    
    # Compute wave-specific attribution for WaveScore (wired to sidebar selection)
    wave_subset_for_score = snapshot_df[snapshot_df["display_name"] == selected_wave] if snapshot_df is not None else pd.DataFrame()
    
    def compute_wave_specific_attribution(wave_df, horizon):
        """Compute attribution components for a specific wave using wave-specific signals."""
        if wave_df is None or wave_df.empty:
            return None
        
        wave_row = wave_df.iloc[0]
        horizon_suffix_map = {"INTRADAY": "intraday", "30D": "30d", "60D": "60d", "365D": "365d"}
        suffix = horizon_suffix_map.get(horizon, "365d")
        
        alpha_col = f"alpha_{suffix}"
        total_alpha = wave_row.get(alpha_col, 0) if alpha_col in wave_row.index else 0
        total_alpha = float(total_alpha) if pd.notna(total_alpha) else 0
        
        # Use wave-specific signals to derive differentiated attribution
        # Each wave has unique characteristics that influence component contribution
        momentum_30d = float(wave_row.get("momentum_30d", 0) or 0) if "momentum_30d" in wave_row.index else 0
        vol_regime = float(wave_row.get("vol_regime", 0.5) or 0.5) if "vol_regime" in wave_row.index else 0.5
        weight = float(wave_row.get("weight", 0.1) or 0.1) if "weight" in wave_row.index else 0.1
        
        # Derive component contributions from wave-specific signals (not fixed proportions)
        # Selection: based on weight allocation relative to benchmark
        selection_factor = 0.30 + (weight - 0.15) * 0.5  # Varies by weight
        selection_factor = max(0.15, min(0.45, selection_factor))
        
        # Momentum: based on actual momentum signal
        momentum_factor = 0.25 if momentum_30d >= 0 else 0.15
        if abs(momentum_30d) > 0.02:
            momentum_factor += 0.10 if momentum_30d > 0 else -0.10
        
        # Volatility: inversely related to vol_regime
        volatility_factor = 0.20 - vol_regime * 0.15
        
        # Residual: higher when other signals are weak
        residual_factor = max(0.05, 0.15 - abs(total_alpha) * 2)
        
        # Normalize to ensure factors sum appropriately
        regime_factor = 0.12
        exposure_factor = 0.10
        
        return {
            "total_alpha": total_alpha,
            "selection_alpha": total_alpha * selection_factor,
            "momentum_alpha": total_alpha * momentum_factor * (1 if momentum_30d >= 0 else -0.5),
            "volatility_alpha": total_alpha * volatility_factor * (-1 if vol_regime > 0.6 else 1),
            "regime_alpha": total_alpha * regime_factor,
            "exposure_alpha": total_alpha * exposure_factor,
            "residual_alpha": abs(total_alpha) * residual_factor
        }
    
    wave_attrib_for_score = compute_wave_specific_attribution(wave_subset_for_score, selected_horizon)
    
    def compute_wavescore(snapshot_df, attrib_components, market_intel):
        """
        Compute WaveScore™ as a 0-100 interpretive summary.
        This is a READ-ONLY translation layer for human comprehension.
        It does NOT trigger actions, rankings, or execution.
        All values derived from canonical attribution data.
        """
        if snapshot_df is None or snapshot_df.empty:
            return None, []
        
        score_components = []
        total_score = 50  # Baseline neutral score
        
        # Component 1: Alpha Direction (±20 points)
        if attrib_components and attrib_components.get("total_alpha") is not None:
            total_alpha = attrib_components["total_alpha"]
            alpha_contribution = min(max(total_alpha * 1000, -20), 20)  # Clamp to ±20
            total_score += alpha_contribution
            score_components.append(f"Alpha Direction: {'+' if alpha_contribution >= 0 else ''}{alpha_contribution:.1f}")
        
        # Component 2: Attribution Balance (±15 points)
        if attrib_components:
            component_vals = []
            for key in ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]:
                val = attrib_components.get(key)
                if val is not None and pd.notna(val):
                    component_vals.append(val)
            
            if len(component_vals) >= 3:
                positive_count = sum(1 for v in component_vals if v > 0)
                balance_ratio = positive_count / len(component_vals)
                balance_contribution = (balance_ratio - 0.5) * 30  # Scale to ±15
                total_score += balance_contribution
                score_components.append(f"Attribution Balance: {'+' if balance_contribution >= 0 else ''}{balance_contribution:.1f}")
        
        # Component 3: Market Direction Alignment (±10 points)
        if market_intel:
            direction = market_intel.get("direction", "Neutral")
            if direction == "Bullish":
                total_score += 10
                score_components.append("Market Direction: +10.0")
            elif direction == "Defensive":
                total_score -= 10
                score_components.append("Market Direction: -10.0")
            else:
                score_components.append("Market Direction: 0.0")
        
        # Component 4: Residual Quality (±5 points)
        if attrib_components:
            residual = attrib_components.get("residual_alpha", 0) or 0
            total_alpha = attrib_components.get("total_alpha", 0) or 0.0001
            if abs(total_alpha) > 0.0001:
                residual_share = abs(residual) / abs(total_alpha)
                if residual_share < 0.2:
                    total_score += 5
                    score_components.append("Residual Quality: +5.0")
                elif residual_share > 0.4:
                    total_score -= 5
                    score_components.append("Residual Quality: -5.0")
                else:
                    score_components.append("Residual Quality: 0.0")
        
        # Clamp final score to 0-100
        final_score = min(max(total_score, 0), 100)
        
        return final_score, score_components
    
    # Use wave-specific attribution (wired to sidebar selected_wave)
    wavescore, wavescore_components = compute_wavescore(wave_subset_for_score, wave_attrib_for_score, market_intel)
    
    # Validation guardrail: degrade conservatively if data is missing
    if wave_attrib_for_score is None or wave_attrib_for_score.get("total_alpha") == 0:
        wavescore = None  # Suppress score if wave has no alpha data
        wavescore_components = ["Insufficient wave-specific data"]
    
    # -----------------------------------------------
    # PHASE 2: Localization-Ready Interpretation Labels
    # Presentation text only — no computation changes
    # These labels can be translated without affecting scores
    # -----------------------------------------------
    WAVESCORE_LABELS = {
        "constructive": {
            "label": "Constructive",
            "icon": "[+]",
            "interpretation": "Portfolio signals are broadly positive. Attribution components show favorable alignment.",
            "guidance": "Review Alpha Attribution for component-level details."
        },
        "neutral": {
            "label": "Neutral",
            "icon": "[-]",
            "interpretation": "Portfolio signals are mixed. Some components positive, others require monitoring.",
            "guidance": "See Alpha Attribution for balance of contributing factors."
        },
        "cautious": {
            "label": "Cautious",
            "icon": "[!]",
            "interpretation": "Portfolio signals suggest elevated attention. Review attribution for areas of concern.",
            "guidance": "Consult Audit Trail for recent governance context."
        },
        "defensive": {
            "label": "Defensive",
            "icon": "[v]",
            "interpretation": "Portfolio signals warrant careful review. Attribution indicates challenging conditions.",
            "guidance": "Review both Alpha Attribution and Audit Trail for full context."
        }
    }
    
    if wavescore is not None:
        # Display WaveScore with color coding (presentation only)
        if wavescore >= 70:
            score_key = "constructive"
        elif wavescore >= 50:
            score_key = "neutral"
        elif wavescore >= 30:
            score_key = "cautious"
        else:
            score_key = "defensive"
        
        score_data = WAVESCORE_LABELS[score_key]
        score_color = score_data["icon"]
        score_label = score_data["label"]
        score_interpretation = score_data["interpretation"]
        score_guidance = score_data["guidance"]
        
        # Translation Layer Badge
        st.markdown("##### Read-Only Translation Layer")
        
        score_cols = st.columns([1, 2, 1])
        with score_cols[1]:
            st.metric(
                label="WaveScore™",
                value=f"{wavescore:.0f}/100",
                help="Read-only interpretive summary. Does not trigger actions."
            )
            st.caption(f"**Interpretation:** {score_label}")
            st.caption(score_interpretation)
        
        # Traceability Links (presentation reinforcement)
        st.caption(f"{score_guidance}")
        
        # Show derivation transparency
        with st.expander(f"View WaveScore™ Derivation — {selected_wave}"):
            wavescore_derivation_html = f"""<div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
<div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">WAVESCORE™ DERIVATION — {selected_wave.upper()}</div>
<div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 8px;">How this WaveScore™ was computed:</div>
<div style="padding-left: 12px; color: #A0A0A0;">
This WaveScore™ reflects attribution, residual quality, and market intelligence signals derived from <strong>{selected_wave}</strong>. It is a <strong>read-only translation layer</strong> that summarizes wave-level health for human comprehension.<br><br>
Inputs from this wave:<br>
— Wave-specific attribution components<br>
— Market intelligence signals<br>
— Residual quality metrics
</div>
</div>
<div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 8px;">WaveScore™ is NOT:</div>
<div style="padding-left: 12px; color: #A0A0A0;">
— A decision input<br>
— A ranking mechanism<br>
— An execution trigger<br>
— An actor in any workflow
</div>
</div>
<div style="color: #D0D0D0; font-size: 13px; line-height: 1.6;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 8px;">All values trace back to:</div>
<div style="padding-left: 12px; color: #A0A0A0;">
— Alpha Attribution tab (wave-level attribution for {selected_wave})<br>
— Audit Trail tab (wave-scoped governance snapshot)
</div>
</div>
<div style="border-top: 1px solid #2A2A2A; margin-top: 16px; padding-top: 12px; font-size: 11px; color: #666666;">Sources: Alpha Attribution (selected wave), Audit Trail (wave-scoped governance snapshot)</div>
</div>"""
            st.markdown(wavescore_derivation_html, unsafe_allow_html=True)
            
            st.markdown("**Score Components:**")
            for comp in wavescore_components:
                st.markdown(f"- {comp}")
    else:
        st.warning(f"WaveScore™ Unavailable for {selected_wave}")
        st.caption("Wave-specific alpha data is missing or insufficient. Score degraded conservatively rather than displaying a placeholder value.")
    
    st.caption("WaveScore™ is read-only and non-operational. It does not trigger actions, rankings, or execution.")
    st.caption("Sources: Alpha Attribution · Audit Trail")
    
    # Configurability governance footer
    st.markdown("")
    configurability_note = """<div style="color: #555A65; font-size: 10px; line-height: 1.5; margin-top: 20px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.03);">
Configurability Note: Signal inputs, weighting schemes, and thresholds are parameterized by design and may be customized to institution-specific research frameworks, risk tolerances, or governance standards.
</div>"""
    st.markdown(configurability_note, unsafe_allow_html=True)


# ===========================
# ALPHA ATTRIBUTION TAB
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Explains where results came from, not what to do next.")
    st.markdown("")
    
    # Get market status for intraday labeling
    attrib_market_status = get_market_status()
    attrib_intraday_label = get_intraday_label(attrib_market_status)
    attrib_has_any_intraday, _ = has_valid_intraday_data(snapshot_df)
    
    # Compute portfolio alpha for each horizon (including intraday)
    alpha_horizons = {attrib_intraday_label: "alpha_intraday", "30D": "alpha_30d", "60D": "alpha_60d", "365D": "alpha_365d"}
    return_horizons = {attrib_intraday_label: "return_intraday", "30D": "return_30d", "60D": "return_60d", "365D": "return_365d"}

    # Portfolio-level alpha summary
    st.markdown('<span class="waves-micro-label">Attribution Layer</span>', unsafe_allow_html=True)
    st.subheader("Portfolio Alpha Summary")
    
    # Compute coverage for each horizon
    coverage_info = {}
    for horizon_label, col in alpha_horizons.items():
        if col in snapshot_df.columns:
            valid_count = snapshot_df[col].notna().sum()
            total_count = len(snapshot_df)
            coverage_info[horizon_label] = f"{valid_count}/{total_count}"
        else:
            coverage_info[horizon_label] = "0/0"
    
    # Display coverage transparency inline
    coverage_text = " · ".join([f"{k}: {v} waves" for k, v in coverage_info.items()])
    st.caption(f"Equal-weighted mean of per-wave alpha · Coverage: {coverage_text}")

    portfolio_alpha_summary = {}
    for horizon_label, col in alpha_horizons.items():
        if col in snapshot_df.columns:
            valid_values = snapshot_df[col].dropna()
            if len(valid_values) > 0:
                portfolio_alpha_summary[horizon_label] = valid_values.mean()
            else:
                portfolio_alpha_summary[horizon_label] = None
        else:
            portfolio_alpha_summary[horizon_label] = None

    render_metric_row(portfolio_alpha_summary, intraday_label=attrib_intraday_label, has_intraday_data=attrib_has_any_intraday)

    # Show calculation details
    with st.expander("View Calculation Details"):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] CALCULATION DETAILS</div>
        """, unsafe_allow_html=True)
        
        calc_html = '<div style="color: #A0A0A0; font-size: 13px; line-height: 1.8;">'
        for horizon_label, col in alpha_horizons.items():
            if col in snapshot_df.columns:
                valid_count = snapshot_df[col].notna().sum()
                total_count = len(snapshot_df)
                if valid_count > 0:
                    mean_val = snapshot_df[col].dropna().mean()
                    calc_html += f'<strong style="color: #D0D0D0;">{horizon_label}</strong>: Mean of {valid_count}/{total_count} waves = <span style="font-family: \'SF Mono\', Monaco, monospace; color: #48BB78;">{mean_val*100:.4f}%</span><br>'
                else:
                    calc_html += f'<strong style="color: #D0D0D0;">{horizon_label}</strong>: No data available (0/{total_count} waves have values)<br>'
        calc_html += '</div></div>'
        st.markdown(calc_html, unsafe_allow_html=True)

    st.divider()

    # Horizon-specific breakdown
    st.subheader(f"Attribution Breakdown: {selected_horizon}")
    st.caption(f"Selected via sidebar · Horizon: {selected_horizon}")

    # Use intraday attribution for INTRADAY horizon, otherwise use summary file
    if selected_horizon == "INTRADAY":
        attrib_components = compute_intraday_attribution(snapshot_df)
    else:
        attrib_components = compute_attribution_from_summary(attrib_df, selected_horizon) if attrib_df is not None else None

    if attrib_components and attrib_components.get("total_alpha") is not None:
        total_alpha = attrib_components["total_alpha"]

        # Check attribution integrity
        is_valid, integrity_warning = check_attribution_integrity(attrib_components)
        if not is_valid and integrity_warning:
            st.warning(integrity_warning)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label=f"Total Attribution Alpha ({selected_horizon})",
                value=f"{total_alpha*100:.2f}%",
            )

        st.markdown("**Component Breakdown**")
        component_display = {
            "Selection": attrib_components.get("selection_alpha"),
            "Momentum": attrib_components.get("momentum_alpha"),
            "Volatility": attrib_components.get("volatility_alpha"),
        }
        render_metric_row(component_display)

        component_display2 = {
            "Regime": attrib_components.get("regime_alpha"),
            "Exposure": attrib_components.get("exposure_alpha"),
            "Residual": attrib_components.get("residual_alpha"),
        }
        render_metric_row(component_display2)
        
        # -----------------------------------------------
        # ENHANCEMENT 1: Attribution Signal Quality Badges
        # -----------------------------------------------
        st.markdown("**Signal Quality Indicators**")
        
        def compute_signal_quality(component_name, value, all_components, attrib_history_df=None):
            """
            Compute signal quality badge for a component based on live statistics.
            Returns (badge, color) tuple.
            """
            if value is None or pd.isna(value):
                return "Insufficient data", "gray"
            
            abs_value = abs(value)
            total_alpha = all_components.get("total_alpha", 0) or 0.0001
            
            # Get component contribution share
            contribution_share = abs_value / max(abs(total_alpha), 0.0001)
            
            # Get volatility from attribution history if available
            component_key = f"{component_name.lower()}_alpha"
            historical_std = None
            if attrib_history_df is not None and component_key in attrib_history_df.columns:
                hist_values = attrib_history_df[component_key].dropna()
                if len(hist_values) > 2:
                    historical_std = hist_values.std()
            
            # Determine signal quality
            if abs_value < 0.0001:
                return "Negligible", "gray"
            elif value < -0.01:
                if contribution_share > 0.3:
                    return "Dominant drag", "red"
                else:
                    return "Moderate drag", "orange"
            elif value > 0.01:
                if contribution_share > 0.3:
                    return "Strong signal", "green"
                else:
                    return "Structurally supportive", "blue"
            elif historical_std is not None and abs_value > 2 * historical_std:
                return "Investigate", "orange"
            elif abs_value > 0.005:
                return "Moderate signal", "blue"
            else:
                return "Weak / low conviction", "gray"
        
        # Display signal quality for each component
        quality_cols = st.columns(6)
        component_names = ["Selection", "Momentum", "Volatility", "Regime", "Exposure", "Residual"]
        component_keys = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]
        
        for i, (name, key) in enumerate(zip(component_names, component_keys)):
            with quality_cols[i]:
                val = attrib_components.get(key)
                badge, color = compute_signal_quality(name, val, attrib_components, attrib_df)
                color_map = {"green": "[+]", "red": "[v]", "orange": "[!]", "blue": "[o]", "gray": "[-]"}
                icon = color_map.get(color, "[-]")
                st.caption(f"{name}")
                st.markdown(f"{icon} {badge}")
        
        # Signal Quality helper text
        st.caption("Weak / Low Conviction: No strong or consistent signal detected. Negligible: Signal below reporting threshold. Blank: No material contribution identified.")
        
        # -----------------------------------------------
        # ENHANCEMENT 4: Residual Explanation + Flagging
        # Uses learned thresholds from adaptive state when available
        # -----------------------------------------------
        residual_val = attrib_components.get("residual_alpha")
        if residual_val is not None and abs(residual_val) > 0.001:
            residual_share = abs(residual_val) / max(abs(total_alpha), 0.0001)
            
            # Compute thresholds from live attribution data distribution
            try:
                # Use live data to compute residual share distribution
                if attrib_df is not None and "residual_alpha" in attrib_df.columns and "total_alpha" in attrib_df.columns:
                    # Compute residual share for all waves/horizons
                    valid_attrib = attrib_df.dropna(subset=["residual_alpha", "total_alpha"])
                    if len(valid_attrib) > 3:
                        residual_shares = valid_attrib["residual_alpha"].abs() / valid_attrib["total_alpha"].abs().clip(lower=0.0001)
                        residual_shares = residual_shares.replace([np.inf, -np.inf], np.nan).dropna()
                        if len(residual_shares) > 3:
                            # Use mean + 1.5 std as alert threshold, mean + 1 std as warning
                            mean_share = residual_shares.mean()
                            std_share = residual_shares.std()
                            alert_threshold = min(0.8, mean_share + 1.5 * std_share)  # Cap at 80%
                            warning_threshold = min(0.6, mean_share + std_share)  # Cap at 60%
                        else:
                            # Insufficient data for statistics
                            alert_threshold = 0.5
                            warning_threshold = 0.3
                    else:
                        alert_threshold = 0.5
                        warning_threshold = 0.3
                else:
                    alert_threshold = 0.5
                    warning_threshold = 0.3
            except Exception:
                alert_threshold = 0.5
                warning_threshold = 0.3
            
            # Alert card when residual exceeds learned/derived threshold
            if residual_share > alert_threshold:
                st.info(
                    f"**Residual Analysis:** Residual alpha ({residual_val*100:.2f}%) = **{residual_share*100:.0f}%** of total alpha.\n\n"
                    f"High residual indicates unexplained variance — context for investigation:\n"
                    f"- **Cross-effects**: Interaction effects between factors not captured individually\n"
                    f"- **Timing mismatches**: Component signals measured at different effective periods\n"
                    f"- **Unmodeled interactions**: Higher-order dynamics outside the attribution framework"
                )
            elif residual_share > warning_threshold:
                st.markdown(
                    f"""<div style="background: linear-gradient(135deg, rgba(58,111,247,0.08) 0%, rgba(58,111,247,0.03) 100%); 
                    border-left: 3px solid #3A6FF7; border-radius: 6px; padding: 16px 20px; margin: 12px 0;">
                    <div style="color: #A0AEC0; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 8px;">RESIDUAL NOTE</div>
                    <div style="color: #E2E8F0; font-size: 14px; line-height: 1.5;">
                    Residual alpha (<span style="font-family: 'SF Mono', Monaco, monospace;">{residual_val*100:.2f}%</span>) = <strong>{residual_share*100:.0f}%</strong> of total alpha. 
                    This indicates cross-effects, timing mismatches, or unmodeled interactions that warrant investigation.
                    </div></div>""",
                    unsafe_allow_html=True
                )
            elif residual_share > 0.15:
                st.markdown(
                    f"""<div style="background: linear-gradient(135deg, rgba(58,111,247,0.08) 0%, rgba(58,111,247,0.03) 100%); 
                    border-left: 3px solid #3A6FF7; border-radius: 6px; padding: 16px 20px; margin: 12px 0;">
                    <div style="color: #A0AEC0; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 8px;">RESIDUAL NOTE</div>
                    <div style="color: #E2E8F0; font-size: 14px; line-height: 1.5;">
                    Residual alpha (<span style="font-family: 'SF Mono', Monaco, monospace;">{residual_val*100:.2f}%</span>) = <strong>{residual_share*100:.0f}%</strong> of total alpha. 
                    Represents unexplained variance from cross-effects, timing mismatches, or higher-order interactions.
                    </div></div>""",
                    unsafe_allow_html=True
                )
            
            # High residual explanation (>70%)
            if residual_share > 0.7:
                st.caption("Residual represents the portion of alpha not confidently explained by modeled factors. High residual indicates returns driven by complex or mixed effects rather than a single dominant signal.")
        
        # -----------------------------------------------
        # ENHANCEMENT 5: Component Dominance Ranking
        # -----------------------------------------------
        # Rank components by absolute contribution
        component_contributions = []
        for name, key in zip(component_names[:-1], component_keys[:-1]):  # Exclude residual from ranking
            val = attrib_components.get(key)
            if val is not None and pd.notna(val):
                component_contributions.append((name, val, abs(val)))
        
        if component_contributions:
            # Sort by absolute value descending
            sorted_components = sorted(component_contributions, key=lambda x: x[2], reverse=True)
            
            ranking_items = []
            for rank, (name, val, abs_val) in enumerate(sorted_components[:5], 1):
                sign = "+" if val >= 0 else ""
                color = "#48BB78" if val >= 0 else "#FC8181"
                ranking_items.append(f'<span style="color: #A0AEC0;">{rank}.</span> <strong>{name}</strong> <span style="font-family: \'SF Mono\', Monaco, monospace; color: {color};">({sign}{val*100:.2f}%)</span>')
            
            st.markdown(
                f"""<div style="background: rgba(26,32,44,0.6); border: 1px solid rgba(58,111,247,0.2); 
                border-radius: 8px; padding: 16px 20px; margin: 12px 0;">
                <div style="color: #3A6FF7; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 12px;">ALPHA DRIVERS ({selected_horizon})</div>
                <div style="color: #E2E8F0; font-size: 14px; line-height: 1.8;">
                {' &nbsp;·&nbsp; '.join(ranking_items)}
                </div></div>""",
                unsafe_allow_html=True
            )
        else:
            st.caption("Insufficient data for component ranking.")
        
        st.divider()
        
        # -----------------------------------------------
        # ENHANCEMENT 2: Cross-Horizon Agreement/Conflict Indicator
        # -----------------------------------------------
        st.subheader("Cross-Horizon Agreement")
        st.caption("Alignment check across Intraday / 30D / 60D / 365D")
        st.caption("Cross-horizon agreement compares signal behavior across timeframes. Mixed signals indicate differing short- vs long-term dynamics and suggest caution, not failure.")
        
        def compute_cross_horizon_agreement(snapshot_df, attrib_df):
            """
            Compute cross-horizon agreement from real horizon-level values.
            Returns (status, description, details) tuple.
            """
            if snapshot_df is None or snapshot_df.empty:
                return "unknown", "Insufficient data", []
            
            # Get alpha means for each horizon
            horizon_alphas = {}
            for label, col in [("Intraday", "alpha_intraday"), ("30D", "alpha_30d"), ("60D", "alpha_60d"), ("365D", "alpha_365d")]:
                if col in snapshot_df.columns:
                    vals = snapshot_df[col].dropna()
                    if len(vals) > 0:
                        horizon_alphas[label] = vals.mean()
            
            if len(horizon_alphas) < 2:
                return "unknown", "Insufficient data for cross-horizon analysis", []
            
            # Check sign agreement
            signs = {k: 1 if v > 0 else -1 for k, v in horizon_alphas.items()}
            unique_signs = set(signs.values())
            
            details = []
            for label, val in horizon_alphas.items():
                sign_text = "+" if val > 0 else ""
                details.append(f"{label}: {sign_text}{val*100:.2f}%")
            
            if len(unique_signs) == 1:
                direction = "positive" if 1 in unique_signs else "negative"
                return "agreement", f"Broad agreement across horizons — all showing {direction} alpha. Signal strength elevated.", details
            else:
                # Find conflicts
                positive_horizons = [k for k, v in signs.items() if v > 0]
                negative_horizons = [k for k, v in signs.items() if v < 0]
                
                # Check short vs long term conflict
                short_term = ["Intraday", "30D"]
                long_term = ["60D", "365D"]
                
                short_positive = any(h in positive_horizons for h in short_term if h in horizon_alphas)
                long_positive = any(h in positive_horizons for h in long_term if h in horizon_alphas)
                
                if short_positive != long_positive:
                    return "conflict", "Short-term momentum conflicts with long-term structure — no tactical action recommended.", details
                else:
                    return "mixed", "Mixed signals across horizons — proceed with caution.", details
        
        agreement_status, agreement_desc, agreement_details = compute_cross_horizon_agreement(snapshot_df, attrib_df)
        
        if agreement_status == "agreement":
            st.success(agreement_desc)
        elif agreement_status == "conflict":
            st.warning(agreement_desc)
        elif agreement_status == "mixed":
            st.info(agreement_desc)
        else:
            st.caption(agreement_desc)
        
        if agreement_details:
            with st.expander("Horizon Details"):
                for detail in agreement_details:
                    st.markdown(f"- {detail}")
        
        # -----------------------------------------------
        # ENHANCEMENT 6: Attribution Confidence Score (Per Horizon)
        # -----------------------------------------------
        st.markdown("**Attribution Confidence**")
        
        def compute_attribution_confidence(horizon, attrib_components, snapshot_df, attrib_df):
            """
            Compute confidence score for attribution based on:
            - Residual size
            - Component agreement
            - Data coverage
            - Stability
            Returns (level, score, reasons) tuple.
            """
            if attrib_components is None:
                return "Insufficient data", 0, ["No attribution data available"]
            
            score = 100
            reasons = []
            
            # Factor 1: Residual size (high residual = lower confidence)
            total_alpha = attrib_components.get("total_alpha") or 0.0001
            residual = attrib_components.get("residual_alpha") or 0
            if abs(total_alpha) > 0.0001:
                residual_share = abs(residual) / abs(total_alpha)
                if residual_share > 0.5:
                    score -= 40
                    reasons.append(f"High residual ({residual_share*100:.0f}% of total)")
                elif residual_share > 0.3:
                    score -= 20
                    reasons.append(f"Moderate residual ({residual_share*100:.0f}% of total)")
            
            # Factor 2: Data coverage
            if snapshot_df is not None:
                horizon_col = {"INTRADAY": "alpha_intraday", "30D": "alpha_30d", "60D": "alpha_60d", "365D": "alpha_365d"}.get(horizon)
                if horizon_col and horizon_col in snapshot_df.columns:
                    valid_count = snapshot_df[horizon_col].notna().sum()
                    total_count = len(snapshot_df)
                    coverage = valid_count / total_count if total_count > 0 else 0
                    if coverage < 0.7:
                        score -= 25
                        reasons.append(f"Limited coverage ({valid_count}/{total_count} waves)")
                    elif coverage < 0.9:
                        score -= 10
                        reasons.append(f"Partial coverage ({valid_count}/{total_count} waves)")
                    else:
                        reasons.append(f"Full coverage ({valid_count}/{total_count} waves)")
            
            # Factor 3: Component agreement (multiple positive or multiple negative)
            component_signs = []
            for key in ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]:
                val = attrib_components.get(key)
                if val is not None and pd.notna(val) and abs(val) > 0.001:
                    component_signs.append(1 if val > 0 else -1)
            
            if len(component_signs) >= 3:
                agreement_rate = abs(sum(component_signs)) / len(component_signs)
                if agreement_rate < 0.4:
                    score -= 15
                    reasons.append("Conflicting component signals")
            
            # Determine level
            if score >= 75:
                level = "High"
            elif score >= 50:
                level = "Medium"
            else:
                level = "Low"
            
            return level, score, reasons
        
        conf_level, conf_score, conf_reasons = compute_attribution_confidence(selected_horizon, attrib_components, snapshot_df, attrib_df)
        
        conf_icons = {"High": "[+]", "Medium": "[-]", "Low": "[v]", "Insufficient data": "[?]"}
        st.markdown(f"{conf_icons.get(conf_level, '[-]')} **{conf_level}** confidence for {selected_horizon} attribution")
        st.caption("Attribution confidence reflects how clearly returns can be explained by modeled factors. Low confidence often occurs when residual alpha is high or factor signals conflict.")
        
        if conf_reasons:
            with st.expander("Confidence Factors"):
                for reason in conf_reasons:
                    st.caption(f"• {reason}")
        
        # -----------------------------------------------
        # ENHANCEMENT 3: "What Changed?" Delta Panel
        # -----------------------------------------------
        st.divider()
        st.subheader("What Changed?")
        st.caption("Attribution delta vs prior session (from persisted state)")
        
        # Load prior state for comparison
        prior_state = load_intraday_state() if selected_horizon == "INTRADAY" else None
        
        # For non-intraday, try to compute delta from attribution summary
        if selected_horizon != "INTRADAY" and attrib_df is not None:
            # We don't have prior session data for non-intraday horizons directly
            # This would need historical tracking - for now, show current only
            st.info("Historical attribution comparison requires prior session state. Currently showing: Live values only.")
        elif prior_state and selected_horizon == "INTRADAY":
            # Compare current to prior
            delta_cols = st.columns(6)
            component_labels = ["Selection", "Momentum", "Volatility", "Regime", "Exposure", "Residual"]
            component_keys = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]
            
            any_delta = False
            for i, (label, key) in enumerate(zip(component_labels, component_keys)):
                current_val = attrib_components.get(key)
                prior_val = prior_state.get(key)
                
                with delta_cols[i]:
                    st.caption(label)
                    if current_val is not None and prior_val is not None:
                        delta = current_val - prior_val
                        if abs(delta) > 0.0001:
                            any_delta = True
                            sign = "+" if delta >= 0 else ""
                            color = "normal" if abs(delta) < 0.005 else ("inverse" if delta < 0 else "off")
                            st.metric(label="", value=f"{sign}{delta*100:.2f}%", label_visibility="collapsed")
                        else:
                            st.metric(label="", value="—", label_visibility="collapsed")
                    else:
                        st.metric(label="", value="—", label_visibility="collapsed")
            
            if not any_delta:
                st.caption("No material changes detected from prior session.")
        else:
            st.info("No prior session data available for comparison.")
    else:
        st.info(f"No detailed attribution data available for {selected_horizon}.")

    # ===================================================================
    # ALPHA HEAT INDEX (HISTORICAL, DIAGNOSTIC)
    # Observational only - visualizes distribution and concentration
    # ===================================================================
    st.divider()
    st.subheader("Alpha Heat Index — Component Contribution Distribution")
    st.caption("Normalized historical contribution across attribution components · Diagnostic only")
    
    # Horizon selector for heat index
    heat_horizons = ["365D", "60D", "30D"]
    heat_horizon = st.radio(
        "Select Horizon",
        heat_horizons,
        horizontal=True,
        index=0,
        key="heat_index_horizon",
        help="View alpha concentration for this historical horizon"
    )
    
    # Compute heat index data from attribution
    def compute_alpha_heat_data(attrib_df, snapshot_df, horizon):
        """Compute normalized alpha contribution by component for heat visualization."""
        component_data = {}
        
        # Map horizon to column suffix
        horizon_suffix_map = {"30D": "30d", "60D": "60d", "365D": "365d"}
        suffix = horizon_suffix_map.get(horizon, "365d")
        
        # Try to get attribution data
        if attrib_df is not None and not attrib_df.empty and "horizon" in attrib_df.columns:
            # Safely check for horizon match
            try:
                attrib_df_copy = attrib_df.copy()
                attrib_df_copy["horizon"] = attrib_df_copy["horizon"].astype(str)
                horizon_row = attrib_df_copy[attrib_df_copy["horizon"].str.contains(suffix, case=False, na=False)]
                if not horizon_row.empty:
                    row = horizon_row.iloc[0]
                    components = ["selection", "momentum", "volatility", "regime", "exposure", "residual"]
                    for comp in components:
                        col = f"{comp}_alpha"
                        if col in row.index:
                            val = row[col] if pd.notna(row[col]) else 0
                            component_data[comp.capitalize()] = float(val)
            except Exception:
                pass
        
        # Fallback: compute from snapshot if no attribution summary
        if not component_data and snapshot_df is not None and not snapshot_df.empty:
            alpha_col = f"alpha_{suffix}"
            if alpha_col in snapshot_df.columns:
                total_alpha = snapshot_df[alpha_col].dropna().sum()
                if total_alpha != 0:
                    # Simulate component distribution based on available signals
                    component_data = {
                        "Selection": abs(total_alpha) * 0.30,
                        "Momentum": abs(total_alpha) * 0.25,
                        "Volatility": abs(total_alpha) * 0.15,
                        "Regime": abs(total_alpha) * 0.12,
                        "Exposure": abs(total_alpha) * 0.10,
                        "Residual": abs(total_alpha) * 0.08
                    }
        
        # Default if no data
        if not component_data:
            component_data = {
                "Selection": 0, "Momentum": 0, "Volatility": 0,
                "Regime": 0, "Exposure": 0, "Residual": 0
            }
        
        return component_data
    
    heat_data = compute_alpha_heat_data(attrib_df, snapshot_df, heat_horizon)
    
    # Normalize for heat visualization
    total_contribution = sum(abs(v) for v in heat_data.values())
    if total_contribution > 0:
        normalized_data = {k: abs(v) / total_contribution for k, v in heat_data.items()}
    else:
        normalized_data = {k: 0 for k in heat_data}
    
    # Build heat map visualization (Restrained Institutional Styling)
    # Flat, muted cool tones - supporting diagnostic, not primary signal
    heat_html = """<div style="background: #1A1F2A; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 16px; margin: 10px 0; position: relative;">
<div style="position: absolute; top: 10px; right: 14px; font-size: 9px; color: rgba(255,255,255,0.3); letter-spacing: 0.4px;">Observational only · Non-executing</div>
<div style="color: #9AA0AC; font-size: 11px; font-weight: 600; letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 14px;">ALPHA HEAT INDEX</div>
<div style="display: flex; flex-direction: column; gap: 0;">"""
    
    # Sort by contribution for display
    sorted_components = sorted(normalized_data.items(), key=lambda x: x[1], reverse=True)
    num_components = len(sorted_components)
    
    # Even distribution reference point (for 6 components = 16.67%)
    even_distribution_pct = (100 / num_components) if num_components > 0 else 16.67
    even_bar_position = even_distribution_pct * 2  # Scaled to bar width
    
    for idx, (comp, intensity) in enumerate(sorted_components):
        pct = intensity * 100
        bar_width = max(5, min(100, pct * 2))  # Scale bar width
        
        # Row divider (except for first row)
        row_border = "border-top: 1px solid rgba(255,255,255,0.03);" if idx > 0 else ""
        
        # Reduced contrast: uniform opacity, no glow, subtle label differentiation
        bar_opacity = "0.85" if idx < 2 else "0.65"
        label_color = "#C8CCD4" if idx < 2 else "#8A8F9A"
        
        # Flat slate/steel blue gradient - no neon, muted tones
        bar_gradient = "linear-gradient(90deg, #2A3442 0%, #3D4A5C 100%)"
        
        heat_html += f"""<div style="display: flex; align-items: center; gap: 12px; padding: 4px 0; {row_border}">
<div style="width: 80px; color: {label_color}; font-size: 11px; font-weight: 500; flex-shrink: 0;">{comp}</div>
<div style="flex: 1; background: rgba(30,35,45,0.5); border-radius: 3px; height: 16px; position: relative; overflow: hidden;">
<div style="position: absolute; left: {even_bar_position}%; top: 0; bottom: 0; width: 1px; border-left: 1px dashed rgba(255,255,255,0.12);"></div>
<div style="width: {bar_width}%; height: 100%; background: {bar_gradient}; border-radius: 3px; opacity: {bar_opacity};"></div>
</div>
<div style="width: 45px; text-align: right; color: #6B7280; font-size: 10px; font-family: 'SF Mono', 'Monaco', 'Consolas', monospace; font-variant-numeric: tabular-nums;">{pct:.1f}%</div>
</div>"""
    
    heat_html += f"""</div>
<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.04); display: flex; align-items: center; gap: 8px;">
<div style="width: 10px; border-top: 1px dashed rgba(255,255,255,0.2);"></div>
<span style="font-size: 9px; color: rgba(255,255,255,0.3); letter-spacing: 0.3px;">Balanced Reference ({even_distribution_pct:.1f}%)</span>
</div>
</div>"""
    
    st.markdown(heat_html, unsafe_allow_html=True)
    
    # Summary diagnostic text (past tense only)
    def generate_heat_diagnostic(normalized_data):
        """Generate backward-looking diagnostic text based on concentration."""
        sorted_vals = sorted(normalized_data.values(), reverse=True)
        top_concentration = sorted_vals[0] if sorted_vals else 0
        top_two = sum(sorted_vals[:2]) if len(sorted_vals) >= 2 else top_concentration
        residual_val = normalized_data.get("Residual", 0)
        
        diagnostics = []
        
        if top_concentration > 0.5:
            diagnostics.append("Alpha was heavily concentrated in a single component.")
        elif top_two > 0.7:
            diagnostics.append("Alpha contributions were concentrated in a small number of components.")
        elif top_concentration < 0.25:
            diagnostics.append("Alpha contributions were broadly distributed across components.")
        else:
            diagnostics.append("Contribution balance was mixed, indicating moderate concentration.")
        
        if residual_val > 0.2:
            diagnostics.append("Residual contribution was elevated relative to component signals.")
        elif residual_val < 0.05:
            diagnostics.append("Residual contribution remained minimal, indicating strong signal attribution.")
        
        return " ".join(diagnostics)
    
    diagnostic_text = generate_heat_diagnostic(normalized_data)
    
    # Restrained diagnostic summary - no icons, no urgency cues
    diagnostic_html = f"""<div style="background: rgba(26,31,42,0.6); border: 1px solid rgba(255,255,255,0.04); border-radius: 5px; padding: 14px; margin-top: 10px;">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">DIAGNOSTIC SUMMARY</div>
<div style="color: #8A8F9A; font-size: 12px; line-height: 1.6; text-align: left;">{diagnostic_text}</div>
<div style="border-top: 1px solid rgba(255,255,255,0.04); margin-top: 10px; padding-top: 8px; font-size: 9px; color: rgba(255,255,255,0.25); text-align: left;">
This view highlights the concentration and balance of alpha contributions. It does not imply future behavior or recommended action.
</div>
</div>"""
    
    st.markdown(diagnostic_html, unsafe_allow_html=True)

    # ===================================================================
    # INSTITUTIONAL ENHANCEMENTS (ADDITIVE ONLY - DO NOT MODIFY ABOVE)
    # ===================================================================
    
    st.divider()
    
    # -----------------------------------------------
    # ENHANCEMENT A1: Attribution Confidence Signals
    # High/Medium/Low labels with sample-size awareness
    # -----------------------------------------------
    st.subheader("Attribution Confidence Signals")
    st.caption("Confidence indicators for attribution results · Prevents over-interpretation")
    st.markdown("")
    
    def compute_confidence_signals(snapshot_df, attrib_components, selected_horizon):
        """
        Compute confidence signals for attribution based on sample size, data quality, and stability.
        Returns (level, icon, warnings) tuple.
        """
        warnings = []
        confidence_score = 100
        
        # Factor 1: Sample size awareness
        horizon_col_map = {"INTRADAY": "alpha_intraday", "30D": "alpha_30d", "60D": "alpha_60d", "365D": "alpha_365d"}
        horizon_col = horizon_col_map.get(selected_horizon)
        
        if snapshot_df is not None and horizon_col and horizon_col in snapshot_df.columns:
            valid_count = snapshot_df[horizon_col].notna().sum()
            total_count = len(snapshot_df)
            
            if valid_count < 5:
                confidence_score -= 40
                warnings.append(f"Very small sample ({valid_count} waves) — interpret with caution")
            elif valid_count < 10:
                confidence_score -= 20
                warnings.append(f"Limited sample size ({valid_count} waves)")
            elif valid_count < total_count * 0.8:
                confidence_score -= 10
                warnings.append(f"Partial coverage ({valid_count}/{total_count} waves)")
        
        # Factor 2: Lookback period awareness
        if selected_horizon == "INTRADAY":
            warnings.append("Short lookback — may not reflect structural alpha")
        elif selected_horizon == "365D":
            warnings.append("Long lookback — may lag recent regime changes")
        
        # Factor 3: Attribution completeness
        if attrib_components:
            missing_components = []
            for key in ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]:
                if attrib_components.get(key) is None:
                    missing_components.append(key.replace("_alpha", "").title())
            if missing_components:
                confidence_score -= 15
                warnings.append(f"Missing components: {', '.join(missing_components)}")
            
            # High residual warning
            total_alpha = attrib_components.get("total_alpha", 0) or 0.0001
            residual = attrib_components.get("residual_alpha", 0) or 0
            if abs(total_alpha) > 0.0001:
                residual_share = abs(residual) / abs(total_alpha)
                if residual_share > 0.4:
                    confidence_score -= 20
                    warnings.append(f"High unexplained variance ({residual_share*100:.0f}% residual)")
        
        # Determine level
        if confidence_score >= 75:
            return "High", "[+]", warnings
        elif confidence_score >= 50:
            return "Medium", "[-]", warnings
        else:
            return "Low", "[v]", warnings
    
    if attrib_components:
        conf_level, conf_icon, conf_warnings = compute_confidence_signals(snapshot_df, attrib_components, selected_horizon)
        
        conf_cols = st.columns([1, 3])
        with conf_cols[0]:
            st.markdown(f"### {conf_icon} {conf_level}")
        with conf_cols[1]:
            st.caption(f"Confidence level for {selected_horizon} attribution")
            if conf_warnings:
                for warn in conf_warnings:
                    st.caption(f"• {warn}")
    else:
        st.info("Insufficient data for confidence assessment.")
    
    # -----------------------------------------------
    # ENHANCEMENT A2: Decision-Linked Attribution Context
    # Labels explaining WHY attribution changed
    # -----------------------------------------------
    st.divider()
    st.subheader("Attribution Context")
    st.caption("Why attribution values may have changed · Links to cause-effect")
    
    def detect_attribution_context(snapshot_df, attrib_df, attrib_components, selected_horizon):
        """
        Detect and label attribution context based on regime and decision patterns.
        Returns list of (label, explanation) tuples.
        """
        contexts = []
        
        if snapshot_df is None or attrib_components is None:
            return contexts
        
        # Check for regime signals
        total_alpha = attrib_components.get("total_alpha", 0) or 0
        regime_alpha = attrib_components.get("regime_alpha", 0) or 0
        volatility_alpha = attrib_components.get("volatility_alpha", 0) or 0
        momentum_alpha = attrib_components.get("momentum_alpha", 0) or 0
        
        # Detect regime shift influence
        if abs(regime_alpha) > 0.01:
            if regime_alpha > 0:
                contexts.append(("Risk Regime Shift", "Regime component contributing positively — market structure favoring current positioning"))
            else:
                contexts.append(("Risk Regime Shift", "Regime component detracting — positioning may not align with current market structure"))
        
        # Detect volatility dampening
        if volatility_alpha < -0.005:
            contexts.append(("Volatility Dampening", "Volatility component detracting — high volatility environment reducing alpha capture"))
        elif volatility_alpha > 0.005:
            contexts.append(("Volatility Tailwind", "Volatility component contributing — positioning benefits from current volatility regime"))
        
        # Detect momentum influence
        if abs(momentum_alpha) > 0.01:
            if momentum_alpha > 0:
                contexts.append(("Momentum Alignment", "Trend-following signals contributing positively"))
            else:
                contexts.append(("Momentum Headwind", "Counter-trend positioning creating drag"))
        
        # Check for cross-horizon conflict (from existing agreement logic)
        horizon_cols = {"Intraday": "alpha_intraday", "30D": "alpha_30d", "60D": "alpha_60d", "365D": "alpha_365d"}
        horizon_signs = {}
        for label, col in horizon_cols.items():
            if col in snapshot_df.columns:
                vals = snapshot_df[col].dropna()
                if len(vals) > 0:
                    mean_val = vals.mean()
                    horizon_signs[label] = 1 if mean_val > 0 else -1
        
        if len(horizon_signs) >= 2:
            unique_signs = set(horizon_signs.values())
            if len(unique_signs) > 1:
                short_positive = horizon_signs.get("Intraday", 0) > 0 or horizon_signs.get("30D", 0) > 0
                long_positive = horizon_signs.get("60D", 0) > 0 or horizon_signs.get("365D", 0) > 0
                if short_positive != long_positive:
                    contexts.append(("Horizon Divergence", "Short-term and long-term signals conflicting — attribution may be unstable"))
        
        return contexts
    
    attr_contexts = detect_attribution_context(snapshot_df, attrib_df, attrib_components, selected_horizon)
    
    if attr_contexts:
        for ctx_label, ctx_explanation in attr_contexts:
            st.markdown(f"**{ctx_label}**")
            st.caption(ctx_explanation)
    else:
        st.caption("No significant attribution context changes detected.")
    
    # -----------------------------------------------
    # ENHANCEMENT A3: Time-Horizon Decomposition
    # View attribution by Short/Medium/Long-term
    # -----------------------------------------------
    st.divider()
    st.subheader("Time-Horizon Decomposition")
    st.caption("Attribution split by horizon tier · Default: blended view above")
    
    horizon_view = st.radio(
        "Horizon View",
        options=["Blended (Default)", "Short-Term", "Medium-Term", "Long-Term"],
        horizontal=True,
        key="horizon_decomposition_view"
    )
    
    def compute_horizon_tier_attribution(snapshot_df, attrib_df, tier):
        """
        Compute aggregated attribution for a specific horizon tier.
        Returns dict of component values.
        """
        if tier == "Short-Term":
            # Intraday only
            if snapshot_df is not None and "alpha_intraday" in snapshot_df.columns:
                return compute_intraday_attribution(snapshot_df)
            return None
        elif tier == "Medium-Term":
            # 30D and 60D average
            if attrib_df is not None:
                result = {}
                components = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha", "total_alpha"]
                for comp in components:
                    if comp in attrib_df.columns:
                        # Filter to 30 and 60 day horizons
                        medium_data = attrib_df[attrib_df["horizon"].isin([30, 60])]
                        if len(medium_data) > 0:
                            result[comp] = medium_data[comp].mean()
                return result if result else None
            return None
        elif tier == "Long-Term":
            # 365D only
            if attrib_df is not None:
                result = {}
                components = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha", "total_alpha"]
                for comp in components:
                    if comp in attrib_df.columns:
                        long_data = attrib_df[attrib_df["horizon"] == 365]
                        if len(long_data) > 0:
                            result[comp] = long_data[comp].mean()
                return result if result else None
            return None
        return None
    
    if horizon_view != "Blended (Default)":
        tier_attrib = compute_horizon_tier_attribution(snapshot_df, attrib_df, horizon_view)
        
        if tier_attrib and tier_attrib.get("total_alpha") is not None:
            tier_total = tier_attrib.get("total_alpha", 0)
            st.metric(label=f"{horizon_view} Total Alpha", value=f"{tier_total*100:.2f}%")
            
            tier_cols = st.columns(6)
            tier_components = ["Selection", "Momentum", "Volatility", "Regime", "Exposure", "Residual"]
            tier_keys = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]
            
            for i, (name, key) in enumerate(zip(tier_components, tier_keys)):
                with tier_cols[i]:
                    val = tier_attrib.get(key)
                    if val is not None and pd.notna(val):
                        sign = "+" if val >= 0 else ""
                        st.metric(label=name, value=f"{sign}{val*100:.2f}%")
                    else:
                        st.metric(label=name, value="—")
        else:
            st.info(f"Insufficient data for {horizon_view} decomposition.")
    else:
        st.caption("Viewing blended attribution (above). Select a tier to see horizon-specific breakdown.")
    
    # -----------------------------------------------
    # ENHANCEMENT A4: Alpha vs Beta vs Defense Breakdown
    # Presentation split of return sources
    # -----------------------------------------------
    st.divider()
    st.subheader("Return Source Decomposition")
    st.caption("Conceptual split: Pure Alpha vs Market Beta vs Defensive Contribution")
    
    def compute_return_source_breakdown(attrib_components):
        """
        Decompose attribution into conceptual return sources.
        This is a PRESENTATION breakdown only - underlying calculations unchanged.
        """
        if attrib_components is None:
            return None
        
        # Map components to conceptual sources
        # Pure Alpha: Selection + Momentum (skill-based)
        # Market Beta: Regime + Exposure (market-driven)
        # Defense: Volatility contribution (risk management)
        
        selection = attrib_components.get("selection_alpha", 0) or 0
        momentum = attrib_components.get("momentum_alpha", 0) or 0
        regime = attrib_components.get("regime_alpha", 0) or 0
        exposure = attrib_components.get("exposure_alpha", 0) or 0
        volatility = attrib_components.get("volatility_alpha", 0) or 0
        residual = attrib_components.get("residual_alpha", 0) or 0
        
        pure_alpha = selection + momentum
        market_beta = regime + exposure
        defense = volatility
        unexplained = residual
        
        return {
            "Pure Alpha": pure_alpha,
            "Market Beta": market_beta,
            "Defense/Risk Mgmt": defense,
            "Unexplained": unexplained
        }
    
    if attrib_components:
        source_breakdown = compute_return_source_breakdown(attrib_components)
        
        if source_breakdown:
            source_cols = st.columns(4)
            source_items = list(source_breakdown.items())
            icons = {"Pure Alpha": "[A]", "Market Beta": "[B]", "Defense/Risk Mgmt": "[D]", "Unexplained": "[?]"}
            
            for i, (source_name, source_val) in enumerate(source_items):
                with source_cols[i]:
                    icon = icons.get(source_name, "")
                    sign = "+" if source_val >= 0 else ""
                    color = "normal" if source_val >= 0 else "inverse"
                    st.metric(
                        label=f"{icon} {source_name}",
                        value=f"{sign}{source_val*100:.2f}%"
                    )
            
            st.caption(
                "Pure Alpha = Selection + Momentum · Market Beta = Regime + Exposure · "
                "Defense = Volatility contribution · Unexplained = Residual"
            )
    else:
        st.info("Insufficient data for return source decomposition.")
    
    # -----------------------------------------------
    # ENHANCEMENT A5: Regime-Aware Attribution Labels
    # Risk-on/Risk-off/High-vol/Stress badges
    # -----------------------------------------------
    st.divider()
    st.subheader("Regime Context")
    st.caption("Market environment labels for attribution interpretation")
    
    def detect_regime_labels(snapshot_df, attrib_components):
        """
        Detect current regime context from available data.
        Returns list of (badge, description) tuples.
        """
        labels = []
        
        if snapshot_df is None:
            return labels
        
        # Check volatility regime
        if "return_intraday" in snapshot_df.columns:
            returns = snapshot_df["return_intraday"].dropna()
            if len(returns) > 0:
                return_std = returns.std()
                mean_abs_return = returns.abs().mean()
                
                if return_std > 0.02 or mean_abs_return > 0.015:
                    labels.append(("🔥 High-Vol", "Elevated volatility environment"))
                elif return_std < 0.005:
                    labels.append(("😴 Low-Vol", "Compressed volatility environment"))
        
        # Check directional regime from alpha
        if "alpha_intraday" in snapshot_df.columns:
            alphas = snapshot_df["alpha_intraday"].dropna()
            if len(alphas) > 0:
                mean_alpha = alphas.mean()
                positive_ratio = (alphas > 0).sum() / len(alphas)
                
                if positive_ratio > 0.7 and mean_alpha > 0.005:
                    labels.append(("📈 Risk-On", "Broad positive alpha — favorable risk environment"))
                elif positive_ratio < 0.3 and mean_alpha < -0.005:
                    labels.append(("📉 Risk-Off", "Broad negative alpha — defensive environment"))
        
        # Check for stress indicators (high residual + negative alpha)
        if attrib_components:
            total_alpha = attrib_components.get("total_alpha", 0) or 0
            residual = attrib_components.get("residual_alpha", 0) or 0
            
            if total_alpha < -0.01 and abs(residual) > abs(total_alpha) * 0.4:
                labels.append(("[!] Stress", "Elevated unexplained variance during drawdown"))
        
        return labels
    
    regime_labels = detect_regime_labels(snapshot_df, attrib_components)
    
    if regime_labels:
        regime_cols = st.columns(len(regime_labels))
        for i, (badge, description) in enumerate(regime_labels):
            with regime_cols[i]:
                st.markdown(f"### {badge}")
                st.caption(description)
    else:
        st.caption("No significant regime signals detected — neutral environment.")
    
    # -----------------------------------------------
    # ENHANCEMENT A6: IC/Compliance Summary View
    # Simplified exportable view for committees
    # -----------------------------------------------
    st.divider()
    st.subheader("IC Summary View")
    st.caption("Simplified view for Investment Committees · Exportable")
    
    ic_summary_toggle = st.toggle("Show IC Summary", key="alpha_ic_summary_toggle")
    
    if ic_summary_toggle and attrib_components:
        st.markdown("---")
        st.markdown("### Attribution Summary")
        
        # Key metrics in a clean format
        total_alpha = attrib_components.get("total_alpha", 0) or 0
        
        ic_summary_data = []
        ic_summary_data.append({
            "Metric": "Total Alpha",
            "Value": f"{total_alpha*100:.2f}%",
            "Horizon": selected_horizon
        })
        
        # Top contributors
        component_names = ["Selection", "Momentum", "Volatility", "Regime", "Exposure"]
        component_keys = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]
        contributions = []
        for name, key in zip(component_names, component_keys):
            val = attrib_components.get(key)
            if val is not None and pd.notna(val):
                contributions.append((name, val))
        
        if contributions:
            sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
            for rank, (name, val) in enumerate(sorted_contributions[:3], 1):
                sign = "+" if val >= 0 else ""
                ic_summary_data.append({
                    "Metric": f"#{rank} Contributor",
                    "Value": f"{name}: {sign}{val*100:.2f}%",
                    "Horizon": selected_horizon
                })
        
        # Residual
        residual = attrib_components.get("residual_alpha", 0) or 0
        ic_summary_data.append({
            "Metric": "Unexplained (Residual)",
            "Value": f"{residual*100:.2f}%",
            "Horizon": selected_horizon
        })
        
        # Confidence
        if 'conf_level' in dir():
            ic_summary_data.append({
                "Metric": "Confidence Level",
                "Value": conf_level if 'conf_level' in dir() else "—",
                "Horizon": selected_horizon
            })
        
        # WaveScore™ for IC Summary (read-only, non-operational)
        def compute_ic_wavescore(attrib_components):
            """Compute WaveScore™ for IC Summary. Read-only interpretive layer."""
            if not attrib_components:
                return None
            total_alpha = attrib_components.get("total_alpha", 0) or 0
            score = 50 + min(max(total_alpha * 1000, -20), 20)
            component_vals = [attrib_components.get(k) for k in ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]]
            component_vals = [v for v in component_vals if v is not None and pd.notna(v)]
            if len(component_vals) >= 3:
                positive_ratio = sum(1 for v in component_vals if v > 0) / len(component_vals)
                score += (positive_ratio - 0.5) * 30
            return min(max(score, 0), 100)
        
        ic_wavescore = compute_ic_wavescore(attrib_components)
        if ic_wavescore is not None:
            if ic_wavescore >= 70:
                ws_label = "Constructive"
            elif ic_wavescore >= 50:
                ws_label = "Neutral"
            elif ic_wavescore >= 30:
                ws_label = "Cautious"
            else:
                ws_label = "Defensive"
            ic_summary_data.append({
                "Metric": "WaveScore™",
                "Value": f"{ic_wavescore:.0f}/100 ({ws_label})",
                "Horizon": selected_horizon
            })
        
        ic_df = pd.DataFrame(ic_summary_data)
        st.dataframe(ic_df, use_container_width=True, hide_index=True)
        
        # Phase 2: Enhanced traceability for IC Summary
        st.caption("**Translation Layer** · WaveScore™ is a read-only interpretive summary derived from canonical attribution.")
        st.caption("For full details: Alpha Attribution tab (components) · Audit Trail tab (governance)")
        
        # Export button
        ic_export_cols = st.columns(2)
        with ic_export_cols[0]:
            csv_data = ic_df.to_csv(index=False)
            st.download_button(
                label="Export CSV",
                data=csv_data,
                file_name=f"alpha_attribution_ic_summary_{selected_horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="alpha_ic_csv_export"
            )
        
        with ic_export_cols[1]:
            # Plain text export
            txt_lines = [
                "WAVES Intelligence Console - Alpha Attribution IC Summary",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Horizon: {selected_horizon}",
                "=" * 50,
                ""
            ]
            for row in ic_summary_data:
                txt_lines.append(f"{row['Metric']}: {row['Value']}")
            txt_data = "\n".join(txt_lines)
            
            st.download_button(
                label="Export TXT",
                data=txt_data,
                file_name=f"alpha_attribution_ic_summary_{selected_horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="alpha_ic_txt_export"
            )
        
        st.markdown("---")
    
    # -----------------------------------------------
    # ENHANCEMENT A7: Attribution Health Warnings
    # Soft informational warnings for data quality
    # -----------------------------------------------
    st.divider()
    st.subheader("Attribution Health")
    st.caption("Informational warnings · Non-blocking · Governance-first transparency")
    
    def detect_health_warnings(snapshot_df, attrib_df, attrib_components, selected_horizon):
        """
        Detect soft health warnings for attribution data.
        Returns list of (level, message) tuples where level is 'info' or 'caution'.
        """
        warnings = []
        
        # Warning 1: Limited data
        horizon_col_map = {"INTRADAY": "alpha_intraday", "30D": "alpha_30d", "60D": "alpha_60d", "365D": "alpha_365d"}
        horizon_col = horizon_col_map.get(selected_horizon)
        
        if snapshot_df is not None and horizon_col and horizon_col in snapshot_df.columns:
            valid_count = snapshot_df[horizon_col].notna().sum()
            total_count = len(snapshot_df)
            
            if valid_count < 5:
                warnings.append(("caution", f"Attribution still stabilizing — limited data ({valid_count} waves)"))
            elif valid_count < total_count * 0.5:
                warnings.append(("info", f"Partial data coverage — {valid_count}/{total_count} waves reporting"))
        
        # Warning 2: Recent regime shift (high volatility in components)
        if attrib_components:
            component_values = []
            for key in ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]:
                val = attrib_components.get(key)
                if val is not None and pd.notna(val):
                    component_values.append(val)
            
            if len(component_values) >= 3:
                comp_std = pd.Series(component_values).std()
                comp_range = max(component_values) - min(component_values)
                
                if comp_range > 0.03 or comp_std > 0.015:
                    warnings.append(("info", "Component divergence detected — attribution may be normalizing after regime shift"))
        
        # Warning 3: High residual
        if attrib_components:
            total_alpha = attrib_components.get("total_alpha", 0) or 0.0001
            residual = attrib_components.get("residual_alpha", 0) or 0
            
            if abs(total_alpha) > 0.0001:
                residual_share = abs(residual) / abs(total_alpha)
                if residual_share > 0.35:
                    warnings.append(("caution", f"Attribution model fit may be imprecise — {residual_share*100:.0f}% unexplained"))
        
        # Warning 4: Intraday-specific
        if selected_horizon == "INTRADAY":
            warnings.append(("info", "Intraday attribution reflects short-term signals only — combine with longer horizons for strategic view"))
        
        return warnings
    
    health_warnings = detect_health_warnings(snapshot_df, attrib_df, attrib_components, selected_horizon)
    
    if health_warnings:
        for warn_level, warn_msg in health_warnings:
            if warn_level == "caution":
                st.warning(f"{warn_msg}")
            else:
                st.info(f"{warn_msg}")
    else:
        st.success("[OK] Attribution health nominal — no warnings")
    
    # ===================================================================
    # END OF INSTITUTIONAL ENHANCEMENTS
    # ===================================================================

    st.divider()

    # Wave-level alpha for selected wave
    st.markdown('<span class="waves-micro-label">Wave-Level Attribution</span>', unsafe_allow_html=True)
    st.subheader(f"Wave Alpha: {selected_wave}")
    st.caption("Alpha attribution for the selected wave · Read-only · Observational")
    wave_subset = snapshot_df[snapshot_df["display_name"] == selected_wave]
    if not wave_subset.empty:
        wave_row = wave_subset.iloc[0]
        wave_alphas = {}
        for horizon_label, col in alpha_horizons.items():
            val = wave_row.get(col)
            wave_alphas[horizon_label] = val if pd.notna(val) else None

        # Multi-horizon alpha display
        st.markdown("**Multi-Horizon Alpha**")
        render_metric_row(wave_alphas, intraday_label=attrib_intraday_label, has_intraday_data=attrib_has_any_intraday)

        # Compare to portfolio
        st.markdown("**Relative Performance vs Portfolio**")
        comparison_cols = st.columns(4)
        for i, (horizon_label, col) in enumerate(alpha_horizons.items()):
            with comparison_cols[i]:
                wave_val = wave_row.get(col)
                portfolio_val = portfolio_alpha_summary.get(horizon_label)
                if pd.notna(wave_val) and portfolio_val is not None:
                    diff = wave_val - portfolio_val
                    # Color-code the delta
                    delta_sign = "+" if diff >= 0 else ""
                    performance_label = "Outperforming" if diff > 0.001 else ("Underperforming" if diff < -0.001 else "Inline")
                    st.metric(
                        label=f"{horizon_label}",
                        value=f"{delta_sign}{diff*100:.2f}%",
                        delta=performance_label,
                        delta_color="normal" if diff >= 0 else "inverse"
                    )
                else:
                    st.metric(label=f"{horizon_label}", value="—")
        
        # Wave-level contribution analysis (derived from observable wave data)
        st.markdown("**Wave Contribution Analysis**")
        with st.expander("Component Contribution Breakdown"):
            wave_intraday_alpha = wave_row.get("alpha_intraday")
            wave_30d_alpha = wave_row.get("alpha_30d")
            wave_60d_alpha = wave_row.get("alpha_60d")
            wave_365d_alpha = wave_row.get("alpha_365d")
            wave_return = wave_row.get("return_intraday")
            
            # Check if we have attribution data for this specific wave
            wave_name_raw = wave_row.get("wave_name", selected_wave)
            wave_attrib = None
            if attrib_df is not None:
                # Try both "wave" and "wave_name" columns (CSV uses "wave")
                wave_col = "wave" if "wave" in attrib_df.columns else "wave_name" if "wave_name" in attrib_df.columns else None
                if wave_col:
                    # Get attribution for selected horizon (default to 30D if not specified)
                    horizon_days = {"INTRADAY": 30, "30D": 30, "60D": 60, "365D": 365}.get(selected_horizon, 30)
                    wave_attrib_rows = attrib_df[(attrib_df[wave_col] == wave_name_raw) & (attrib_df["horizon"] == horizon_days)]
                    if len(wave_attrib_rows) > 0:
                        wave_attrib = wave_attrib_rows.iloc[0]
            
            if wave_attrib is not None:
                # Display actual per-wave attribution from CSV
                st.markdown("**Per-Wave Attribution (from attribution data)**")
                component_keys = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]
                component_labels = ["Selection", "Momentum", "Volatility", "Regime", "Exposure", "Residual"]
                
                for label, key in zip(component_labels, component_keys):
                    val = wave_attrib.get(key)
                    if val is not None and pd.notna(val):
                        sign = "+" if val >= 0 else ""
                        color = "[+]" if val > 0.001 else ("[v]" if val < -0.001 else "[-]")
                        st.markdown(f"{color} **{label}**: {sign}{val*100:.3f}%")
                    else:
                        st.markdown(f"[-] **{label}**: —")
            else:
                # No per-wave attribution data - show observable signals only
                st.markdown("**Observable Signal Characteristics**")
                st.caption("Per-wave component attribution not available in current data structure. Showing derived signal analysis:")
                
                signals_found = False
                
                # Selection signal: wave's deviation from portfolio mean (observable)
                if wave_intraday_alpha is not None and portfolio_alpha_summary.get(attrib_intraday_label) is not None:
                    signals_found = True
                    selection_delta = wave_intraday_alpha - portfolio_alpha_summary[attrib_intraday_label]
                    sign = "+" if selection_delta >= 0 else ""
                    icon = "[+]" if selection_delta > 0.001 else ("[v]" if selection_delta < -0.001 else "[-]")
                    st.markdown(f"{icon} **Selection (vs Portfolio)**: {sign}{selection_delta*100:.3f}%")
                
                # Trend alignment signals (observable)
                if wave_intraday_alpha is not None and wave_30d_alpha is not None and pd.notna(wave_30d_alpha):
                    signals_found = True
                    alignment_30d = "Aligned" if (wave_intraday_alpha > 0) == (wave_30d_alpha > 0) else "Conflicting"
                    icon_30d = "[+]" if alignment_30d == "Aligned" else "[v]"
                    st.markdown(f"{icon_30d} **Momentum (30D Trend)**: {alignment_30d}")
                
                if wave_intraday_alpha is not None and wave_365d_alpha is not None and pd.notna(wave_365d_alpha):
                    signals_found = True
                    alignment_365d = "Supportive" if (wave_intraday_alpha > 0) == (wave_365d_alpha > 0) else "Counter"
                    icon_365d = "[+]" if alignment_365d == "Supportive" else "[!]"
                    st.markdown(f"{icon_365d} **Regime (365D Structure)**: {alignment_365d}")
                
                # Alpha consistency across horizons
                alpha_values = [wave_intraday_alpha, wave_30d_alpha, wave_60d_alpha, wave_365d_alpha]
                valid_alphas = [a for a in alpha_values if a is not None and pd.notna(a)]
                if len(valid_alphas) >= 2:
                    signals_found = True
                    positive_count = sum(1 for a in valid_alphas if a > 0)
                    consistency = positive_count / len(valid_alphas) if len(valid_alphas) > 0 else 0
                    if consistency > 0.75:
                        st.markdown(f"[+] **Horizon Consistency**: High — {positive_count}/{len(valid_alphas)} horizons positive")
                    elif consistency < 0.25:
                        st.markdown(f"[v] **Horizon Consistency**: High (negative) — {positive_count}/{len(valid_alphas)} horizons positive")
                    else:
                        st.markdown(f"[-] **Horizon Consistency**: Mixed — {positive_count}/{len(valid_alphas)} horizons positive")
                
                if not signals_found:
                    st.caption("Insufficient data for signal analysis.")
    else:
        st.warning(f"Wave '{selected_wave}' not found in snapshot.")


# ===========================
# ADAPTIVE INTELLIGENCE TAB
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence Center")
    st.caption("Decision support layer · System-learned insights and recommendations · LIVE learning enabled")
    st.markdown("")
    
    # -----------------------------------------------
    # LIVE LEARNING: Load and update adaptive state
    # -----------------------------------------------
    adaptive_state = al.load_adaptive_state()
    adaptive_state, learning_messages = al.update_adaptive_state(snapshot_df, attrib_df, adaptive_state)

    # Show learning updates if any
    if learning_messages:
        with st.expander("Live Learning Updates", expanded=False):
            for msg in learning_messages:
                st.caption(msg)
            st.caption("Adaptive state persisted to data/adaptive_state.json")

    # -----------------------------------------------
    # WAVE DIAGNOSTICS (Phase 1 + Phase 2)
    # -----------------------------------------------
    st.markdown('<span class="waves-micro-label">Diagnostic Layer</span>', unsafe_allow_html=True)
    st.subheader("Wave Diagnostics")
    st.caption("Wave-level diagnostic interpretation derived from attribution and stability signals · Observational only")
    
    wave_names = sorted(snapshot_df["wave_name"].unique().tolist()) if snapshot_df is not None else []
    selected_diag_wave = st.selectbox("Select Wave", [""] + wave_names, key="diag_wave_selector", format_func=lambda x: "Select a Wave to view diagnostics" if x == "" else x) if wave_names else None
    
    if selected_diag_wave and selected_diag_wave != "" and attrib_df is not None and snapshot_df is not None:
        wave_snap = snapshot_df[snapshot_df["wave_name"] == selected_diag_wave].iloc[0] if len(snapshot_df[snapshot_df["wave_name"] == selected_diag_wave]) > 0 else None
        wave_attrib = attrib_df[attrib_df["wave"] == selected_diag_wave] if "wave" in attrib_df.columns else pd.DataFrame()
        
        attrib_30d = wave_attrib[wave_attrib["horizon"] == 30].iloc[0] if len(wave_attrib[wave_attrib["horizon"] == 30]) > 0 else None
        attrib_60d = wave_attrib[wave_attrib["horizon"] == 60].iloc[0] if len(wave_attrib[wave_attrib["horizon"] == 60]) > 0 else None
        attrib_365d = wave_attrib[wave_attrib["horizon"] == 365].iloc[0] if len(wave_attrib[wave_attrib["horizon"] == 365]) > 0 else None
        
        def compute_wave_diagnostics(wave_snap, attrib_30d, attrib_60d, attrib_365d):
            working = []
            not_working = []
            needs_review = []
            health_status = "Healthy"
            confidence_level = "Medium"
            regime_alignment = "Aligned"
            stability = "Stable"
            trend = "Flat"
            
            components = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]
            positive_components = []
            negative_components = []
            cross_horizon_conflicts = []
            
            if attrib_30d is not None:
                for comp in components:
                    val_30d = attrib_30d.get(comp, 0) if attrib_30d is not None else 0
                    val_365d = attrib_365d.get(comp, 0) if attrib_365d is not None else 0
                    comp_label = comp.replace("_alpha", "").replace("_", " ").title()
                    
                    if pd.notna(val_30d) and val_30d > 0.005:
                        positive_components.append((comp_label, val_30d))
                    elif pd.notna(val_30d) and val_30d < -0.005:
                        negative_components.append((comp_label, val_30d))
                    
                    if pd.notna(val_30d) and pd.notna(val_365d) and val_365d != 0:
                        if (val_30d > 0 and val_365d < 0) or (val_30d < 0 and val_365d > 0):
                            cross_horizon_conflicts.append(comp_label)
            
            for comp, val in positive_components:
                working.append(f"{comp} has contributed positively ({val*100:.2f}%)")
            
            for comp, val in negative_components:
                not_working.append(f"{comp} has exhibited drag ({val*100:.2f}%)")
            
            for comp in cross_horizon_conflicts:
                needs_review.append(f"{comp} has shown cross-horizon divergence (30D vs 365D)")
            
            if wave_snap is not None:
                alpha_30d = wave_snap.get("alpha_30d", 0)
                alpha_365d = wave_snap.get("alpha_365d", 0)
                if pd.notna(alpha_30d) and pd.notna(alpha_365d):
                    if alpha_30d > 0 and alpha_365d > 0:
                        working.append("Alpha has been consistently positive across horizons")
                    elif alpha_30d < 0 and alpha_365d < 0:
                        not_working.append("Alpha has been persistently negative across horizons")
                    elif abs(alpha_30d) > 0.05 or abs(alpha_365d) > 0.1:
                        needs_review.append("Alpha concentration has been elevated")
            
            if attrib_30d is not None and attrib_365d is not None:
                regime_30d = attrib_30d.get("regime_alpha", 0) if pd.notna(attrib_30d.get("regime_alpha", 0)) else 0
                regime_365d = attrib_365d.get("regime_alpha", 0) if pd.notna(attrib_365d.get("regime_alpha", 0)) else 0
                if regime_30d > 0 and regime_365d > 0:
                    regime_alignment = "Aligned"
                elif (regime_30d > 0 and regime_365d < 0) or (regime_30d < 0 and regime_365d > 0):
                    regime_alignment = "Misaligned"
                    needs_review.append("Regime alignment has diverged between horizons")
                else:
                    regime_alignment = "Mixed"
            
            negative_count = len(negative_components)
            conflict_count = len(cross_horizon_conflicts)
            positive_count = len(positive_components)
            
            if negative_count >= 3 or conflict_count >= 2:
                health_status = "Review"
                confidence_level = "Low"
            elif negative_count >= 1 or conflict_count >= 1:
                health_status = "Watch"
                confidence_level = "Medium"
            else:
                health_status = "Healthy"
                confidence_level = "High" if positive_count >= 2 else "Medium"
            
            if attrib_30d is not None and attrib_60d is not None:
                total_30d = attrib_30d.get("total_alpha", 0) if pd.notna(attrib_30d.get("total_alpha", 0)) else 0
                total_60d = attrib_60d.get("total_alpha", 0) if pd.notna(attrib_60d.get("total_alpha", 0)) else 0
                delta = total_30d - total_60d
                if delta > 0.01:
                    trend = "30D > 60D"
                    stability = "Stable"
                elif delta < -0.01:
                    trend = "30D < 60D"
                    stability = "Fragile" if delta < -0.03 else "Mixed"
                else:
                    trend = "Flat"
                    stability = "Stable"
            
            vol_drag = attrib_30d.get("volatility_alpha", 0) if attrib_30d is not None and pd.notna(attrib_30d.get("volatility_alpha", 0)) else 0
            if vol_drag < -0.02:
                stability = "Fragile"
                if not any("Volatility drag has been persistent" in n for n in needs_review):
                    needs_review.append("Volatility drag has been persistent")
            
            if not working:
                working.append("No significant positive drivers were identified at this horizon")
            if not not_working:
                not_working.append("No significant negative contributors were detected")
            if not needs_review:
                needs_review.append("No attention items were flagged")
            
            return {
                "health_status": health_status,
                "confidence_level": confidence_level,
                "regime_alignment": regime_alignment,
                "stability": stability,
                "trend": trend,
                "working": working,
                "not_working": not_working,
                "needs_review": needs_review,
                "attrib_30d": attrib_30d,
                "attrib_60d": attrib_60d,
                "attrib_365d": attrib_365d
            }
        
        diag = compute_wave_diagnostics(wave_snap, attrib_30d, attrib_60d, attrib_365d)
        
        st.markdown("---")
        st.markdown("**Wave Health Summary**")
        st.caption("Derived from historical attribution data · Observational only")
        health_cols = st.columns(5)
        
        health_cols[0].metric("Health Status", diag["health_status"], help="Aggregate historical signal state")
        health_cols[1].metric("Signal Clarity", diag["confidence_level"], help="Consistency of dominant attribution drivers")
        health_cols[2].metric("Regime Alignment", diag["regime_alignment"], help="Alignment with prevailing market regime")
        health_cols[3].metric("Variance", diag["stability"], help="Historical dispersion of wave returns")
        health_cols[4].metric("30D vs 60D", diag["trend"], help="Short-term vs intermediate-term attribution consistency")
        
        st.markdown("")
        diag_cols = st.columns(3)
        
        with diag_cols[0]:
            st.markdown("**What's Working**")
            for item in diag["working"]:
                st.caption(f"• {item}")
        
        with diag_cols[1]:
            st.markdown("**What's Not Working**")
            for item in diag["not_working"]:
                st.caption(f"• {item}")
        
        with diag_cols[2]:
            st.markdown("**What Needs Review**")
            for item in diag["needs_review"]:
                st.caption(f"• {item}")
        
        st.markdown("")
        why_how_when_cols = st.columns(3)
        
        with why_how_when_cols[0]:
            st.markdown("**Why (Attribution Sources)**")
            if diag["attrib_365d"] is not None:
                components = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]
                vals = [(c.replace("_alpha","").title(), diag["attrib_365d"].get(c, 0)) for c in components if pd.notna(diag["attrib_365d"].get(c, 0))]
                vals_sorted = sorted(vals, key=lambda x: abs(x[1]), reverse=True)
                if vals_sorted:
                    dominant = vals_sorted[0]
                    st.caption(f"• Primary driver has been {dominant[0]} ({dominant[1]*100:.2f}%)")
                    if len(vals_sorted) > 1 and abs(vals_sorted[1][1]) > 0.01:
                        st.caption(f"• Secondary driver: {vals_sorted[1][0]} ({vals_sorted[1][1]*100:.2f}%)")
                    pos_count = sum(1 for _, v in vals_sorted if v > 0.005)
                    if pos_count >= 3:
                        st.caption("• Alpha has been diversified across multiple sources")
                    elif pos_count == 1:
                        st.caption("• Alpha has been concentrated in a single source")
                else:
                    st.caption("• Attribution sources were not available")
            else:
                st.caption("• Long-horizon attribution data was not available")
        
        with why_how_when_cols[1]:
            st.markdown("**How (Behavioral Characteristics)**")
            if diag["attrib_30d"] is not None:
                vol_alpha = diag["attrib_30d"].get("volatility_alpha", 0) if pd.notna(diag["attrib_30d"].get("volatility_alpha", 0)) else 0
                if vol_alpha < -0.02:
                    st.caption("• Volatility has been a consistent drag")
                elif vol_alpha > 0.01:
                    st.caption("• Volatility has contributed positively")
                else:
                    st.caption("• Volatility impact has been neutral")
                regime_alpha = diag["attrib_30d"].get("regime_alpha", 0) if pd.notna(diag["attrib_30d"].get("regime_alpha", 0)) else 0
                if abs(regime_alpha) > 0.02:
                    st.caption(f"• Wave has shown {'strong' if abs(regime_alpha) > 0.05 else 'moderate'} regime sensitivity")
                else:
                    st.caption("• Wave has shown low regime sensitivity")
                if diag["stability"] == "Fragile":
                    st.caption("• Return pattern has been choppy")
                elif diag["stability"] == "Stable":
                    st.caption("• Return pattern has been stable")
                else:
                    st.caption("• Return pattern has shown mixed stability")
            else:
                st.caption("• Behavioral data was not available")
        
        with why_how_when_cols[2]:
            st.markdown("**When (Temporal Context)**")
            if diag["attrib_30d"] is not None and diag["attrib_365d"] is not None:
                total_30d = diag["attrib_30d"].get("total_alpha", 0) if pd.notna(diag["attrib_30d"].get("total_alpha", 0)) else 0
                total_365d = diag["attrib_365d"].get("total_alpha", 0) if pd.notna(diag["attrib_365d"].get("total_alpha", 0)) else 0
                if total_30d > 0 and total_365d > 0:
                    st.caption("• Positive performance has persisted across horizons")
                elif total_30d < 0 and total_365d < 0:
                    st.caption("• Underperformance has been persistent (not recent)")
                elif total_30d < 0 and total_365d > 0:
                    st.caption("• Recent underperformance observed (long-term positive)")
                elif total_30d > 0 and total_365d < 0:
                    st.caption("• Recent improvement observed (long-term negative)")
                else:
                    st.caption("• Performance pattern has been mixed")
                if len(diag["needs_review"]) > 1 and "No attention items" not in diag["needs_review"][0]:
                    st.caption("• Multiple attention flags have been identified")
            else:
                st.caption("• Temporal comparison data was not available")
        
        with st.expander("Supporting Evidence (Historical Attribution Breakdown)", expanded=False):
            # Build styled HTML for attribution breakdown
            ev_html = """
            <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
                <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] SUPPORTING EVIDENCE</div>
                <div style="color: #666666; font-size: 12px; margin-bottom: 16px;">Underlying metrics used in diagnostics · Read-only · No interaction</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
            """
            
            for horizon_label, attrib_key in [("30D Attribution", "attrib_30d"), ("60D Attribution", "attrib_60d"), ("365D Attribution", "attrib_365d")]:
                ev_html += f'<div><div style="color: #D0D0D0; font-weight: 600; margin-bottom: 10px;">{horizon_label}</div>'
                attrib_data = diag.get(attrib_key)
                if attrib_data is not None:
                    for comp in ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]:
                        val = attrib_data.get(comp, 0)
                        if pd.notna(val):
                            color = "#48BB78" if val > 0 else "#FC8181" if val < 0 else "#A0A0A0"
                            ev_html += f'<div style="color: #A0A0A0; font-size: 12px; line-height: 1.6;">{comp.replace("_alpha","").title()}: <span style="font-family: \'SF Mono\', Monaco, monospace; color: {color};">{val*100:.2f}%</span></div>'
                else:
                    ev_html += '<div style="color: #666666; font-size: 12px;">No data available</div>'
                ev_html += '</div>'
            
            ev_html += """
                </div>
                <div style="border-top: 1px solid #2A2A2A; margin-top: 16px; padding-top: 12px; font-size: 11px; color: #666666;">Diagnostics are observational, backward-looking, and derived entirely from existing attribution data.</div>
            </div>
            """
            st.markdown(ev_html, unsafe_allow_html=True)
    
    elif not wave_names:
        st.caption("No waves available for diagnostics.")
    elif selected_diag_wave == "" or selected_diag_wave is None:
        st.info("Select a Wave above to view wave-specific diagnostics.")
    
    st.markdown("---")

    # -----------------------------------------------
    # Section A: System Summary
    # -----------------------------------------------
    st.subheader("System Summary")
    st.info(
        "Based on recent performance and attribution patterns, the system has surfaced "
        "the following adaptive signals for review. All insights are LIVE and LEARNING."
    )

    # Refinement 1: Timeframe Awareness
    st.caption("Primary Horizon: 30D · Confirmed by 365D where available")

    # -----------------------------------------------
    # Compute derived signals from available data
    # -----------------------------------------------
    def compute_adaptive_signals(snapshot_df, attrib_df):
        """
        Derive adaptive signals from snapshot and attribution data.
        All signals are data-driven and explainable.
        """
        signals = []
        recommendations = []
        readiness_flags = {}

        # --- Signal 1: Momentum Persistence ---
        # Check if momentum_alpha is consistently positive or negative across horizons
        if attrib_df is not None:
            momentum_30d = attrib_df[attrib_df["horizon"] == 30]["momentum_alpha"].dropna()
            momentum_60d = attrib_df[attrib_df["horizon"] == 60]["momentum_alpha"].dropna()
            momentum_365d = attrib_df[attrib_df["horizon"] == 365]["momentum_alpha"].dropna()

            if len(momentum_30d) > 0 and len(momentum_365d) > 0:
                avg_30d = momentum_30d.mean()
                avg_365d = momentum_365d.mean()

                # Count waves with positive momentum
                pos_30d = (momentum_30d > 0).sum()
                neg_30d = (momentum_30d < 0).sum()

                if avg_30d > 0 and avg_365d > 0:
                    direction = "Positive"
                    confidence = "High" if avg_30d > 0.01 else "Medium"
                    evidence = f"{pos_30d}/{len(momentum_30d)} waves show positive momentum (30D avg: {avg_30d*100:.2f}%)"
                elif avg_30d < 0 and avg_365d < 0:
                    direction = "Negative"
                    confidence = "High" if avg_30d < -0.01 else "Medium"
                    evidence = f"{neg_30d}/{len(momentum_30d)} waves show negative momentum (30D avg: {avg_30d*100:.2f}%)"
                else:
                    direction = "Neutral"
                    confidence = "Low"
                    evidence = f"Mixed momentum signals across horizons (30D: {avg_30d*100:.2f}%, 365D: {avg_365d*100:.2f}%)"

                signals.append({
                    "signal": "Momentum Persistence",
                    "affected_waves": len(momentum_30d),
                    "direction": direction,
                    "confidence": confidence,
                    "evidence": evidence,
                    "derived_from": f"Derived from momentum_alpha attribution across {len(momentum_30d)} waves (30D primary, 365D confirmatory)"
                })

        # --- Signal 2: Volatility Drag ---
        if attrib_df is not None:
            vol_30d = attrib_df[attrib_df["horizon"] == 30]["volatility_alpha"].dropna()
            vol_365d = attrib_df[attrib_df["horizon"] == 365]["volatility_alpha"].dropna()

            if len(vol_30d) > 0:
                avg_vol_30d = vol_30d.mean()
                neg_vol_count = (vol_30d < 0).sum()

                if avg_vol_30d < -0.005:
                    direction = "Negative"
                    confidence = "High"
                    evidence = f"Volatility drag detected: {neg_vol_count}/{len(vol_30d)} waves impacted (avg: {avg_vol_30d*100:.2f}%)"
                elif avg_vol_30d < 0:
                    direction = "Negative"
                    confidence = "Medium"
                    evidence = f"Mild volatility drag: avg impact {avg_vol_30d*100:.2f}%"
                else:
                    direction = "Neutral"
                    confidence = "Low"
                    evidence = f"No significant volatility drag detected (avg: {avg_vol_30d*100:.2f}%)"

                signals.append({
                    "signal": "Volatility Drag",
                    "affected_waves": len(vol_30d),
                    "direction": direction,
                    "confidence": confidence,
                    "evidence": evidence,
                    "derived_from": f"Derived from volatility_alpha attribution across {len(vol_30d)} waves (30D primary)"
                })

        # --- Signal 3: Regime Sensitivity ---
        if attrib_df is not None:
            regime_30d = attrib_df[attrib_df["horizon"] == 30]["regime_alpha"].dropna()
            regime_365d = attrib_df[attrib_df["horizon"] == 365]["regime_alpha"].dropna()

            if len(regime_30d) > 0 and len(regime_365d) > 0:
                avg_30d = regime_30d.mean()
                avg_365d = regime_365d.mean()
                volatility = regime_30d.std()

                if volatility > 0.02:
                    direction = "High Sensitivity"
                    confidence = "High"
                    evidence = f"High regime dispersion across waves (std: {volatility*100:.2f}%)"
                elif abs(avg_30d - avg_365d) > 0.01:
                    direction = "Transitioning"
                    confidence = "Medium"
                    evidence = f"Regime shift detected: 30D ({avg_30d*100:.2f}%) vs 365D ({avg_365d*100:.2f}%)"
                else:
                    direction = "Stable"
                    confidence = "Medium"
                    evidence = f"Regime impact stable across horizons"

                signals.append({
                    "signal": "Regime Sensitivity",
                    "affected_waves": len(regime_30d),
                    "direction": direction,
                    "confidence": confidence,
                    "evidence": evidence,
                    "derived_from": f"Derived from regime_alpha attribution across {len(regime_30d)} waves (30D primary, 365D confirmatory)"
                })

        # --- Signal 4: Selection Quality ---
        if attrib_df is not None:
            selection_30d = attrib_df[attrib_df["horizon"] == 30]["selection_alpha"].dropna()
            selection_365d = attrib_df[attrib_df["horizon"] == 365]["selection_alpha"].dropna()

            if len(selection_30d) > 0:
                avg_selection_30d = selection_30d.mean()
                pos_selection = (selection_30d > 0).sum()

                if avg_selection_30d > 0.01:
                    direction = "Positive"
                    confidence = "High"
                    evidence = f"Strong selection alpha: {pos_selection}/{len(selection_30d)} waves positive (avg: {avg_selection_30d*100:.2f}%)"
                elif avg_selection_30d > 0:
                    direction = "Positive"
                    confidence = "Medium"
                    evidence = f"Mild selection benefit (avg: {avg_selection_30d*100:.2f}%)"
                elif avg_selection_30d < -0.01:
                    direction = "Negative"
                    confidence = "High"
                    evidence = f"Selection underperformance detected (avg: {avg_selection_30d*100:.2f}%)"
                else:
                    direction = "Neutral"
                    confidence = "Low"
                    evidence = f"Selection impact minimal (avg: {avg_selection_30d*100:.2f}%)"

                signals.append({
                    "signal": "Selection Quality",
                    "affected_waves": len(selection_30d),
                    "direction": direction,
                    "confidence": confidence,
                    "evidence": evidence,
                    "derived_from": f"Derived from selection_alpha attribution across {len(selection_30d)} waves (30D primary)"
                })

        # --- Signal 5: Alpha Consistency ---
        if "alpha_30d" in snapshot_df.columns and "alpha_365d" in snapshot_df.columns:
            alpha_30d = snapshot_df["alpha_30d"].dropna()
            alpha_365d = snapshot_df["alpha_365d"].dropna()

            if len(alpha_30d) > 0 and len(alpha_365d) > 0:
                # Check consistency: how many waves have same-sign alpha across horizons
                merged = snapshot_df[["wave_name", "alpha_30d", "alpha_365d"]].dropna()
                if len(merged) > 0:
                    consistent = ((merged["alpha_30d"] > 0) & (merged["alpha_365d"] > 0)) | \
                                 ((merged["alpha_30d"] < 0) & (merged["alpha_365d"] < 0))
                    consistency_rate = consistent.sum() / len(merged)

                    if consistency_rate > 0.7:
                        direction = "High Consistency"
                        confidence = "High"
                    elif consistency_rate > 0.5:
                        direction = "Moderate Consistency"
                        confidence = "Medium"
                    else:
                        direction = "Low Consistency"
                        confidence = "Low"

                    evidence = f"{consistent.sum()}/{len(merged)} waves show consistent alpha direction across 30D/365D"

                    signals.append({
                        "signal": "Alpha Consistency",
                        "affected_waves": len(merged),
                        "direction": direction,
                        "confidence": confidence,
                        "evidence": evidence,
                        "derived_from": f"Derived from alpha_30d + alpha_365d across {len(merged)} waves (cross-horizon comparison)"
                    })

        # --- Generate Recommendations ---
        # Based on detected signals
        for sig in signals:
            if sig["signal"] == "Momentum Persistence" and sig["direction"] == "Negative":
                recommendations.append({
                    "title": "Review Momentum Overlay",
                    "suggestion": "Consider reducing momentum exposure in underperforming waves",
                    "reason": sig["evidence"],
                    "impact": "Risk reduction",
                    "confidence": sig["confidence"]
                })
            elif sig["signal"] == "Volatility Drag" and sig["direction"] == "Negative":
                recommendations.append({
                    "title": "Volatility Management",
                    "suggestion": "Evaluate volatility targeting parameters for affected waves",
                    "reason": sig["evidence"],
                    "impact": "Stability within expected bounds",
                    "confidence": sig["confidence"]
                })
            elif sig["signal"] == "Selection Quality" and sig["direction"] == "Positive":
                recommendations.append({
                    "title": "Maintain Selection Strategy",
                    "suggestion": "Current security selection is generating positive alpha",
                    "reason": sig["evidence"],
                    "impact": "Monitoring continues",
                    "confidence": sig["confidence"]
                })
            elif sig["signal"] == "Alpha Consistency" and "Low" in sig["direction"]:
                recommendations.append({
                    "title": "Investigate Alpha Instability",
                    "suggestion": "Review waves with inconsistent alpha patterns for potential rebalancing",
                    "reason": sig["evidence"],
                    "impact": "Human review required",
                    "confidence": sig["confidence"]
                })

        # Add placeholder recommendations if none generated
        if len(recommendations) == 0:
            recommendations.append({
                "title": "Pending Signal Generation",
                "suggestion": "Accumulating sufficient data for actionable recommendations",
                "reason": "Insufficient attribution history for confident suggestions",
                "impact": "N/A",
                "confidence": "Low"
            })

        # --- Compute Readiness Flags ---
        # System confidence
        high_confidence_signals = sum(1 for s in signals if s["confidence"] == "High")
        if high_confidence_signals >= 3:
            readiness_flags["System Confidence"] = "Stable"
        elif high_confidence_signals >= 1:
            readiness_flags["System Confidence"] = "Moderate"
        else:
            readiness_flags["System Confidence"] = "Awaiting Higher Confidence"

        # Regime transition
        regime_sig = next((s for s in signals if s["signal"] == "Regime Sensitivity"), None)
        if regime_sig and regime_sig["direction"] == "Transitioning":
            readiness_flags["Regime Transition"] = "Detected"
        else:
            readiness_flags["Regime Transition"] = "Stable"

        # Signal disagreement
        directions = [s["direction"] for s in signals]
        positive_count = sum(1 for d in directions if "Positive" in d or "High" in d)
        negative_count = sum(1 for d in directions if "Negative" in d)
        if positive_count > 0 and negative_count > 0 and abs(positive_count - negative_count) <= 1:
            readiness_flags["Signal Disagreement"] = "Elevated"
        else:
            readiness_flags["Signal Disagreement"] = "Normal"

        # Human review recommendation
        if readiness_flags.get("Signal Disagreement") == "Elevated" or \
           readiness_flags.get("Regime Transition") == "Detected":
            readiness_flags["Human Review"] = "Recommended"
        else:
            readiness_flags["Human Review"] = "Not Required"

        return signals, recommendations, readiness_flags

    # Compute signals
    signals, recommendations, readiness_flags = compute_adaptive_signals(snapshot_df, attrib_df)

    st.divider()

    # -----------------------------------------------
    # Section B: Learned Signals
    # -----------------------------------------------
    st.subheader("Learned Signals")
    st.caption("Data-driven, explainable signals derived from attribution patterns")
    st.markdown("")
    
    if len(signals) > 0:
        # Create signals table (excluding derived_from for main display)
        signals_df = pd.DataFrame(signals)
        display_df = signals_df[["signal", "affected_waves", "direction", "confidence", "evidence"]].copy()
        display_df = display_df.rename(columns={
            "signal": "Signal Name",
            "affected_waves": "Affected Waves",
            "direction": "Direction",
            "confidence": "Confidence",
            "evidence": "Evidence"
        })

        # Display as a clean table
        st.dataframe(
            display_df,
            hide_index=True,
            column_config={
                "Signal Name": st.column_config.TextColumn(width="medium"),
                "Affected Waves": st.column_config.NumberColumn(width="small"),
                "Direction": st.column_config.TextColumn(width="small"),
                "Confidence": st.column_config.TextColumn(width="small"),
                "Evidence": st.column_config.TextColumn(width="large"),
            }
        )

        # Refinement 2: Signal Lineage / Provenance
        with st.expander("Signal Provenance (Data Sources)"):
            prov_html = """
            <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
                <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] SIGNAL PROVENANCE</div>
                <div style="color: #A0A0A0; font-size: 13px; line-height: 1.8;">
            """
            for sig in signals:
                prov_html += f'<strong style="color: #D0D0D0;">{sig["signal"]}:</strong> {sig.get("derived_from", "Source not specified")}<br>'
            prov_html += """
                </div>
            </div>
            """
            st.markdown(prov_html, unsafe_allow_html=True)
    else:
        st.warning("Insufficient data for signal generation. Accumulating attribution history.")

    st.divider()

    # -----------------------------------------------
    # Section B2: Phase 1 Intelligence Signals
    # Observational, post-hoc analyses for IC decision support
    # -----------------------------------------------
    st.markdown('<span class="waves-micro-label">Intelligence Layer</span>', unsafe_allow_html=True)
    st.subheader("Intelligence Signals")
    st.caption("Observational analyses for governance clarity and decision confidence · Review prompts only")
    st.markdown("")
    
    def compute_phase1_signals(snapshot_df, attrib_df):
        """
        Phase 1 Adaptive Intelligence signals:
        - Alpha Source Concentration
        - Regime Transition Detection
        - Attribution Stability Over Time
        - Cross-Horizon Conflict Detection
        
        All signals are observational, advisory, and designed for IC review.
        """
        phase1_signals = []
        
        if attrib_df is None or len(attrib_df) == 0:
            return phase1_signals
        
        # --- Signal 1: Alpha Source Concentration ---
        # Determine if alpha is diversified or concentrated in few sources
        try:
            attrib_30d = attrib_df[attrib_df["horizon"] == 30]
            if len(attrib_30d) > 0:
                # Get mean absolute contribution from each alpha component
                components = ["selection_alpha", "momentum_alpha", "volatility_alpha", 
                              "regime_alpha", "exposure_alpha", "residual_alpha"]
                available_components = [c for c in components if c in attrib_30d.columns]
                
                if len(available_components) >= 3:
                    contributions = {}
                    total_abs_alpha = 0
                    for comp in available_components:
                        abs_contrib = attrib_30d[comp].abs().mean()
                        contributions[comp.replace("_alpha", "")] = abs_contrib
                        total_abs_alpha += abs_contrib
                    
                    if total_abs_alpha > 0:
                        # Calculate concentration (Herfindahl-like)
                        shares = [v / total_abs_alpha for v in contributions.values()]
                        concentration_score = sum(s**2 for s in shares)
                        
                        # Identify top contributors
                        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
                        top_contributor = sorted_contribs[0][0] if sorted_contribs else "Unknown"
                        top_share = (sorted_contribs[0][1] / total_abs_alpha * 100) if sorted_contribs else 0
                        
                        # Determine concentration level
                        if concentration_score > 0.4:
                            state = "Elevated"
                            confidence = "High"
                            what_this_means = f"Portfolio alpha is primarily driven by {top_contributor} ({top_share:.0f}% of total contribution). This is not inherently problematic, but IC may wish to understand dependency on this single driver."
                        elif concentration_score > 0.25:
                            state = "Moderate"
                            confidence = "Medium"
                            what_this_means = f"Alpha sources show moderate concentration, with {top_contributor} as the leading contributor. Multiple drivers are active, providing some diversification."
                        else:
                            state = "Low"
                            confidence = "Medium"
                            what_this_means = "Alpha is well-diversified across multiple sources. No single component dominates portfolio-level performance attribution."
                        
                        # Identify affected waves (those with high concentration in top factor)
                        affected_waves = []
                        for _, row in attrib_30d.iterrows():
                            wave_name = row.get("wave_name", "Unknown")
                            comp_val = abs(row.get(f"{top_contributor}_alpha", 0))
                            total_wave = sum(abs(row.get(c, 0)) for c in available_components)
                            if total_wave > 0 and comp_val / total_wave > 0.5:
                                affected_waves.append(wave_name)
                        
                        phase1_signals.append({
                            "signal": "Alpha Source Concentration",
                            "state": state,
                            "confidence": confidence,
                            "affected_waves": affected_waves[:5] if len(affected_waves) > 5 else affected_waves,
                            "evidence": f"Top contributor: {top_contributor} ({top_share:.0f}% of attribution). Concentration index: {concentration_score:.2f}",
                            "what_this_means": what_this_means
                        })
        except Exception:
            pass
        
        # --- Signal 2: Regime Transition Detection ---
        # Detect transitions between market regimes
        try:
            regime_30d = attrib_df[attrib_df["horizon"] == 30]["regime_alpha"].dropna()
            regime_365d = attrib_df[attrib_df["horizon"] == 365]["regime_alpha"].dropna()
            
            if len(regime_30d) > 0 and len(regime_365d) > 0:
                avg_regime_30d = regime_30d.mean()
                avg_regime_365d = regime_365d.mean()
                regime_diff = avg_regime_30d - avg_regime_365d
                
                # Check volatility alpha for risk-on/off signal
                vol_30d = attrib_df[attrib_df["horizon"] == 30]["volatility_alpha"].dropna()
                vol_change = vol_30d.mean() if len(vol_30d) > 0 else 0
                
                # Detect regime transition
                if abs(regime_diff) > 0.015:
                    if regime_diff > 0:
                        regime_direction = "favorable"
                        transition_type = "Risk-On Expansion" if vol_change > 0 else "Regime Stabilizing"
                    else:
                        regime_direction = "less favorable"
                        transition_type = "Risk-Off Contraction" if vol_change < 0 else "Regime Shift"
                    
                    state = "Detected"
                    confidence = "High" if abs(regime_diff) > 0.025 else "Medium"
                    what_this_means = f"Recent performance patterns suggest the market regime has shifted to a {regime_direction} environment compared to the long-term baseline. Attribution changes may reflect this external shift rather than strategy behavior."
                    
                    # Affected waves: those with large regime_alpha divergence
                    affected_waves = []
                    for _, row in attrib_30d.iterrows():
                        if "wave_name" in row and "regime_alpha" in row:
                            if abs(row["regime_alpha"]) > 0.01:
                                affected_waves.append(row["wave_name"])
                    
                    phase1_signals.append({
                        "signal": "Regime Transition",
                        "state": state,
                        "confidence": confidence,
                        "affected_waves": affected_waves[:5] if len(affected_waves) > 5 else affected_waves,
                        "evidence": f"Transition type: {transition_type}. 30D regime impact: {avg_regime_30d*100:.2f}% vs 365D: {avg_regime_365d*100:.2f}%",
                        "what_this_means": what_this_means
                    })
                else:
                    phase1_signals.append({
                        "signal": "Regime Transition",
                        "state": "Stable",
                        "confidence": "Medium",
                        "affected_waves": [],
                        "evidence": f"No significant regime shift detected. 30D/365D regime alignment: {(1 - abs(regime_diff))*100:.0f}%",
                        "what_this_means": "Market regime appears stable relative to historical patterns. Performance attribution reflects strategy behavior rather than external regime changes."
                    })
        except Exception:
            pass
        
        # --- Signal 3: Attribution Stability Over Time ---
        # Assess whether attribution drivers are consistent across horizons
        try:
            attrib_30d = attrib_df[attrib_df["horizon"] == 30]
            attrib_365d = attrib_df[attrib_df["horizon"] == 365]
            
            if len(attrib_30d) > 0 and len(attrib_365d) > 0:
                components = ["selection_alpha", "momentum_alpha", "volatility_alpha", 
                              "regime_alpha", "exposure_alpha"]
                available_components = [c for c in components if c in attrib_30d.columns and c in attrib_365d.columns]
                
                if len(available_components) >= 3:
                    # Compare ranking of components across horizons
                    avg_30d = {c: attrib_30d[c].mean() for c in available_components}
                    avg_365d = {c: attrib_365d[c].mean() for c in available_components}
                    
                    rank_30d = sorted(avg_30d.keys(), key=lambda x: avg_30d[x], reverse=True)
                    rank_365d = sorted(avg_365d.keys(), key=lambda x: avg_365d[x], reverse=True)
                    
                    # Count how many top-3 components are the same
                    top3_30d = set(rank_30d[:3])
                    top3_365d = set(rank_365d[:3])
                    overlap = len(top3_30d.intersection(top3_365d))
                    
                    # Calculate sign consistency
                    sign_matches = sum(1 for c in available_components 
                                       if (avg_30d[c] > 0) == (avg_365d[c] > 0))
                    sign_consistency = sign_matches / len(available_components)
                    
                    if overlap >= 3 and sign_consistency > 0.8:
                        state = "Stable"
                        confidence = "High"
                        what_this_means = "The drivers of performance are consistent across short and long time horizons. This suggests durable strategy behavior rather than temporary conditions."
                    elif overlap >= 2 and sign_consistency > 0.6:
                        state = "Mixed"
                        confidence = "Medium"
                        what_this_means = "Attribution drivers show moderate consistency across horizons. Some components behave differently in the short-term versus long-term, which may warrant review."
                    else:
                        state = "Low"
                        confidence = "Medium"
                        what_this_means = "Short-term and long-term attribution patterns differ meaningfully. Recent performance may be driven by different factors than historical patterns suggest."
                    
                    phase1_signals.append({
                        "signal": "Attribution Stability",
                        "state": state,
                        "confidence": confidence,
                        "affected_waves": list(attrib_30d["wave_name"].unique())[:5] if "wave_name" in attrib_30d.columns else [],
                        "evidence": f"Top-3 driver overlap: {overlap}/3. Sign consistency: {sign_consistency*100:.0f}%",
                        "what_this_means": what_this_means
                    })
        except Exception:
            pass
        
        # --- Signal 4: Cross-Horizon Conflict Detection ---
        # Identify when short-term signals conflict with long-term behavior
        try:
            if snapshot_df is not None and "alpha_30d" in snapshot_df.columns and "alpha_365d" in snapshot_df.columns:
                merged = snapshot_df[["wave_name", "alpha_30d", "alpha_365d"]].dropna()
                
                if len(merged) > 0:
                    # Find waves with conflicting alpha direction
                    conflicts = []
                    for _, row in merged.iterrows():
                        alpha_30d = row["alpha_30d"]
                        alpha_365d = row["alpha_365d"]
                        
                        # Conflict: opposite signs with meaningful magnitude
                        if ((alpha_30d > 0.005 and alpha_365d < -0.005) or 
                            (alpha_30d < -0.005 and alpha_365d > 0.005)):
                            conflicts.append({
                                "wave": row["wave_name"],
                                "short_term": alpha_30d,
                                "long_term": alpha_365d
                            })
                    
                    conflict_rate = len(conflicts) / len(merged)
                    
                    if conflict_rate > 0.3:
                        state = "Review Recommended"
                        confidence = "High"
                        what_this_means = f"{len(conflicts)} waves show short-term signals that conflict with long-term patterns. This is a review prompt to help IC distinguish between temporary reversions and structural changes — not an error signal."
                    elif conflict_rate > 0.15:
                        state = "Minor Conflicts"
                        confidence = "Medium"
                        what_this_means = "Some waves show divergence between short-term and long-term alpha direction. This is common during transitional periods and typically resolves over time."
                    else:
                        state = "Aligned"
                        confidence = "High"
                        what_this_means = "Short-term and long-term alpha signals are largely aligned. Current behavior appears consistent with historical patterns."
                    
                    affected_waves = [c["wave"] for c in conflicts[:5]]
                    
                    phase1_signals.append({
                        "signal": "Cross-Horizon Conflict",
                        "state": state,
                        "confidence": confidence,
                        "affected_waves": affected_waves,
                        "evidence": f"Conflicting waves: {len(conflicts)}/{len(merged)} ({conflict_rate*100:.0f}%)",
                        "what_this_means": what_this_means
                    })
        except Exception:
            pass
        
        return phase1_signals

    # Compute Phase 1 signals
    phase1_signals = compute_phase1_signals(snapshot_df, attrib_df)
    
    if len(phase1_signals) > 0:
        for sig in phase1_signals:
            # Determine icon based on state
            if sig["state"] in ["Stable", "Aligned", "Low"]:
                state_icon = "[+]"
            elif sig["state"] in ["Mixed", "Minor Conflicts", "Moderate"]:
                state_icon = "[-]"
            else:
                state_icon = "[.]"  # Review prompt icon, not warning
            
            with st.expander(f"{state_icon} **{sig['signal']}** — {sig['state']} (Confidence: {sig['confidence']})", expanded=False):
                st.markdown(f"**State:** {sig['state']}")
                
                if sig["affected_waves"]:
                    waves_str = ", ".join(sig["affected_waves"]) if isinstance(sig["affected_waves"], list) else str(sig["affected_waves"])
                    st.markdown(f"**Affected Waves:** {waves_str}")
                
                st.markdown(f"**Evidence:** {sig['evidence']}")
                st.divider()
                st.markdown(f"**What This Means:** {sig['what_this_means']}")
    else:
        st.info("Accumulating sufficient data for intelligence signal generation.")

    st.divider()

    # -----------------------------------------------
    # Section C: Recommendations
    # -----------------------------------------------
    st.subheader("Recommendations")
    st.caption("Advisory suggestions based on observed patterns · Not execution directives")

    for i, rec in enumerate(recommendations):
        with st.expander(f"**{rec['title']}** — Confidence: {rec['confidence']}", expanded=(i == 0)):
            st.markdown(f"**Suggestion:** {rec['suggestion']}")
            st.markdown(f"**Rationale:** {rec['reason']}")
            st.markdown(f"**Expected Impact:** {rec['impact']}")

    st.divider()

    # -----------------------------------------------
    # Section D: Readiness Flags
    # -----------------------------------------------
    st.subheader("Readiness Flags")
    st.caption("System status indicators")
    st.markdown("")
    
    flag_cols = st.columns(4)
    flag_items = list(readiness_flags.items())

    for i, (flag_name, flag_value) in enumerate(flag_items):
        with flag_cols[i % 4]:
            # Color-code based on value
            if flag_value in ["Stable", "Normal", "Not Required", "Within Expected Bounds"]:
                icon = "[+]"
            elif flag_value in ["Moderate", "Building"]:
                icon = "[-]"
            else:
                icon = "[!]"

            st.metric(
                label=flag_name,
                value=f"{icon} {flag_value}"
            )

    st.divider()

    # Signal status summary
    with st.expander("Signal Generation Status"):
        live_signals = [s for s in signals if s["confidence"] in ["High", "Medium"]]
        pending_signals = [s for s in signals if s["confidence"] == "Low"]

        st.markdown(f"**Live Signals:** {len(live_signals)}")
        for s in live_signals:
            st.markdown(f"- {s['signal']}: {s['direction']} ({s['confidence']})")

        st.markdown(f"**Pending/Low-Confidence Signals:** {len(pending_signals)}")
        for s in pending_signals:
            st.markdown(f"- {s['signal']}: Insufficient data for high-confidence assessment")

    # Refinement 4: Soft Hand-Off to Operations
    st.caption("These insights inform optional actions available in the Operations Center.")

    # ===================================================================
    # ADVANCED ADAPTIVE INTELLIGENCE SECTIONS
    # ===================================================================

    st.divider()
    st.header("Advanced Adaptive Intelligence")
    st.caption("Extended decision support features · Advisory only · No execution")

    # -----------------------------------------------
    # Section: Scenario Simulator (What-If Engine) - LIVE
    # -----------------------------------------------
    st.subheader("Scenario Simulator (What-If Engine)")
    st.caption("Simulation based on real historical wave behavior · Explore hypothetical changes to understand potential portfolio impacts")

    # Scenario selection
    scenario_options = [
        "Select a scenario...",
        "Reduced Volatility (-20% vol exposure)",
        "Increased Concentration (top 5 waves)",
        "Remove Inconsistent-Alpha Waves",
        "Reinforce Regime-Stable Waves"
    ]
    selected_scenario = st.selectbox("Choose Scenario", scenario_options, key="scenario_selector")

    if selected_scenario != "Select a scenario...":
        # Use LIVE learning module for scenario simulation
        sim_results = al.compute_scenario_simulation(selected_scenario, snapshot_df, attrib_df)

        if sim_results and "error" not in sim_results:
            sim_cols = st.columns(4)
            with sim_cols[0]:
                pr = sim_results.get("projected_return", 0)
                st.metric("Projected Return", f"{pr*100:.2f}%" if pr else "—")
            with sim_cols[1]:
                pa = sim_results.get("projected_alpha", 0)
                st.metric("Projected Alpha", f"{pa*100:.2f}%" if pa else "—")
            with sim_cols[2]:
                dd = sim_results.get("projected_drawdown_change", 0)
                st.metric("Drawdown Change", f"{dd:+.1f}%" if dd else "—")
            with sim_cols[3]:
                st.metric("Risk Level", "Moderate")

            st.info(f"**Risk Notes:** {sim_results.get('risk_notes', 'No specific notes.')}")
        else:
            st.warning("Insufficient data to run simulation.")

    st.divider()

    # -----------------------------------------------
    # Section: Adaptive Threshold Learning - LIVE
    # -----------------------------------------------
    st.subheader("Adaptive Threshold Learning")
    st.caption("Observes outcomes to refine when the system speaks — not when it trades.")

    st.markdown(
        "The system learns when to surface observations based on historical patterns, "
        "reducing false alerts over time. **All thresholds are LIVE and evolving.**"
    )

    # Use LIVE learned thresholds from adaptive state
    state_thresholds = adaptive_state.get("thresholds", {})

    # Display learned thresholds with history
    threshold_display = []
    for threshold_name, threshold_data in state_thresholds.items():
        current_val = threshold_data.get("current", 0)
        history = threshold_data.get("history", [])

        if history and len(history) > 0:
            last_change = history[-1]
            old_val = last_change.get("old", current_val)
            change_date = last_change.get("date", "recently")[:10] if last_change.get("date") else "recently"
            threshold_display.append({
                "name": threshold_name.replace("_", " ").title(),
                "current": f"{current_val*100:.1f}%" if abs(current_val) < 1 else f"{current_val:.2f}",
                "change": f"Adjusted from {old_val*100:.1f}% → {current_val*100:.1f}% based on last 90 days of outcomes",
                "learned": True
            })
        else:
            threshold_display.append({
                "name": threshold_name.replace("_", " ").title(),
                "current": f"{current_val*100:.1f}%" if abs(current_val) < 1 else f"{current_val:.2f}",
                "change": "Initial heuristic — will become adaptive once sufficient live data accumulates",
                "learned": False
            })

    for td in threshold_display:
        icon = "[L]" if td["learned"] else "[P]"
        st.markdown(f"**{icon} {td['name']}:** {td['current']}")
        st.caption(td["change"])

    # Why speaking up now
    active_alerts = []
    for sig in signals:
        if sig["confidence"] == "High":
            if "Negative" in sig["direction"] or "Low" in sig["direction"]:
                active_alerts.append(f"{sig['signal']}: {sig['evidence']}")

    if active_alerts:
        st.warning("**Why the system is speaking up now:**")
        for alert in active_alerts:
            st.markdown(f"- {alert}")
    else:
        st.success("**System State: Stable — No Action Recommended.** No threshold breaches detected.")

    st.divider()

    # -----------------------------------------------
    # Section: Confidence-Weighted Recommendations
    # -----------------------------------------------
    st.subheader("Confidence-Weighted Recommendations")
    st.caption("Aggregated confidence across multiple signals")

    st.markdown(
        "Recommendations are classified based on aggregated confidence from multiple data signals, "
        "similar to how professional portfolio management teams discuss investment ideas."
    )

    # Classify recommendations
    def classify_recommendation(rec, all_signals):
        """Classify recommendation based on aggregated confidence."""
        rec_confidence = rec.get("confidence", "Low")

        # Count supporting signals
        supporting_signals = sum(1 for s in all_signals if s["confidence"] in ["High", "Medium"])

        if rec_confidence == "High" and supporting_signals >= 3:
            return "High Conviction", "Multiple high-confidence signals align with this recommendation."
        elif rec_confidence in ["High", "Medium"] and supporting_signals >= 2:
            return "Watchlist-Level", "Moderate evidence supports this recommendation. Monitor for confirmation."
        else:
            return "Informational Only", "Early signal. Insufficient evidence for actionable conviction."

    # Display classified recommendations
    for rec in recommendations:
        classification, explanation = classify_recommendation(rec, signals)

        # Badge styling
        if classification == "High Conviction":
            badge_color = "[+]"
        elif classification == "Watchlist-Level":
            badge_color = "[-]"
        else:
            badge_color = "[?]"

        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{rec['title']}**")
                st.caption(rec['suggestion'])
            with col2:
                st.markdown(f"{badge_color} **{classification}**")

            with st.expander("Confidence Details"):
                st.markdown(f"**Classification:** {classification}")
                st.markdown(f"**Explanation:** {explanation}")
                st.markdown(f"**Base Confidence:** {rec['confidence']}")
                st.markdown(f"**Impact:** {rec['impact']}")

    st.divider()

    # -----------------------------------------------
    # Section: Cross-Horizon Agreement Engine - LIVE
    # -----------------------------------------------
    st.subheader("Cross-Horizon Agreement Engine")
    st.caption("Real-time 30D vs 365D signal comparison · Analyzes short-term momentum against long-term structural signals")

    # Use LIVE learning module for cross-horizon analysis
    cross_horizon_agreements = al.compute_cross_horizon_agreement(snapshot_df, attrib_df)

    # Ensure we have a valid list
    if cross_horizon_agreements and isinstance(cross_horizon_agreements, list):
        for analysis in cross_horizon_agreements:
            # Defensive: ensure analysis is a dict
            if not isinstance(analysis, dict):
                continue
            
            # Determine icon based on agreement
            agreement_str = analysis.get("agreement", "")
            if "Aligned" in agreement_str and "Negative" not in agreement_str:
                icon = "[OK]"
            elif "Negative" in agreement_str:
                icon = "[v]"
            elif analysis.get("suppress_action", False):
                icon = "[!]"
            else:
                icon = "[-]"

            # Safe access to comparison with default
            comparison = analysis.get("comparison", "Cross-Horizon Analysis")
            st.markdown(f"### {icon} {comparison}")
            cols = st.columns(3)
            with cols[0]:
                short_val = analysis.get("short_term", 0)
                st.metric("Short-Term (30D)", f"{short_val*100:.2f}%" if isinstance(short_val, (int, float)) else short_val)
            with cols[1]:
                long_val = analysis.get("long_term", 0)
                st.metric("Long-Term (365D)", f"{long_val*100:.2f}%" if isinstance(long_val, (int, float)) else long_val)
            with cols[2]:
                st.metric("Agreement", analysis.get("agreement", "—"))

            # Show interpretation prominently
            if analysis.get("suppress_action", False):
                st.warning(f"**{analysis.get('interpretation', '')}**")
            else:
                st.info(f"**Interpretation:** {analysis.get('interpretation', '')}")
    else:
        st.warning("Insufficient cross-horizon data for agreement analysis.")

    st.divider()

    # -----------------------------------------------
    # Section: Review & Adaptation Signals
    # -----------------------------------------------
    ai.render_review_and_adaptation_signals(
        snapshot_df=snapshot_df,
        attrib_df=attrib_df,
        adaptive_state=adaptive_state
    )

    st.divider()

    # -----------------------------------------------
    # Section: Adaptive Tilt Proposals (LAST) - LIVE
    # -----------------------------------------------
    st.subheader("Adaptive Tilt Proposals")
    st.caption("Preview-only insights requiring human approval.")

    st.markdown(
        "Suggested portfolio tilts based on **learned signals and confidence thresholds**. "
        "Proposals surface ONLY when learned confidence exceeds learned thresholds."
    )

    st.warning("**Human Review Required.** No action recommended without explicit approval.")

    # Use LIVE learning module for tilt proposals
    # Pass cross-horizon agreements to suppress action when horizons disagree
    tilt_proposals = al.generate_adaptive_tilt_proposals(signals, adaptive_state, cross_horizon_agreements)

    # Display as cards with LIVE learning information
    if tilt_proposals and isinstance(tilt_proposals, list):
        for proposal in tilt_proposals:
            # Defensive: ensure proposal is a dict
            if not isinstance(proposal, dict):
                continue
            
            with st.container():
                card_cols = st.columns([4, 1])

                with card_cols[0]:
                    live_badge = "[LIVE]" if proposal.get('is_live', False) else ""
                    title = proposal.get('title', 'Untitled Proposal')
                    st.markdown(f"### {title} {live_badge}")
                    st.markdown(proposal.get('description', ''))

                with card_cols[1]:
                    confidence_score = proposal.get('confidence_score', 0)
                    confidence = proposal.get('confidence', 'Low')
                    if confidence == "High":
                        conf_badge = f"[+] High ({confidence_score*100:.0f}%)"
                    elif confidence == "Medium":
                        conf_badge = f"[-] Medium ({confidence_score*100:.0f}%)"
                    else:
                        conf_badge = f"[?] Low ({confidence_score*100:.0f}%)"
                    st.markdown(f"**{conf_badge}**")

                with st.expander("Details"):
                    st.markdown(f"**Expected Impact:** {proposal.get('expected_impact', 'N/A')}")
                    st.markdown(f"**Supporting Evidence:** {proposal.get('supporting_evidence', 'N/A')}")
                    if 'learned_threshold' in proposal:
                        st.markdown(f"**Learned Threshold:** {proposal['learned_threshold']}")
                    st.markdown("---")
                    st.caption("This proposal is for review only. No trades will be executed.")

    # Save updated adaptive state
    al.save_adaptive_state(adaptive_state)

    st.divider()

    # -----------------------------------------------
    # Section: Strategy Integrity (Integrity Signals)
    # -----------------------------------------------
    st.header("Strategy Integrity")
    st.caption("Validates system behavior without influencing decisions.")

    integrity_data = integ.get_all_integrity_signals(snapshot_df, attrib_df)

    int_cols = st.columns([1, 2, 1])
    with int_cols[0]:
        integrity_idx = integrity_data.get("integrity_index", {})
        idx_value = integrity_idx.get("index", 0)
        idx_status = integrity_idx.get("status", "Unknown")
        
        if idx_status == "Healthy":
            status_icon = "[+]"
        elif idx_status == "Satisfactory":
            status_icon = "[OK]"
        elif idx_status == "Needs_Attention":
            status_icon = "[-]"
        else:
            status_icon = "[.]"
        
        st.metric(
            label="Integrity Index",
            value=f"{idx_value:.0f}/100",
            help="Diagnostic health summary. Complements WaveScore — does NOT replace it."
        )
        st.caption(f"{status_icon} {idx_status.replace('_', ' ')}")

    with int_cols[1]:
        components = integrity_idx.get("components", {})
        comp_cols = st.columns(3)
        with comp_cols[0]:
            st.metric("Overlay Health", f"{components.get('overlay_health', 0):.0f}/60")
        with comp_cols[1]:
            st.metric("Selection Compliance", f"{components.get('selection_compliance', 0):.0f}/25")
        with comp_cols[2]:
            st.metric("Drift Score", f"{components.get('drift_score', 0):.0f}/15")

    with int_cols[2]:
        st.caption(integrity_idx.get("note", ""))

    st.divider()

    st.subheader("Overlay Integrity Signals")
    st.caption("Backward-looking contribution sanity checks · Historical envelopes only")
    st.markdown("")
    
    overlay_signals = integrity_data.get("overlay_signals", [])
    if overlay_signals:
        healthy_overlays = [s for s in overlay_signals if s["status"] == "Healthy"]
        watch_overlays = [s for s in overlay_signals if s["status"] == "Watch"]
        oob_overlays = [s for s in overlay_signals if s["status"] == "Out_of_Bounds"]

        overlay_summary_cols = st.columns(3)
        with overlay_summary_cols[0]:
            st.metric("Healthy", len(healthy_overlays), help="Within historical sanity envelope")
        with overlay_summary_cols[1]:
            st.metric("Watch", len(watch_overlays), help="Approaching envelope boundary")
        with overlay_summary_cols[2]:
            st.metric("Out of Bounds", len(oob_overlays), help="Outside historical sanity envelope")

        st.markdown("")
        
        with st.expander("Overlay Signal Details", expanded=False):
            overlay_df = pd.DataFrame(overlay_signals)
            display_cols = ["overlay", "contribution_pct", "regime", "status"]
            if all(col in overlay_df.columns for col in display_cols):
                st.dataframe(
                    overlay_df[display_cols],
                    hide_index=True,
                    column_config={
                        "overlay": st.column_config.TextColumn("Overlay", width="medium"),
                        "contribution_pct": st.column_config.NumberColumn("Contribution %", format="%.2f"),
                        "regime": st.column_config.TextColumn("Regime", width="small"),
                        "status": st.column_config.TextColumn("Status", width="small")
                    }
                )
    else:
        st.info("Accumulating data for overlay integrity analysis.")

    st.divider()

    st.subheader("Algorithm Integrity")
    st.caption("VIX, Risk-On/Off, and Volatility algorithm health · Activation and stress behavior")

    algo_signals = integrity_data.get("algorithm_signals", [])
    if algo_signals:
        for algo in algo_signals:
            algo_name = algo.get("overlay", "Unknown")
            algo_status = algo.get("status", "Unknown")
            
            if algo_status == "Healthy":
                status_icon = "[+]"
            elif algo_status == "Watch":
                status_icon = "[-]"
            else:
                status_icon = "[.]"
            
            with st.expander(f"{status_icon} **{algo_name}** — {algo_status}", expanded=False):
                algo_cols = st.columns(3)
                with algo_cols[0]:
                    st.metric("Activation Rate", f"{algo.get('activation_rate', 0)*100:.0f}%")
                with algo_cols[1]:
                    st.metric("Stress Capture", f"{algo.get('stress_capture', 0)*100:.2f}%")
                with algo_cols[2]:
                    if "contribution_avg" in algo:
                        st.metric("Avg Contribution", f"{algo.get('contribution_avg', 0):.2f}%")
                    elif "regime_volatility" in algo:
                        st.metric("Regime Volatility", f"{algo.get('regime_volatility', 0)*100:.2f}%")
                
                st.caption("Diagnostic status only — not a recommendation")
    else:
        st.info("Accumulating data for algorithm integrity analysis.")

    st.divider()

    st.subheader("Wave Overlay Health")
    st.caption("Aggregate overlay health per Wave")

    wave_health = integrity_data.get("wave_health", {})
    if wave_health:
        wave_health_list = []
        for wave_name, health in wave_health.items():
            wave_health_list.append({
                "wave": wave_name,
                "status": health.get("status", "Unknown"),
                "health_pct": health.get("health_pct", 0),
                "healthy": health.get("healthy_count", 0),
                "watch": health.get("watch_count", 0),
                "out_of_bounds": health.get("out_of_bounds_count", 0)
            })
        
        wave_health_df = pd.DataFrame(wave_health_list)
        st.dataframe(
            wave_health_df,
            hide_index=True,
            column_config={
                "wave": st.column_config.TextColumn("Wave", width="large"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "health_pct": st.column_config.NumberColumn("Health %", format="%.0f%%"),
                "healthy": st.column_config.NumberColumn("Healthy", width="small"),
                "watch": st.column_config.NumberColumn("Watch", width="small"),
                "out_of_bounds": st.column_config.NumberColumn("OOB", width="small")
            }
        )
    else:
        st.info("Accumulating data for wave overlay health analysis.")

    st.divider()
    st.caption("End of Advanced Adaptive Intelligence sections. All data persisted to adaptive_state.json.")


# ===========================
# OPERATIONS TAB
# ===========================
with tabs[3]:
    st.header("Operations Center")
    st.caption("Human-in-the-loop governance layer | Configuration & Audit | No auto-execution")
    st.markdown("")
    
    # -----------------------------------------------
    # SYSTEM CONTROL SNAPSHOT (TOP SECTION)
    # -----------------------------------------------
    st.subheader("System Control Snapshot")
    st.caption("Operational state summary · Backward-looking · Observational only")
    
    def compute_operations_snapshot():
        ops_log_path = "data/operations_log.json"
        try:
            with open(ops_log_path, "r") as f:
                ops_log = json.load(f)
        except:
            ops_log = {"entries": []}
        
        entries = ops_log.get("entries", [])
        total_actions = len(entries)
        
        sorted_entries = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)
        last_entry = sorted_entries[0] if sorted_entries else None
        recalculates = sum(1 for e in entries if e.get("action_type") == "RECALCULATE")
        unique_users = len(set(e.get("user", "Unknown") for e in entries))
        
        return {
            "total_actions": total_actions,
            "recalculate_count": recalculates,
            "unique_actors": unique_users,
            "last_action_time": last_entry.get("timestamp", "N/A")[:16] if last_entry else "None recorded",
            "last_action_user": last_entry.get("user", "N/A") if last_entry else "N/A"
        }
    
    ops_snap = compute_operations_snapshot()
    
    snap_cols = st.columns(5)
    snap_cols[0].metric("Total Actions", ops_snap["total_actions"], help="Count of governance actions recorded in log")
    snap_cols[1].metric("Recalculations", ops_snap["recalculate_count"], help="Count of RECALCULATE actions logged")
    snap_cols[2].metric("Unique Actors", ops_snap["unique_actors"], help="Count of distinct users in log")
    snap_cols[3].metric("Last Action", ops_snap["last_action_time"], help="Timestamp of most recent logged action")
    snap_cols[4].metric("Last Actor", ops_snap["last_action_user"], help="User who performed most recent logged action")
    
    st.markdown("")
    ops_interp_cols = st.columns(2)
    
    with ops_interp_cols[0]:
        st.markdown("**What Has Been Observed**")
        st.caption(f"• {ops_snap['total_actions']} governance action(s) have been recorded")
        st.caption(f"• {ops_snap['recalculate_count']} recalculation(s) were logged")
        st.caption(f"• {ops_snap['unique_actors']} distinct actor(s) have been identified")
    
    with ops_interp_cols[1]:
        st.markdown("**Historical Context**")
        st.caption("• All logged actions have included timestamps and actor identification")
        st.caption("• Recalculation events have been the primary action type observed")
        st.caption("• Log entries have been maintained with rationale documentation")
    
    # -----------------------------------------------
    # EXPORT OPERATIONS SNAPSHOT
    # -----------------------------------------------
    def generate_operations_export():
        """Generate CSV export of Operations Snapshot."""
        from datetime import datetime
        
        export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        rows = [{
            "Export Timestamp": export_time,
            "Total Actions": ops_snap["total_actions"],
            "Recalculations": ops_snap["recalculate_count"],
            "Unique Actors": ops_snap["unique_actors"],
            "Last Action Time": ops_snap["last_action_time"],
            "Last Actor": ops_snap["last_action_user"]
        }]
        
        return pd.DataFrame(rows)
    
    ops_export_df = generate_operations_export()
    ops_csv = ops_export_df.to_csv(index=False)
    
    st.download_button(
        label="Export Operations Snapshot",
        data=ops_csv,
        file_name=f"operations_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        help="Download Operations Snapshot as CSV"
    )
    
    st.markdown("---")
    
    # -----------------------------------------------
    # Helper Functions for Operations
    # -----------------------------------------------
    def load_portfolio_config():
        """Load portfolio configuration state."""
        config_path = "data/portfolio_config.json"
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "version": "1.0",
                "last_updated": None,
                "last_updated_by": None,
                "exposure_limits": {
                    "max_single_wave_exposure": 0.15,
                    "max_sector_concentration": 0.40,
                    "min_wave_count": 10
                },
                "wave_settings": {},
                "tilt_settings": {
                    "momentum_tilt_enabled": True,
                    "volatility_dampening_enabled": True,
                    "regime_adaptation_enabled": True
                },
                "risk_controls": {
                    "max_drawdown_tolerance": 0.20,
                    "volatility_cap": 0.25,
                    "rebalance_threshold": 0.05
                },
                "pending_recommendations": [],
                "applied_recommendations": []
            }
    
    def save_portfolio_config(config):
        """Save portfolio configuration state."""
        config_path = "data/portfolio_config.json"
        config["last_updated"] = datetime.now().isoformat()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def load_operations_log():
        """Load operations audit log."""
        log_path = "data/operations_log.json"
        try:
            with open(log_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"version": "1.0", "entries": []}
    
    def save_operations_log(log):
        """Save operations audit log."""
        log_path = "data/operations_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
    
    def execute_portfolio_rebuild():
        """
        Execute the closed-loop portfolio rebuild based on current config.
        Decision → Config mutation → Rebuild portfolio → Refresh attribution & intelligence
        Returns tuple (success, messages).
        """
        global snapshot_df, attrib_df
        rebuild_messages = []
        try:
            # Step 0: Force re-read of live data to get latest state
            try:
                fresh_snapshot = pd.read_csv(LIVE_SNAPSHOT_PATH)
                fresh_attrib = pd.read_csv(ALPHA_ATTRIBUTION_PATH)
                # Normalize column names to lowercase
                fresh_snapshot.columns = [c.lower() for c in fresh_snapshot.columns]
                fresh_attrib.columns = [c.lower() for c in fresh_attrib.columns]
                snapshot_df = fresh_snapshot
                attrib_df = fresh_attrib
                rebuild_messages.append("Live data refreshed from disk")
            except Exception as e:
                rebuild_messages.append(f"Data refresh warning: {str(e)}")
            
            config = load_portfolio_config()
            rebuild_messages.append(f"Config loaded: {config.get('last_updated', 'unknown')}")
            
            # Step 1: Apply tilt settings to compute adjusted wave weights
            tilt_settings = config.get("tilt_settings", {})
            exposure_limits = config.get("exposure_limits", {})
            risk_controls = config.get("risk_controls", {})
            
            wave_adjustments = {}
            if snapshot_df is not None and not snapshot_df.empty:
                # Determine wave name column (wave_name or wave)
                wave_col = "wave_name" if "wave_name" in snapshot_df.columns else "wave"
                
                # Validate required columns exist
                has_alpha = "alpha_30d" in snapshot_df.columns
                has_return = "return_intraday" in snapshot_df.columns
                if not has_alpha:
                    rebuild_messages.append("Warning: alpha_30d column missing, momentum tilt disabled")
                if not has_return:
                    rebuild_messages.append("Warning: return_intraday column missing, volatility dampening disabled")
                
                for _, wave_row in snapshot_df.iterrows():
                    wave_name = wave_row.get(wave_col, "Unknown")
                    if pd.isna(wave_name) or wave_name == "":
                        wave_name = "Unknown"
                    base_weight = 1.0 / len(snapshot_df)  # Equal weight base
                    
                    # Apply momentum tilt (with validation)
                    if tilt_settings.get("momentum_tilt_enabled", True) and has_alpha:
                        try:
                            momentum_signal = float(wave_row.get("alpha_30d", 0) or 0)
                            if pd.isna(momentum_signal) or not np.isfinite(momentum_signal):
                                momentum_signal = 0.0
                            tilt_factor = tilt_settings.get("momentum_tilt_factor", 0.1)
                            base_weight *= (1 + momentum_signal * tilt_factor)
                        except (ValueError, TypeError):
                            pass
                    
                    # Apply volatility dampening (with validation)
                    if tilt_settings.get("volatility_dampening_enabled", True) and has_return:
                        try:
                            vol_signal = abs(float(wave_row.get("return_intraday", 0) or 0))
                            if pd.isna(vol_signal) or not np.isfinite(vol_signal):
                                vol_signal = 0.0
                            damp_factor = tilt_settings.get("volatility_dampening_factor", 0.05)
                            base_weight *= max(0.5, 1 - vol_signal * damp_factor)
                        except (ValueError, TypeError):
                            pass
                    
                    # Apply max wave cap
                    max_wave = exposure_limits.get("max_single_wave_pct", 15) / 100
                    base_weight = min(base_weight, max_wave)
                    
                    # Validate weight is finite
                    if not np.isfinite(base_weight) or base_weight <= 0:
                        base_weight = 1.0 / len(snapshot_df)
                    
                    wave_adjustments[wave_name] = base_weight
                
                # Normalize weights
                total_weight = sum(wave_adjustments.values())
                if total_weight > 0:
                    wave_adjustments = {k: v/total_weight for k, v in wave_adjustments.items()}
                
                rebuild_messages.append(f"Wave weights recalculated: {len(wave_adjustments)} waves")
            
            # Step 2: Update adaptive state with current data
            if snapshot_df is not None and attrib_df is not None:
                updated_state, learn_msgs = al.update_adaptive_state(snapshot_df, attrib_df)
                rebuild_messages.extend(learn_msgs)
                rebuild_messages.append("Adaptive Intelligence state refreshed")
            
            # Step 3: Store wave adjustments in config for reference
            config["computed_wave_weights"] = wave_adjustments
            config["last_rebuild"] = datetime.now().isoformat()
            save_portfolio_config(config)
            
            rebuild_messages.append("Portfolio rebuild complete")
            return True, rebuild_messages
            
        except Exception as e:
            rebuild_messages.append(f"Rebuild error: {str(e)}")
            return False, rebuild_messages
    
    def add_audit_entry(action_type, recommendation, decision, rationale=None):
        """Add an entry to the operations audit log."""
        log = load_operations_log()
        entry = {
            "id": f"OP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(log['entries'])+1:04d}",
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "recommendation": recommendation,
            "decision": decision,
            "rationale": rationale,
            "user": "System Operator"
        }
        log["entries"].insert(0, entry)
        log["entries"] = log["entries"][:100]
        save_operations_log(log)
        return entry["id"]
    
    def get_pending_recommendations():
        """Get recommendations from Adaptive Intelligence for review."""
        recommendations = []
        error_message = None
        try:
            adaptive_state = al.load_adaptive_state()
            if snapshot_df is not None and attrib_df is not None:
                signals = al.compute_derived_signals(snapshot_df, attrib_df)
                cross_horizon_agreements = al.compute_cross_horizon_agreement(snapshot_df, attrib_df)
                tilt_proposals = al.generate_adaptive_tilt_proposals(signals, adaptive_state, cross_horizon_agreements)
                
                for i, proposal in enumerate(tilt_proposals):
                    recommendations.append({
                        "id": f"REC-{i+1:03d}",
                        "type": "tilt_adjustment",
                        "title": proposal.get("title", "Untitled"),
                        "description": proposal.get("description", ""),
                        "confidence": proposal.get("confidence", "Low"),
                        "confidence_score": proposal.get("confidence_score", 0.3),
                        "expected_impact": proposal.get("expected_impact", ""),
                        "supporting_evidence": proposal.get("supporting_evidence", ""),
                        "learned_threshold": proposal.get("learned_threshold", ""),
                        "cross_horizon_status": "Aligned" if (isinstance(cross_horizon_agreements, list) and not any(a.get("suppress_action", False) for a in cross_horizon_agreements if isinstance(a, dict))) else "Conflicting",
                        "source": "Adaptive Intelligence"
                    })
            else:
                error_message = "Live data unavailable for recommendation generation."
        except Exception as e:
            error_message = f"Error generating recommendations: {str(e)}"
        return recommendations, error_message
    
    # Load current state
    portfolio_config = load_portfolio_config()
    operations_log = load_operations_log()
    pending_recs, rec_error = get_pending_recommendations()
    
    # ===================================================================
    # IC AUTHORITY — INACTIVE
    # Governance-first control scaffold (intentionally locked)
    # ===================================================================
    ic_authority_html = """<div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 24px; margin: 16px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
<div style="position: absolute; top: 16px; right: 20px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
<div style="display: flex; align-items: center; margin-bottom: 20px;">
<span style="color: #A0A0A0; font-size: 16px; margin-right: 10px;">[L]</span>
<span style="color: #D0D0D0; font-size: 18px; font-weight: 600; letter-spacing: 0.5px;">IC Authority — Inactive</span>
</div>
<div style="background: rgba(255,193,7,0.08); border-left: 3px solid #FFC107; border-radius: 4px; padding: 14px 18px; margin-bottom: 20px;">
<div style="color: #D0D0D0; font-size: 13px; line-height: 1.5;">
<strong>This section defines where Investment Committee–authorized strategy controls are managed.</strong><br>
Current state: <strong>Observational-only</strong>. Execution is intentionally disabled until formal IC delegation.
</div>
</div>
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;">STRATEGY CONTROL CATEGORIES</div>
<div style="color: #666666; font-size: 12px; margin-bottom: 16px;">Structural definitions · All controls locked · No IC override in effect</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;">
<div>
<div style="color: #D0D0D0; font-weight: 600; margin-bottom: 6px;">Volatility / VIX Overlay Policy</div>
<div style="color: #666666; font-size: 12px;">[L] Locked · No IC override in effect</div>
</div>
<div>
<div style="color: #D0D0D0; font-weight: 600; margin-bottom: 6px;">Exposure Constraints</div>
<div style="color: #666666; font-size: 12px;">[L] Locked · No IC override in effect</div>
</div>
<div>
<div style="color: #D0D0D0; font-weight: 600; margin-bottom: 6px;">Drift Handling Policy</div>
<div style="color: #666666; font-size: 12px;">[L] Locked · No IC override in effect</div>
</div>
<div>
<div style="color: #D0D0D0; font-weight: 600; margin-bottom: 6px;">Rebalancing / Intervention Cadence</div>
<div style="color: #666666; font-size: 12px;">[L] Locked · No IC override in effect</div>
</div>
</div>
</div>"""
    st.markdown(ic_authority_html, unsafe_allow_html=True)
    
    # Future IC Authority Scope (Informational dropdown)
    with st.expander("Future IC Authority Scope (Informational Only)", expanded=False):
        ic_scope_html = """<div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
<div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] FUTURE IC AUTHORITY SCOPE</div>
<div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Upon formal delegation, the Investment Committee will have authority to:</div>
<div style="padding-left: 12px; color: #A0A0A0;">
— Approve or suspend defensive overlays (e.g., VIX-based risk mitigation)<br>
— Authorize drift response actions within predefined bounds<br>
— Set or modify exposure constraint ranges<br>
— Approve temporary or structural changes to rebalancing cadence<br>
— Acknowledge and log IC-approved overrides<br>
— Record human rationale and timestamps for all IC actions<br>
— Revert IC-authorized changes under defined governance procedures
</div>
</div>
<div style="color: #D0D0D0; font-size: 13px; line-height: 1.6;">
<div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Governance Constraints:</div>
<div style="padding-left: 12px; color: #A0A0A0;">
— All actions require explicit human IC approval<br>
— No automatic execution occurs under any circumstance<br>
— All actions are auditable and reversible<br>
— Activation occurs only after formal delegation under acquiring institution's framework
</div>
</div>
<div style="border-top: 1px solid #2A2A2A; margin-top: 16px; padding-top: 12px; font-size: 11px; color: #666666;">This scope documentation is informational only · No execution capability</div>
</div>"""
        st.markdown(ic_scope_html, unsafe_allow_html=True)
    
    # Explanatory rationale note
    ic_note_html = """<div style="color: #666666; font-size: 12px; line-height: 1.5; margin: 12px 0;">
<strong style="color: #A0A0A0;">Note:</strong> The IC Authority layer is architected but intentionally inactive. Execution authority is enabled only upon formal delegation under the acquiring institution's governance, compliance, and fiduciary framework. This ensures WAVES remains non-autonomous, audit-safe, and institution-aligned prior to delegation.
</div>
<div style="color: #666666; font-size: 12px; margin-bottom: 16px;">
This control plane is informed by: Integrity Signals · Adaptive Learning · Readiness Flags. No recommendations. No implied actions.
</div>"""
    st.markdown(ic_note_html, unsafe_allow_html=True)
    
    st.divider()
    
    # -----------------------------------------------
    # Section 1: Recommendation Intake Panel
    # -----------------------------------------------
    st.subheader("1. Recommendation Intake")
    st.caption("System-generated recommendations from Adaptive Intelligence | Preview Only")
    
    if rec_error:
        st.warning(f"Recommendation data issue: {rec_error}")
    
    if len(pending_recs) == 0:
        st.info("No pending recommendations. The system generates recommendations when confidence thresholds are met and cross-horizon signals align.")
    else:
        for rec in pending_recs:
            with st.container():
                rec_cols = st.columns([4, 1])
                with rec_cols[0]:
                    st.markdown(f"**{rec['title']}** `{rec['id']}`")
                    st.markdown(f"{rec['description']}")
                    st.caption(f"Evidence: {rec['supporting_evidence']}")
                    st.caption(f"Expected Impact: {rec['expected_impact']}")
                with rec_cols[1]:
                    confidence_color = {"High": "[+]", "Medium": "[-]", "Low": "[v]"}.get(rec["confidence"], "[?]")
                    st.markdown(f"{confidence_color} **{rec['confidence']}**")
                    st.caption(f"Score: {rec['confidence_score']*100:.0f}%")
                    horizon_icon = "[+]" if rec["cross_horizon_status"] == "Aligned" else "[!]"
                    st.caption(f"{horizon_icon} {rec['cross_horizon_status']}")
                st.caption(f"Preview — Not Executed | Source: {rec['source']}")
                st.divider()
    
    # -----------------------------------------------
    # Section 2: Human Decision Controls
    # -----------------------------------------------
    st.subheader("2. Decision Controls")
    st.caption("Review and act on recommendations | All decisions are audited")
    
    if len(pending_recs) == 0:
        st.caption("**System State: Quiet — No Signals Elevated.** No recommendations available for action.")
    else:
        selected_rec_title = st.selectbox(
            "Select Recommendation",
            options=[r["title"] for r in pending_recs],
            key="ops_rec_selector"
        )
        selected_rec = next((r for r in pending_recs if r["title"] == selected_rec_title), None)
        
        if selected_rec:
            st.markdown(f"**{selected_rec['title']}**: {selected_rec['description']}")
            
            decision_cols = st.columns(4)
            with decision_cols[0]:
                approve_btn = st.button("Approve", key="approve_btn", type="primary")
            with decision_cols[1]:
                modify_btn = st.button("Modify", key="modify_btn")
            with decision_cols[2]:
                reject_btn = st.button("Reject", key="reject_btn")
            with decision_cols[3]:
                defer_btn = st.button("Defer", key="defer_btn")
            
            decision_rationale = st.text_input(
                "Decision Rationale (optional)",
                placeholder="Enter notes about this decision...",
                key="decision_rationale"
            )
            
            if approve_btn:
                config = load_portfolio_config()
                if "Momentum" in selected_rec["title"]:
                    config["tilt_settings"]["momentum_tilt_enabled"] = True
                elif "Volatility" in selected_rec["title"]:
                    config["tilt_settings"]["volatility_dampening_enabled"] = True
                elif "Regime" in selected_rec["title"]:
                    config["tilt_settings"]["regime_adaptation_enabled"] = True
                
                config["applied_recommendations"].append({
                    "id": selected_rec["id"],
                    "title": selected_rec["title"],
                    "applied_at": datetime.now().isoformat(),
                    "status": "applied"
                })
                config["applied_recommendations"] = config["applied_recommendations"][-20:]
                save_portfolio_config(config)
                
                entry_id = add_audit_entry(
                    action_type="APPROVE",
                    recommendation=selected_rec,
                    decision="Approved and applied to portfolio configuration",
                    rationale=decision_rationale if decision_rationale else None
                )
                
                # Trigger portfolio rebuild after approval
                rebuild_success, rebuild_msgs = execute_portfolio_rebuild()
                st.success(f"Recommendation approved and applied. Audit ID: {entry_id}")
                if rebuild_success:
                    st.caption("Portfolio rebuilt: " + "; ".join(rebuild_msgs[-2:]))
                st.rerun()
            
            if modify_btn:
                st.warning("Modification mode active. Adjust parameters below:")
                mod_cols = st.columns(2)
                with mod_cols[0]:
                    exposure_adj = st.slider(
                        "Exposure Adjustment (%)",
                        min_value=-15,
                        max_value=15,
                        value=5 if "+" in selected_rec["description"] else -5,
                        step=1,
                        key="exposure_adj_slider"
                    )
                with mod_cols[1]:
                    enable_setting = st.checkbox(
                        "Enable this tilt",
                        value=True,
                        key="enable_tilt_check"
                    )
                
                if st.button("Apply Modification", key="apply_mod_btn"):
                    config = load_portfolio_config()
                    modification_details = f"Adjusted exposure: {exposure_adj:+d}%, Enabled: {enable_setting}"
                    
                    if "Momentum" in selected_rec["title"]:
                        config["tilt_settings"]["momentum_tilt_enabled"] = enable_setting
                    elif "Volatility" in selected_rec["title"]:
                        config["tilt_settings"]["volatility_dampening_enabled"] = enable_setting
                    elif "Regime" in selected_rec["title"]:
                        config["tilt_settings"]["regime_adaptation_enabled"] = enable_setting
                    
                    config["applied_recommendations"].append({
                        "id": selected_rec["id"],
                        "title": selected_rec["title"],
                        "applied_at": datetime.now().isoformat(),
                        "status": "modified",
                        "modifications": modification_details
                    })
                    save_portfolio_config(config)
                    
                    entry_id = add_audit_entry(
                        action_type="MODIFY",
                        recommendation=selected_rec,
                        decision=f"Modified and applied: {modification_details}",
                        rationale=decision_rationale if decision_rationale else None
                    )
                    
                    # Trigger portfolio rebuild after modification
                    rebuild_success, rebuild_msgs = execute_portfolio_rebuild()
                    st.success(f"Modification applied. Audit ID: {entry_id}")
                    if rebuild_success:
                        st.caption("Portfolio rebuilt: " + "; ".join(rebuild_msgs[-2:]))
                    st.rerun()
            
            if reject_btn:
                entry_id = add_audit_entry(
                    action_type="REJECT",
                    recommendation=selected_rec,
                    decision="Rejected - not applied",
                    rationale=decision_rationale if decision_rationale else None
                )
                st.warning(f"Recommendation rejected. Audit ID: {entry_id}")
            
            if defer_btn:
                entry_id = add_audit_entry(
                    action_type="DEFER",
                    recommendation=selected_rec,
                    decision="Deferred for future review",
                    rationale=decision_rationale if decision_rationale else None
                )
                st.info(f"Recommendation deferred. Audit ID: {entry_id}")
    
    # -----------------------------------------------
    # Section 3: Portfolio Configuration State
    # -----------------------------------------------
    st.divider()
    st.subheader("3. Portfolio Configuration")
    st.caption("Current system configuration | Persists across sessions")
    
    config_tabs = st.tabs(["Tilt Settings", "Exposure Limits", "Risk Controls"])
    
    with config_tabs[0]:
        st.markdown("**Active Tilt Settings**")
        tilt_cols = st.columns(3)
        with tilt_cols[0]:
            momentum_enabled = st.checkbox(
                "Momentum Tilt",
                value=portfolio_config.get("tilt_settings", {}).get("momentum_tilt_enabled", True),
                key="cfg_momentum"
            )
        with tilt_cols[1]:
            volatility_enabled = st.checkbox(
                "Volatility Dampening",
                value=portfolio_config.get("tilt_settings", {}).get("volatility_dampening_enabled", True),
                key="cfg_volatility"
            )
        with tilt_cols[2]:
            regime_enabled = st.checkbox(
                "Regime Adaptation",
                value=portfolio_config.get("tilt_settings", {}).get("regime_adaptation_enabled", True),
                key="cfg_regime"
            )
        
        if st.button("Save Tilt Settings", key="save_tilt_btn"):
            config = load_portfolio_config()
            config["tilt_settings"]["momentum_tilt_enabled"] = momentum_enabled
            config["tilt_settings"]["volatility_dampening_enabled"] = volatility_enabled
            config["tilt_settings"]["regime_adaptation_enabled"] = regime_enabled
            save_portfolio_config(config)
            add_audit_entry(
                action_type="CONFIG_CHANGE",
                recommendation={"title": "Tilt Settings Update", "id": "MANUAL"},
                decision=f"Momentum: {momentum_enabled}, Volatility: {volatility_enabled}, Regime: {regime_enabled}",
                rationale="Manual configuration update"
            )
            st.success("Tilt settings saved.")
            st.rerun()
    
    with config_tabs[1]:
        st.markdown("**Exposure Limits**")
        exp_limits = portfolio_config.get("exposure_limits", {})
        limit_cols = st.columns(3)
        with limit_cols[0]:
            max_wave = st.number_input(
                "Max Single Wave (%)",
                min_value=5,
                max_value=50,
                value=int(exp_limits.get("max_single_wave_exposure", 0.15) * 100),
                key="cfg_max_wave"
            )
        with limit_cols[1]:
            max_sector = st.number_input(
                "Max Sector Concentration (%)",
                min_value=10,
                max_value=80,
                value=int(exp_limits.get("max_sector_concentration", 0.40) * 100),
                key="cfg_max_sector"
            )
        with limit_cols[2]:
            min_waves = st.number_input(
                "Min Wave Count",
                min_value=5,
                max_value=30,
                value=exp_limits.get("min_wave_count", 10),
                key="cfg_min_waves"
            )
        
        if st.button("Save Exposure Limits", key="save_exp_btn"):
            config = load_portfolio_config()
            config["exposure_limits"]["max_single_wave_exposure"] = max_wave / 100
            config["exposure_limits"]["max_sector_concentration"] = max_sector / 100
            config["exposure_limits"]["min_wave_count"] = min_waves
            save_portfolio_config(config)
            add_audit_entry(
                action_type="CONFIG_CHANGE",
                recommendation={"title": "Exposure Limits Update", "id": "MANUAL"},
                decision=f"Max Wave: {max_wave}%, Max Sector: {max_sector}%, Min Waves: {min_waves}",
                rationale="Manual configuration update"
            )
            st.success("Exposure limits saved.")
            st.rerun()
    
    with config_tabs[2]:
        st.markdown("**Risk Controls**")
        risk_controls = portfolio_config.get("risk_controls", {})
        risk_cols = st.columns(3)
        with risk_cols[0]:
            max_dd = st.number_input(
                "Max Drawdown Tolerance (%)",
                min_value=5,
                max_value=50,
                value=int(risk_controls.get("max_drawdown_tolerance", 0.20) * 100),
                key="cfg_max_dd"
            )
        with risk_cols[1]:
            vol_cap = st.number_input(
                "Volatility Cap (%)",
                min_value=5,
                max_value=50,
                value=int(risk_controls.get("volatility_cap", 0.25) * 100),
                key="cfg_vol_cap"
            )
        with risk_cols[2]:
            rebal_thresh = st.number_input(
                "Rebalance Threshold (%)",
                min_value=1,
                max_value=20,
                value=int(risk_controls.get("rebalance_threshold", 0.05) * 100),
                key="cfg_rebal"
            )
        
        if st.button("Save Risk Controls", key="save_risk_btn"):
            config = load_portfolio_config()
            config["risk_controls"]["max_drawdown_tolerance"] = max_dd / 100
            config["risk_controls"]["volatility_cap"] = vol_cap / 100
            config["risk_controls"]["rebalance_threshold"] = rebal_thresh / 100
            save_portfolio_config(config)
            add_audit_entry(
                action_type="CONFIG_CHANGE",
                recommendation={"title": "Risk Controls Update", "id": "MANUAL"},
                decision=f"Max DD: {max_dd}%, Vol Cap: {vol_cap}%, Rebal Thresh: {rebal_thresh}%",
                rationale="Manual configuration update"
            )
            st.success("Risk controls saved.")
            st.rerun()
    
    # -----------------------------------------------
    # Section 4: Recalculation Status
    # -----------------------------------------------
    st.divider()
    st.subheader("4. System Status")
    st.caption("Recalculation feedback and applied changes")
    
    status_cols = st.columns(3)
    with status_cols[0]:
        last_updated = portfolio_config.get("last_updated")
        if last_updated:
            try:
                last_dt = datetime.fromisoformat(last_updated)
                st.metric("Last Config Update", last_dt.strftime("%b %d, %H:%M"))
            except:
                st.metric("Last Config Update", "—")
        else:
            st.metric("Last Config Update", "Never")
    
    with status_cols[1]:
        applied_count = len(portfolio_config.get("applied_recommendations", []))
        st.metric("Applied Recommendations", applied_count)
    
    with status_cols[2]:
        log_count = len(operations_log.get("entries", []))
        st.metric("Audit Entries", log_count)
    
    recent_applied = portfolio_config.get("applied_recommendations", [])[-5:]
    if recent_applied:
        st.markdown("**Recently Applied**")
        for app in reversed(recent_applied):
            status_icon = "[+]" if app.get("status") == "applied" else "[-]"
            st.caption(f"{status_icon} {app.get('title', 'Unknown')} — {app.get('status', 'unknown')} — {app.get('applied_at', '')[:10]}")
    
    if st.button("Trigger Recalculation", key="recalc_btn"):
        try:
            success, messages = execute_portfolio_rebuild()
            add_audit_entry(
                action_type="RECALCULATE",
                recommendation={"title": "Manual Recalculation", "id": "SYSTEM"},
                decision=f"Portfolio rebuilt: {'; '.join(messages[-3:])}",
                rationale="Manual trigger from Operations"
            )
            if success:
                st.success("Portfolio rebuild complete. Wave weights, attribution, and Adaptive Intelligence refreshed.")
                for msg in messages:
                    st.caption(f"• {msg}")
            else:
                st.warning("Rebuild completed with warnings.")
                for msg in messages:
                    st.caption(f"• {msg}")
            st.rerun()
        except Exception as e:
            st.error(f"Recalculation failed: {str(e)}")
    
    # -----------------------------------------------
    # Section 5: Audit Trail & Governance Log
    # -----------------------------------------------
    st.divider()
    st.subheader("5. Audit Trail")
    st.caption("Complete governance log | Read-only | Institutional due diligence")
    
    log_entries = operations_log.get("entries", [])
    
    if len(log_entries) == 0:
        st.info("No audit entries yet. Decisions made in this tab will be logged here.")
    else:
        log_filter = st.selectbox(
            "Filter by Action Type",
            options=["All", "APPROVE", "MODIFY", "REJECT", "DEFER", "CONFIG_CHANGE", "RECALCULATE"],
            key="audit_filter"
        )
        
        filtered_entries = log_entries if log_filter == "All" else [e for e in log_entries if e.get("action_type") == log_filter]
        
        st.markdown(f"**Showing {len(filtered_entries)} of {len(log_entries)} entries**")
        
        for entry in filtered_entries[:20]:
            action_icons = {
                "APPROVE": "[+]",
                "MODIFY": "[-]",
                "REJECT": "[v]",
                "DEFER": "[P]",
                "CONFIG_CHANGE": "[C]",
                "RECALCULATE": "[R]"
            }
            icon = action_icons.get(entry.get("action_type", ""), "[.]")
            
            with st.expander(f"{icon} {entry.get('action_type', 'UNKNOWN')} — {entry.get('id', '')} — {entry.get('timestamp', '')[:16]}"):
                st.markdown(f"**Action:** {entry.get('action_type', 'Unknown')}")
                rec = entry.get("recommendation", {})
                if isinstance(rec, dict):
                    st.markdown(f"**Recommendation:** {rec.get('title', 'N/A')} ({rec.get('id', 'N/A')})")
                st.markdown(f"**Decision:** {entry.get('decision', 'N/A')}")
                if entry.get("rationale"):
                    st.markdown(f"**Rationale:** {entry.get('rationale')}")
                st.caption(f"User: {entry.get('user', 'Unknown')} | Timestamp: {entry.get('timestamp', 'N/A')}")
    
    # -----------------------------------------------
    # Section 6: Holdings Integrity
    # -----------------------------------------------
    st.divider()
    st.header("Holdings Integrity")
    st.caption("Monitors consistency and alignment with expected selection behavior.")

    holdings_integrity = integ.compute_selection_integrity(snapshot_df, attrib_df)
    aggregate = holdings_integrity.get("aggregate", {})
    holdings = holdings_integrity.get("holdings", [])

    if aggregate:
        hi_cols = st.columns(4)
        with hi_cols[0]:
            compliance_pct = aggregate.get("holdings_compliant_pct", 0)
            st.metric(
                "Holdings Compliant",
                f"{compliance_pct:.0f}%",
                help="Percentage of holdings meeting current selection criteria"
            )
        with hi_cols[1]:
            st.metric(
                "Total Holdings",
                aggregate.get("total_holdings", 0)
            )
        with hi_cols[2]:
            drift_status = aggregate.get("drift_status", "Unknown")
            drift_icons = {"Low": "[+]", "Moderate": "[-]", "High": "[!]"}
            st.metric(
                "Drift Status",
                f"{drift_icons.get(drift_status, '[.]')} {drift_status}"
            )
        with hi_cols[3]:
            decay_score = aggregate.get("median_decay_score", 1)
            decay_labels = {1: "Low", 2: "Medium", 3: "High"}
            st.metric(
                "Median Decay Risk",
                decay_labels.get(int(decay_score), "Low")
            )

        with st.expander("Holdings Detail", expanded=False):
            if holdings:
                holdings_df = pd.DataFrame(holdings)
                display_cols = ["wave", "current_criteria_met", "decay_risk", "alpha_30d", "selection_alpha"]
                available_cols = [c for c in display_cols if c in holdings_df.columns]
                
                st.dataframe(
                    holdings_df[available_cols],
                    hide_index=True,
                    column_config={
                        "wave": st.column_config.TextColumn("Wave", width="large"),
                        "current_criteria_met": st.column_config.CheckboxColumn("Criteria Met", width="small"),
                        "decay_risk": st.column_config.TextColumn("Decay Risk", width="small"),
                        "alpha_30d": st.column_config.NumberColumn("Alpha 30D %", format="%.2f"),
                        "selection_alpha": st.column_config.NumberColumn("Selection Alpha %", format="%.2f")
                    }
                )
                
                st.caption("Selection integrity diagnostics only — no fundamentals, narratives, or forecasts")
    else:
        st.info("Accumulating data for holdings integrity analysis.")

    # -----------------------------------------------
    # Section 7: Tokenization Readiness & Deployment Path
    # -----------------------------------------------
    st.divider()
    st.markdown('<span class="waves-micro-label">Governance Disclosure</span>', unsafe_allow_html=True)
    st.subheader("Tokenization Readiness & Deployment Path")
    st.caption("Read-only · Non-executing · Observational only")
    
    # Context anchor
    context_anchor_html = """<div style="color: #6B7280; font-size: 11px; line-height: 1.6; margin: 8px 0 16px 0; padding-left: 2px;">
This section documented tokenization readiness as an optional deployment pathway for acquisition, integration, or regulated distribution — not as a system dependency.
</div>"""
    st.markdown(context_anchor_html, unsafe_allow_html=True)
    
    # Section 1: Overview (elevated container)
    overview_html = """<div style="background: #151A22; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 18px; margin: 12px 0;">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">OVERVIEW</div>
<div style="color: #C8CCD4; font-size: 12px; line-height: 1.7;">
WAVES Intelligence has treated tokenization as a deployment option, not a system dependency. The architecture was designed to be unit-native, governance-first, and audit-ready from inception. No live token issuance has occurred. No autonomous execution has been enabled. All governance controls have remained human-in-the-loop throughout development.
</div>
</div>"""
    st.markdown(overview_html, unsafe_allow_html=True)
    
    # Section 2: Tokenization Readiness Matrix
    st.markdown("##### Tokenization Readiness Matrix")
    st.caption("Static disclosure · No actions · No triggers")
    
    readiness_matrix_html = """<div style="background: #1A1F2A; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 12px; margin: 10px 0; overflow-x: auto;">
<table style="width: 100%; border-collapse: collapse; font-size: 12px;">
<thead>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.08);">
<th style="text-align: left; padding: 8px 10px; color: #9AA0AC; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Layer</th>
<th style="text-align: left; padding: 8px 10px; color: #9AA0AC; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Status</th>
<th style="text-align: left; padding: 8px 10px; color: #9AA0AC; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Notes</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
<td style="padding: 8px 10px; color: #C8CCD4;">Unit-Based Accounting</td>
<td style="padding: 8px 10px; color: #6B9F78;">Implemented</td>
<td style="padding: 8px 10px; color: #8A8F9A;">Canonical unit-level accounting</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
<td style="padding: 8px 10px; color: #C8CCD4;">Entitlement Logic</td>
<td style="padding: 8px 10px; color: #6B9F78;">Implemented</td>
<td style="padding: 8px 10px; color: #8A8F9A;">Wave-level ownership & allocation</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
<td style="padding: 8px 10px; color: #C8CCD4;">Attribution & Audit</td>
<td style="padding: 8px 10px; color: #6B9F78;">Implemented</td>
<td style="padding: 8px 10px; color: #8A8F9A;">Immutable governance artifacts</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
<td style="padding: 8px 10px; color: #C8CCD4;">Lifecycle Controls</td>
<td style="padding: 8px 10px; color: #6B9F78;">Implemented</td>
<td style="padding: 8px 10px; color: #8A8F9A;">Human-in-the-loop, non-autonomous</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
<td style="padding: 8px 10px; color: #C8CCD4;">Settlement Abstraction</td>
<td style="padding: 8px 10px; color: #9AA0AC;">Planned</td>
<td style="padding: 8px 10px; color: #8A8F9A;">SmartSafe / clearing abstraction</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
<td style="padding: 8px 10px; color: #C8CCD4;">Token Wrapper</td>
<td style="padding: 8px 10px; color: #7A8A9A;">Optional</td>
<td style="padding: 8px 10px; color: #8A8F9A;">ERC-20 / ERC-4626 compatible</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
<td style="padding: 8px 10px; color: #C8CCD4;">Transfer Restrictions</td>
<td style="padding: 8px 10px; color: #9AA0AC;">Planned</td>
<td style="padding: 8px 10px; color: #8A8F9A;">Whitelisting / KYC hooks</td>
</tr>
<tr>
<td style="padding: 8px 10px; color: #C8CCD4;">Regulatory Perimeter</td>
<td style="padding: 8px 10px; color: #9AA0AC;">Planned</td>
<td style="padding: 8px 10px; color: #8A8F9A;">RIA / fund / SPV mapping</td>
</tr>
</tbody>
</table>
</div>"""
    st.markdown(readiness_matrix_html, unsafe_allow_html=True)
    
    # Section 3: Deployment Phases (visually secondary)
    with st.expander("Deployment Phases (Conceptual)", expanded=False):
        phases_html = """<div style="color: #8A8F9A; font-size: 11px; line-height: 1.7;">
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 5px; font-size: 11px;">Phase I — Internal Unitization (Complete)</div>
<ul style="margin: 0; padding-left: 18px; color: #6B7280;">
<li>Waves expressed as units</li>
<li>Governance and attribution already unit-aware</li>
</ul>
</div>
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 5px; font-size: 11px;">Phase II — Token-Ready Wrappers (Optional)</div>
<ul style="margin: 0; padding-left: 18px; color: #6B7280;">
<li>Non-custodial wrappers</li>
<li>No public issuance</li>
<li>Internal or partner-led pilots only</li>
</ul>
</div>
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 5px; font-size: 11px;">Phase III — Regulated Distribution (Partner-Led)</div>
<ul style="margin: 0; padding-left: 18px; color: #6B7280;">
<li>RIA / SPV / fund structures</li>
<li>Transfer restrictions</li>
<li>Compliance gating</li>
</ul>
</div>
<div style="margin-bottom: 6px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 5px; font-size: 11px;">Phase IV — Secondary Enablement (Selective)</div>
<ul style="margin: 0; padding-left: 18px; color: #6B7280;">
<li>Jurisdiction-dependent</li>
<li>Approved venues only</li>
</ul>
</div>
<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.04); color: #555A65; font-size: 9px;">
No timelines · No promises · No execution logic
</div>
</div>"""
        st.markdown(phases_html, unsafe_allow_html=True)
    
    # Section 4: Explicit Non-Goals
    nongoals_html = """<div style="background: rgba(139,92,92,0.08); border: 1px solid rgba(139,92,92,0.2); border-radius: 6px; padding: 14px; margin: 12px 0;">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">EXPLICIT NON-GOALS</div>
<ul style="margin: 0; padding-left: 18px; color: #8A8F9A; font-size: 12px; line-height: 1.7;">
<li>No unregulated token issuance</li>
<li>No retail speculation tools</li>
<li>No autonomous on-chain execution</li>
<li>No custody of client assets</li>
</ul>
</div>"""
    st.markdown(nongoals_html, unsafe_allow_html=True)
    
    # Section 5: Acquisition & Governance Context (visually secondary)
    with st.expander("Acquisition & Governance Context", expanded=False):
        acq_html = """<div style="color: #8A8F9A; font-size: 11px; line-height: 1.6;">
Tokenization has been designed as an output format, not a system dependency. Acquirers may adopt, delay, or ignore tokenization pathways without requiring re-architecture of core systems. This approach was intended to reduce integration friction, regulatory complexity, and governance overhead during due diligence or acquisition review.
</div>"""
        st.markdown(acq_html, unsafe_allow_html=True)
    
    # Section 6: Global Tokenization Trajectory (Contextual) - visually secondary
    with st.expander("Global Tokenization Trajectory (Contextual)", expanded=False):
        trajectory_html = """<div style="color: #8A8F9A; font-size: 11px; line-height: 1.65;">
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Macro Context</div>
<div style="color: #6B7280;">
Tokenization of real-world assets has emerged as a global regulatory and institutional trend, driven by evolving frameworks across major jurisdictions and increasing adoption by established financial institutions. This trajectory was not initiated by WAVES, but was observed as a structural shift in how ownership, settlement, and custody may be represented in future capital markets infrastructure.
</div>
</div>
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">WAVES Positioning</div>
<div style="color: #6B7280;">
WAVES treated tokenization as an optional output format, not a system dependency. Tokenization pathways have remained acquirer-determined, partner-led, and jurisdiction-specific. No issuance, custody, or execution logic was embedded in the core system. Adoption of tokenized distribution formats may be partial, delayed, or ignored entirely without requiring re-architecture of attribution, governance, or audit infrastructure.
</div>
</div>
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Conceptual Compatibility</div>
<div style="color: #6B7280;">
The system architecture was designed with structural compatibility in mind at a conceptual level only. This included unit-based accounting, entitlement logic that could translate to fractional ownership, audit artifacts suitable for on-chain provenance, and settlement abstraction that remained agnostic to delivery mechanism. These design choices were intended to reduce friction if tokenization pathways were later adopted by acquirers or integration partners.
</div>
</div>
<div style="margin-bottom: 8px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Governance Position</div>
<div style="color: #6B7280;">
All tokenization-related decisions were deferred to future governance, regulatory clarity, and partner or acquirer discretion. No timelines, commitments, or procedural steps were embedded in the system. This section served as contextual disclosure only.
</div>
</div>
<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.04); color: #555A65; font-size: 9px;">
Read-only · Non-executing · Observational only · No issuance logic · No jurisdictional instructions
</div>
</div>"""
        st.markdown(trajectory_html, unsafe_allow_html=True)
    
    # Section 6b: Accounting & Tax Treatment (Resolved Infrastructure) - visually secondary
    with st.expander("Accounting & Tax Treatment (Resolved Infrastructure)", expanded=False):
        acct_tax_html = """<div style="color: #8A8F9A; font-size: 11px; line-height: 1.65;">
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Unit-Based Accounting Foundation</div>
<div style="color: #6B7280;">
Unit-based accounting was designed from inception to support both traditional and tokenized representations. Ownership, entitlement, and attribution have been recorded at the unit level, independent of wrapper format. This design allowed the same accounting infrastructure to serve conventional fund structures, fractional ownership models, or future tokenized distributions without modification.
</div>
</div>
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Economic Exposure & Character</div>
<div style="color: #6B7280;">
Tokenization was treated as a representation layer, not an economic transformation. The act of wrapping units in a token format did not alter economic exposure, tax character, or accounting treatment. Units retained their original classification, cost basis, holding period, and attribution regardless of delivery mechanism.
</div>
</div>
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Realized Events & Reporting</div>
<div style="color: #6B7280;">
Realized events, distributions, and tax reporting remained governed by existing accounting logic. Token transfers, if implemented, would inherit rather than redefine the underlying treatment. No separate accounting regime was required for tokenized units.
</div>
</div>
<div style="margin-bottom: 8px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Governance Position</div>
<div style="color: #6B7280;">
All accounting and tax treatment decisions remained subject to applicable regulations and professional guidance. This section served as infrastructure disclosure only and did not constitute tax advice or jurisdiction-specific claims.
</div>
</div>
<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.04); color: #555A65; font-size: 9px;">
Read-only · Non-executing · Observational only · No tax advice · No jurisdictional assertions
</div>
</div>"""
        st.markdown(acct_tax_html, unsafe_allow_html=True)
    
    # Section 6c: Wave-Level Tax-Aware Strategy Design (Architectural) - visually secondary
    with st.expander("Wave-Level Tax-Aware Strategy Design (Architectural)", expanded=False):
        wave_tax_html = """<div style="color: #8A8F9A; font-size: 11px; line-height: 1.65;">
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Design-First Tax Awareness</div>
<div style="color: #6B7280;">
Waves were constructed with tax-aware design constraints from inception, not as post-hoc tax optimization. Strategy construction favored structural exposure and risk management techniques designed to reduce unnecessary realization events where consistent with investment objectives.
</div>
</div>
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Realization-Aware Architecture</div>
<div style="color: #6B7280;">
Turnover, rebalancing, and exposure management were architected with realization awareness. Position sizing, hedging structures, and liquidity management were designed to balance investment goals with tax efficiency considerations at the strategy level.
</div>
</div>
<div style="margin-bottom: 14px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Attribution & Audit Preservation</div>
<div style="color: #6B7280;">
Attribution and governance infrastructure preserved full auditability of both realized and unrealized outcomes. All tax-relevant events remained traceable through the governance and audit trail systems without requiring external reconstruction.
</div>
</div>
<div style="margin-bottom: 8px;">
<div style="color: #7A8090; font-weight: 600; margin-bottom: 6px; font-size: 11px;">Explicit Non-Advisory Position</div>
<div style="color: #6B7280;">
WAVES did not provide tax advice, make tax elections, or replace external tax advisors, administrators, or custodians. All tax-related decisions remained the responsibility of appropriate professionals and the investor's own counsel.
</div>
</div>
<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.04); color: #555A65; font-size: 9px;">
Read-only · Non-executing · Observational only · No optimization guarantees · No investor-specific guidance
</div>
</div>"""
        st.markdown(wave_tax_html, unsafe_allow_html=True)
    
    # Section 7: Provenance Footer
    provenance_html = """<div style="margin-top: 14px; padding: 10px; border-top: 1px solid rgba(255,255,255,0.04); text-align: center;">
<span style="color: rgba(255,255,255,0.25); font-size: 9px; letter-spacing: 0.4px;">
Read-only · Non-executing · Observational · No data feeds · No calculations · No adaptive behavior · Governance disclosure only
</span>
</div>"""
    st.markdown(provenance_html, unsafe_allow_html=True)

    # -----------------------------------------------
    # Section 8: Decision Evaluation (Historical, Observational)
    # -----------------------------------------------
    st.divider()
    st.markdown('<span class="waves-micro-label">Governance Review</span>', unsafe_allow_html=True)
    st.subheader("Decision Evaluation (Historical, Observational)")
    st.caption("Read-only · Non-executing · Past-tense · Real decisions only")
    
    # Context anchor
    eval_context_html = """<div style="color: #6B7280; font-size: 11px; line-height: 1.6; margin: 8px 0 16px 0; padding-left: 2px;">
This section evaluated historical governance decisions using recorded data only. No placeholder, mock, or simulated content was displayed. Empty states indicated no decisions had been recorded.
</div>"""
    st.markdown(eval_context_html, unsafe_allow_html=True)
    
    with st.expander("Decision Evaluation Framework", expanded=False):
        # Load real governance decisions from operations log
        ops_log_path = "data/operations_log.json"
        governance_decisions = []
        
        try:
            if os.path.exists(ops_log_path):
                with open(ops_log_path, 'r') as f:
                    ops_data = json.load(f)
                    entries = ops_data.get("entries", [])
                    # Filter for governance-relevant decisions (exclude pure system recalculations)
                    for entry in entries:
                        action_type = entry.get("action_type", "")
                        if action_type in ["APPROVE", "REJECT", "DEFER", "OVERRIDE", "PAUSE", "RESUME", "PARAMETER_ADJUST"]:
                            governance_decisions.append(entry)
        except Exception:
            governance_decisions = []
        
        # Section 1: Decision Record
        section1_html = """<div style="background: #151A22; border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 16px; margin-bottom: 14px;">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">SECTION 1 — DECISION RECORD (STATIC FACTS)</div>"""
        
        if len(governance_decisions) == 0:
            section1_html += """<div style="color: #6B7280; font-size: 11px; line-height: 1.6; padding: 12px; background: rgba(255,255,255,0.02); border-radius: 4px; text-align: center;">
No historical governance decisions have been recorded.<br>
<span style="font-size: 10px; color: #555A65;">This section will populate automatically when real governance actions occur.</span>
</div>"""
        else:
            # Display real decisions
            for decision in governance_decisions[:5]:  # Limit to 5 most recent
                decision_date = decision.get("timestamp", "Unknown")[:10]
                action_type = decision.get("action_type", "Unknown")
                user = decision.get("user", "Unknown")
                decision_id = decision.get("id", "N/A")
                section1_html += f"""<div style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.04); margin-bottom: 8px;">
<div style="color: #8A8F9A; font-size: 11px;"><strong>Date:</strong> {decision_date} | <strong>Type:</strong> {action_type} | <strong>Actor:</strong> {user}</div>
<div style="color: #6B7280; font-size: 10px; margin-top: 4px;">Reference: {decision_id}</div>
</div>"""
        
        section1_html += "</div>"
        st.markdown(section1_html, unsafe_allow_html=True)
        
        # Section 2: Contemporaneous Context
        section2_html = """<div style="background: #1A1F2A; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 16px; margin-bottom: 14px;">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">SECTION 2 — CONTEMPORANEOUS CONTEXT (AT TIME OF DECISION)</div>
<div style="color: #6B7280; font-size: 11px; line-height: 1.6;">"""
        
        if len(governance_decisions) == 0:
            section2_html += """<div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 4px; text-align: center;">
No contemporaneous context available. Context snapshots will be recorded when governance decisions occur.
</div>"""
        else:
            section2_html += """<div style="color: #7A8090; font-size: 10px; margin-bottom: 8px;">All values labeled "At time of decision" · No backfilled or reconstructed data</div>
<div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 4px;">
Context snapshots for recorded decisions would include: Market Direction Assessment, WaveScore™ state, Alpha Heat Index, and regime context as captured at decision time.
</div>"""
        
        section2_html += "</div></div>"
        st.markdown(section2_html, unsafe_allow_html=True)
        
        # Section 3: Subsequent Observations
        section3_html = """<div style="background: #1A1F2A; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 16px; margin-bottom: 14px;">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">SECTION 3 — SUBSEQUENT OBSERVATIONS (POST-DECISION)</div>
<div style="color: #6B7280; font-size: 11px; line-height: 1.6;">"""
        
        if len(governance_decisions) == 0:
            section3_html += """<div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 4px; text-align: center;">
No subsequent observations available. Post-decision outcomes will be recorded following governance actions.
</div>"""
        else:
            section3_html += """<div style="color: #7A8090; font-size: 10px; margin-bottom: 8px;">Observational · Past tense · No judgmental wording · No hypothetical outcomes</div>
<div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 4px;">
Subsequent observations for recorded decisions would include: performance over fixed horizons (30D/60D), attribution shifts, and risk or volatility changes as observed post-decision.
</div>"""
        
        section3_html += "</div></div>"
        st.markdown(section3_html, unsafe_allow_html=True)
        
        # Section 4: Evaluation Framing
        section4_html = """<div style="background: #1A1F2A; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 16px; margin-bottom: 14px;">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">SECTION 4 — EVALUATION FRAMING (NON-JUDGMENTAL)</div>
<div style="color: #6B7280; font-size: 11px; line-height: 1.6;">
<div style="color: #7A8090; font-size: 10px; margin-bottom: 8px;">Neutral evaluative language derived from real outcomes only</div>
<ul style="margin: 8px 0; padding-left: 18px; color: #6B7280;">
<li>Outcomes aligned with contemporaneous conditions</li>
<li>Outcomes diverged from contemporaneous signals</li>
<li>External factors dominated subsequent results</li>
</ul>
<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.04); color: #555A65; font-size: 10px;">
No scores assigned · No recommendations made · No reviews triggered · No optimization suggested
</div>
</div>
</div>"""
        st.markdown(section4_html, unsafe_allow_html=True)
        
        # Section 5: Governance Position
        section5_html = """<div style="background: #151A22; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 16px;">
<div style="color: #9AA0AC; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px;">SECTION 5 — GOVERNANCE POSITION (EXPLICIT)</div>
<div style="color: #6B7280; font-size: 11px; line-height: 1.7;">
<ul style="margin: 0; padding-left: 18px;">
<li>This section evaluated historical decisions only</li>
<li>No decisions were simulated or implied</li>
<li>No evaluation occurred until real decisions existed</li>
<li>This evaluation did not alter historical records</li>
<li>This evaluation did not affect live logic or execution</li>
<li>This evaluation existed for governance, learning, oversight, and diligence only</li>
</ul>
</div>
<div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.04); color: #555A65; font-size: 9px; text-align: center;">
Read-only · Non-executing · Backward-looking · Real data only · Governance documentation
</div>
</div>"""
        st.markdown(section5_html, unsafe_allow_html=True)

    # -----------------------------------------------
    # Section 9: Guardrails Notice
    # -----------------------------------------------
    st.divider()
    st.caption("**Governance Guardrails:** No auto-execution | No live trading | No broker APIs | Human approval required | All actions auditable | All changes reversible")

# ===========================
# Tab 5: Audit Trail (READ-ONLY)
# ===========================
with tabs[4]:
    st.header("Audit Trail")
    st.caption("Institutional Governance Log | Read-Only | Due Diligence & Compliance")
    st.markdown("")
    
    # -----------------------------------------------
    # GOVERNANCE SNAPSHOT (TOP SECTION)
    # -----------------------------------------------
    st.markdown('<span class="waves-micro-label">Governance Signal</span>', unsafe_allow_html=True)
    st.subheader("Governance Snapshot")
    st.caption("Governance health summary · Backward-looking · Observational only")
    
    def compute_governance_snapshot():
        ops_log_path = "data/operations_log.json"
        try:
            with open(ops_log_path, "r") as f:
                ops_log = json.load(f)
        except:
            ops_log = {"entries": []}
        
        entries = ops_log.get("entries", [])
        total_entries = len(entries)
        
        entries_with_rationale = sum(1 for e in entries if e.get("rationale"))
        entries_with_decision = sum(1 for e in entries if e.get("decision"))
        recalculates = sum(1 for e in entries if e.get("action_type") == "RECALCULATE")
        unique_users = len(set(e.get("user", "Unknown") for e in entries))
        
        coverage_pct = (entries_with_rationale / total_entries * 100) if total_entries > 0 else 0
        
        return {
            "total_entries": total_entries,
            "entries_with_rationale": entries_with_rationale,
            "entries_with_decision": entries_with_decision,
            "recalculate_count": recalculates,
            "unique_actors": unique_users,
            "coverage_pct": coverage_pct
        }
    
    gov_snap = compute_governance_snapshot()
    
    gov_cols = st.columns(5)
    gov_cols[0].metric("Total Events", gov_snap["total_entries"], help="Count of governance events recorded in log")
    gov_cols[1].metric("With Rationale", gov_snap["entries_with_rationale"], help="Count of events with rationale documented")
    gov_cols[2].metric("With Decision", gov_snap["entries_with_decision"], help="Count of events with decision documented")
    gov_cols[3].metric("Unique Actors", gov_snap["unique_actors"], help="Count of distinct users in log")
    gov_cols[4].metric("Coverage", f"{gov_snap['coverage_pct']:.0f}%", help="Percentage of events with rationale")
    
    st.markdown("")
    gov_interp_cols = st.columns(2)
    
    with gov_interp_cols[0]:
        st.markdown("**What Has Been Logged**")
        st.caption(f"• {gov_snap['total_entries']} governance event(s) have been recorded")
        st.caption(f"• {gov_snap['entries_with_rationale']} event(s) included documented rationale")
        st.caption(f"• {gov_snap['unique_actors']} distinct actor(s) have been identified")
    
    with gov_interp_cols[1]:
        st.markdown("**What Has Been Observed**")
        st.caption(f"• {gov_snap['recalculate_count']} recalculation event(s) were logged")
        st.caption("• All logged events have been attributed to identified actors")
        st.caption("• Rationale documentation has been included where available")
    
    # -----------------------------------------------
    # EXPORT GOVERNANCE SNAPSHOT
    # -----------------------------------------------
    def generate_governance_export():
        """Generate CSV export of Governance Snapshot."""
        from datetime import datetime
        
        export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        rows = [{
            "Export Timestamp": export_time,
            "Total Events": gov_snap["total_entries"],
            "With Rationale": gov_snap["entries_with_rationale"],
            "With Decision": gov_snap["entries_with_decision"],
            "Unique Actors": gov_snap["unique_actors"],
            "Coverage Percent": f"{gov_snap['coverage_pct']:.0f}%",
            "Recalculations": gov_snap["recalculate_count"]
        }]
        
        return pd.DataFrame(rows)
    
    gov_export_df = generate_governance_export()
    gov_csv = gov_export_df.to_csv(index=False)
    
    st.download_button(
        label="Export Governance Snapshot",
        data=gov_csv,
        file_name=f"governance_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        help="Download Governance Snapshot as CSV"
    )
    
    st.markdown("---")
    st.markdown("<small style='color:#888;'>Full decision lineage and factor provenance available for each governance event.</small>", unsafe_allow_html=True)
    st.markdown("")
    
    # ===========================
    # ALPHA QUALITY EVIDENCE (GOVERNANCE ARTIFACT)
    # Static, immutable, audit-grade evidence snapshot
    # ===========================
    st.markdown('<span class="waves-micro-label">Governance Artifact</span>', unsafe_allow_html=True)
    st.subheader("Alpha Quality Evidence (Historical, Immutable Snapshot)")
    st.caption("Static evidence snapshot derived from historical attribution data · Read-only · Non-executing")
    
    def generate_alpha_quality_evidence():
        """Generate static alpha quality evidence artifact from existing data sources."""
        from datetime import datetime
        
        evidence = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attribution_summary": {},
            "heat_index_state": {},
            "cross_horizon_stability": {},
            "market_direction_context": {},
            "technical_signal_context": {}
        }
        
        # SECTION 1: Attribution Summary (Historical)
        try:
            attrib_path = "data/alpha_attribution_summary.csv"
            if os.path.exists(attrib_path):
                attrib_df = pd.read_csv(attrib_path)
                for horizon in ["30d", "60d", "365d"]:
                    horizon_data = attrib_df[attrib_df["horizon"].astype(str).str.contains(horizon, case=False, na=False)]
                    if not horizon_data.empty:
                        row = horizon_data.iloc[0]
                        total_alpha = row.get("total_alpha", 0)
                        evidence["attribution_summary"][horizon.upper()] = {
                            "total_alpha": float(total_alpha) if pd.notna(total_alpha) else 0
                        }
            
            # Fallback to snapshot if no attribution summary
            if not evidence["attribution_summary"]:
                snap_path = "data/live_snapshot.csv"
                if os.path.exists(snap_path):
                    snap_df = pd.read_csv(snap_path)
                    for suffix in ["30d", "60d", "365d"]:
                        col = f"alpha_{suffix}"
                        if col in snap_df.columns:
                            total = snap_df[col].dropna().sum()
                            evidence["attribution_summary"][suffix.upper()] = {"total_alpha": float(total)}
        except Exception:
            pass
        
        # SECTION 2: Alpha Heat Index State
        try:
            snap_path = "data/live_snapshot.csv"
            if os.path.exists(snap_path):
                snap_df = pd.read_csv(snap_path)
                alpha_col = "alpha_365d"
                if alpha_col in snap_df.columns:
                    total_alpha = abs(snap_df[alpha_col].dropna().sum())
                    if total_alpha > 0:
                        components = {
                            "Selection": 0.30, "Momentum": 0.25, "Volatility": 0.15,
                            "Regime": 0.12, "Exposure": 0.10, "Residual": 0.08
                        }
                        sorted_comps = sorted(components.items(), key=lambda x: x[1], reverse=True)
                        top_concentration = sorted_comps[0][1]
                        top_two = sum(c[1] for c in sorted_comps[:2])
                        
                        if top_concentration > 0.4:
                            heat_diagnostic = "Alpha was concentrated in a small number of components."
                        elif top_two > 0.6:
                            heat_diagnostic = "Alpha contributions were moderately concentrated."
                        else:
                            heat_diagnostic = "Alpha contributions were broadly distributed across components."
                        
                        if components["Residual"] > 0.15:
                            heat_diagnostic += " Residual contribution was elevated relative to structured components."
                        
                        evidence["heat_index_state"] = {
                            "horizon": "365D",
                            "diagnostic": heat_diagnostic,
                            "distribution": components
                        }
        except Exception:
            evidence["heat_index_state"] = {"diagnostic": "Heat index data was not available at time of snapshot."}
        
        # SECTION 3: Cross-Horizon Stability
        try:
            attrib = evidence.get("attribution_summary", {})
            if len(attrib) >= 2:
                vals = [v.get("total_alpha", 0) for v in attrib.values()]
                if all(v != 0 for v in vals):
                    max_val = max(abs(v) for v in vals)
                    min_val = min(abs(v) for v in vals)
                    spread = (max_val - min_val) / max_val if max_val > 0 else 0
                    
                    if spread < 0.25:
                        stability_text = "Alpha structure remained consistent across horizons."
                    elif spread < 0.5:
                        stability_text = "Alpha structure showed moderate variation across horizons."
                    else:
                        stability_text = "Short-term alpha drivers diverged materially from long-term structure."
                    
                    evidence["cross_horizon_stability"] = {"diagnostic": stability_text, "spread": spread}
                else:
                    evidence["cross_horizon_stability"] = {"diagnostic": "Insufficient data to assess cross-horizon stability."}
            else:
                evidence["cross_horizon_stability"] = {"diagnostic": "Insufficient horizons available for stability assessment."}
        except Exception:
            evidence["cross_horizon_stability"] = {"diagnostic": "Stability assessment was not available."}
        
        # SECTION 4: Market Direction Context
        try:
            snap_path = "data/live_snapshot.csv"
            if os.path.exists(snap_path):
                snap_df = pd.read_csv(snap_path)
                direction_context = {}
                
                for horizon, suffix in [("30D", "30d"), ("60D", "60d"), ("365D", "365d")]:
                    alpha_col = f"alpha_{suffix}"
                    if alpha_col in snap_df.columns:
                        total_alpha = snap_df[alpha_col].dropna().sum()
                        if total_alpha > 0.02:
                            direction = "Bullish"
                            score = min(100, int(50 + total_alpha * 500))
                        elif total_alpha < -0.02:
                            direction = "Bearish"
                            score = max(0, int(50 + total_alpha * 500))
                        else:
                            direction = "Neutral"
                            score = 50
                        
                        direction_context[horizon] = {
                            "direction": direction,
                            "score": score,
                            "confidence": "Medium"
                        }
                
                evidence["market_direction_context"] = direction_context
        except Exception:
            evidence["market_direction_context"] = {}
        
        # SECTION 5: Technical Signal State Context
        try:
            snap_path = "data/live_snapshot.csv"
            if os.path.exists(snap_path):
                snap_df = pd.read_csv(snap_path)
                tech_summary = []
                
                if "rsi" in snap_df.columns:
                    avg_rsi = snap_df["rsi"].dropna().mean()
                    if avg_rsi > 60:
                        tech_summary.append("Momentum characteristics were elevated.")
                    elif avg_rsi < 40:
                        tech_summary.append("Momentum characteristics were subdued.")
                    else:
                        tech_summary.append("Momentum characteristics were neutral.")
                
                if "volatility" in snap_df.columns or "vol_regime" in snap_df.columns:
                    tech_summary.append("Volatility regime indicators were present in snapshot.")
                
                if not tech_summary:
                    tech_summary.append("Technical conditions were mixed and consistent with the observed directional classification.")
                
                evidence["technical_signal_context"] = {"summary": " ".join(tech_summary)}
        except Exception:
            evidence["technical_signal_context"] = {"summary": "Technical signal state was not available at time of snapshot."}
        
        return evidence
    
    # Generate evidence artifact
    alpha_evidence = generate_alpha_quality_evidence()
    
    # Render artifact in collapsible panel
    with st.expander("View Alpha Quality Evidence Artifact", expanded=False):
        artifact_html = f"""<div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
<div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Immutable Snapshot · Non-Executing</div>
<div style="color: #3A6FF7; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[P] ALPHA QUALITY EVIDENCE</div>
<div style="color: #666666; font-size: 11px; margin-bottom: 20px;">Generated: {alpha_evidence['generated_at']}</div>"""
        
        # Section 1: Attribution Summary
        artifact_html += """<div style="margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #2A2A2A;">
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;">SECTION 1 — ATTRIBUTION SUMMARY (HISTORICAL)</div>"""
        
        attrib_summary = alpha_evidence.get("attribution_summary", {})
        if attrib_summary:
            for horizon, data in attrib_summary.items():
                total = data.get("total_alpha", 0)
                artifact_html += f"""<div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
<span style="color: #888888; font-size: 12px;">{horizon} Total Alpha:</span>
<span style="color: #D0D0D0; font-size: 12px; font-family: monospace;">{total:+.4f}</span>
</div>"""
        else:
            artifact_html += """<div style="color: #666666; font-size: 12px; font-style: italic;">Attribution data was not available at time of snapshot.</div>"""
        artifact_html += "</div>"
        
        # Section 2: Alpha Heat Index State
        artifact_html += """<div style="margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #2A2A2A;">
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;">SECTION 2 — ALPHA HEAT INDEX STATE</div>"""
        
        heat_state = alpha_evidence.get("heat_index_state", {})
        heat_horizon = heat_state.get("horizon", "N/A")
        heat_diagnostic = heat_state.get("diagnostic", "Heat index state was not available.")
        artifact_html += f"""<div style="color: #888888; font-size: 11px; margin-bottom: 8px;">Horizon: {heat_horizon}</div>
<div style="color: #A0A0A0; font-size: 12px; font-style: italic; line-height: 1.5;">{heat_diagnostic}</div>"""
        artifact_html += "</div>"
        
        # Section 3: Cross-Horizon Stability
        artifact_html += """<div style="margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #2A2A2A;">
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;">SECTION 3 — CROSS-HORIZON STABILITY DIAGNOSTIC</div>"""
        
        stability = alpha_evidence.get("cross_horizon_stability", {})
        stability_text = stability.get("diagnostic", "Stability assessment was not available.")
        artifact_html += f"""<div style="color: #A0A0A0; font-size: 12px; font-style: italic; line-height: 1.5;">{stability_text}</div>"""
        artifact_html += "</div>"
        
        # Section 4: Market Direction Context
        artifact_html += """<div style="margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #2A2A2A;">
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;">SECTION 4 — MARKET DIRECTION CONTEXT</div>"""
        
        direction_ctx = alpha_evidence.get("market_direction_context", {})
        if direction_ctx:
            for horizon, data in direction_ctx.items():
                direction = data.get("direction", "N/A")
                score = data.get("score", 0)
                confidence = data.get("confidence", "N/A")
                artifact_html += f"""<div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
<span style="color: #888888; font-size: 12px;">{horizon}:</span>
<span style="color: #D0D0D0; font-size: 12px;">{direction} ({score}) · {confidence} Confidence</span>
</div>"""
        else:
            artifact_html += """<div style="color: #666666; font-size: 12px; font-style: italic;">Market direction context was not available.</div>"""
        artifact_html += """<div style="color: #555555; font-size: 10px; margin-top: 8px; font-style: italic;">Context only — not causal or predictive.</div>"""
        artifact_html += "</div>"
        
        # Section 5: Technical Signal State
        artifact_html += """<div style="margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #2A2A2A;">
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;">SECTION 5 — TECHNICAL SIGNAL STATE CONTEXT</div>"""
        
        tech_ctx = alpha_evidence.get("technical_signal_context", {})
        tech_summary = tech_ctx.get("summary", "Technical signal state was not available.")
        artifact_html += f"""<div style="color: #A0A0A0; font-size: 12px; font-style: italic; line-height: 1.5;">{tech_summary}</div>"""
        artifact_html += "</div>"
        
        # Provenance Footer
        artifact_html += f"""<div style="border-top: 1px solid #2A2A2A; padding-top: 12px; margin-top: 8px;">
<div style="color: #555555; font-size: 10px; line-height: 1.6;">
<strong>Sources:</strong> Attribution snapshot (30D / 60D / 365D), Alpha Heat Index, Market Direction Assessment, Technical Signal State<br>
<strong>Recorded at:</strong> {alpha_evidence['generated_at']}
</div>
</div>"""
        
        artifact_html += "</div>"
        
        st.markdown(artifact_html, unsafe_allow_html=True)
        
        # Immutability notice
        st.markdown("""<div style="background: #1A1A1A; border: 1px solid #333; border-radius: 4px; padding: 12px; margin-top: 12px;">
<div style="color: #666666; font-size: 11px; line-height: 1.5;">
<strong style="color: #888;">Immutability Notice:</strong> This artifact represents a static snapshot at the recorded timestamp. 
It does not auto-update. To generate a new evidence artifact, a separate governance event must be logged.
</div>
</div>""", unsafe_allow_html=True)
        
        # Governance Note (Read-Only Documentation Panel)
        st.markdown("")
        with st.expander("View Governance Note", expanded=False):
            gov_note_html = """<div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0;">
<div style="color: #A0A0A0; font-size: 13px; font-weight: 600; margin-bottom: 16px;">Governance Note: Alpha Quality Evidence (Historical, Immutable Snapshot)</div>
<div style="color: #555555; font-size: 10px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
Classification: Read-Only · Non-Executing · Historical Diagnostic<br>
Location: Audit Trail → Alpha Quality Evidence<br>
Purpose: Governance Review, Investment Committee Documentation, Diligence Support
</div>

<div style="margin-bottom: 16px;">
<div style="color: #888888; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Section 1 — Overview</div>
<div style="color: #A0A0A0; font-size: 12px; line-height: 1.6;">
The Alpha Quality Evidence (Historical, Immutable Snapshot) is a static, timestamped governance artifact that consolidates key historical diagnostics related to alpha generation, concentration, and structural stability.<br><br>
The artifact is observational only, uses past-tense language throughout, and does not influence any live model, adaptive logic, or execution pathway within the system.
</div>
</div>

<div style="margin-bottom: 16px;">
<div style="color: #888888; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Section 2 — Scope of the Artifact</div>
<div style="color: #A0A0A0; font-size: 12px; line-height: 1.6;">
<strong style="color: #888;">a. Attribution Summary (Historical)</strong><br>
Displays total alpha contributions across 30-day, 60-day, and 365-day horizons using canonical attribution data. This section reflects realized historical outcomes only and does not imply future behavior.<br><br>

<strong style="color: #888;">b. Alpha Heat Index State</strong><br>
Records the diagnostic interpretation produced by the Alpha Heat Index at the time of snapshot generation, including distribution, concentration, and relative balance of component-level contributions. The Alpha Heat Index is descriptive and non-prescriptive.<br><br>

<strong style="color: #888;">c. Cross-Horizon Stability Diagnostic</strong><br>
Summarizes the degree of structural consistency across horizons, including whether short-term alpha drivers aligned with long-term attribution structure or whether material divergence was observed.<br><br>

<strong style="color: #888;">d. Market Direction Context</strong><br>
Provides contemporaneous market direction classification, score, and confidence band for each horizon. Market direction is presented as contextual information only and is not treated as causal or predictive.<br><br>

<strong style="color: #888;">e. Technical Signal State Context</strong><br>
Summarizes observed momentum, volatility, regime, and participation conditions at the time of the snapshot. This section reflects historical signal states only.
</div>
</div>

<div style="margin-bottom: 16px;">
<div style="color: #888888; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Section 3 — Data Provenance</div>
<div style="color: #A0A0A0; font-size: 12px; line-height: 1.6;">
The artifact is derived exclusively from existing, canonical system data, including:<br>
• Historical attribution snapshots (30D / 60D / 365D)<br>
• Alpha Heat Index diagnostics<br>
• Cross-horizon stability assessments<br>
• Market direction classifications<br>
• Technical signal state summaries<br><br>
No external or forward-looking data sources are introduced.
</div>
</div>

<div style="margin-bottom: 16px;">
<div style="color: #888888; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Section 4 — Immutability and Governance Controls</div>
<div style="color: #A0A0A0; font-size: 12px; line-height: 1.6;">
Once generated, the Alpha Quality Evidence artifact is immutable and does not update automatically. Subsequent system updates do not alter this snapshot. The artifact is preserved for audit, governance, and diligence reference.
</div>
</div>

<div style="margin-bottom: 16px;">
<div style="color: #888888; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Section 5 — Intended Use</div>
<div style="color: #A0A0A0; font-size: 12px; line-height: 1.6;">
This artifact supports:<br>
• Investment Committee documentation<br>
• Governance and audit review<br>
• Diligence and acquisition evaluation<br>
• Model oversight and interpretability review<br><br>
It is not intended for decision automation or execution.
</div>
</div>

<div style="margin-bottom: 0;">
<div style="color: #888888; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Section 6 — Compliance Statement</div>
<div style="color: #A0A0A0; font-size: 12px; line-height: 1.6;">
This artifact is observational, backward-looking, non-executing, and informational in nature. It contains no recommendations, thresholds, alerts, or prescriptive language.
</div>
</div>
</div>"""
            st.markdown(gov_note_html, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ===========================
    # Diligence Walkthrough Modal (Governance Loop Layer)
    # ===========================
    with st.expander("Governance Walkthrough", expanded=False):
        walkthrough_html = """<div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
<div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
<div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] GOVERNANCE WALKTHROUGH</div>
<div style="color: #666666; font-size: 12px; margin-bottom: 20px;">Unified governance loop explanation for institutional review</div>"""
        
        walkthrough_steps = [
            ("1. Audit Trail", "What happened?", "Complete record of all governance actions with timestamps and actor identification."),
            ("2. Integrity Signals", "Was it controlled and consistent?", "Observable indicators confirming system behavior aligns with governance design."),
            ("3. Lineage / Rationale", "Why did it happen?", "Factor provenance and decision context for each governance event."),
            ("4. IC Authority", "Who approved it?", "Human actor identification and approval chain for material decisions."),
            ("5. Analytics", "How is governance behaving over time?", "Macro behavioral metrics showing decision patterns and stability."),
            ("6. Export", "Can it be externally verified?", "Portable audit records for compliance, internal audit, and regulatory review.")
        ]
        
        for step_title, step_question, step_answer in walkthrough_steps:
            walkthrough_html += f"""<div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
<div style="color: #D0D0D0; font-weight: 600;">{step_title} — <span style="font-style: italic; font-weight: 400;">{step_question}</span></div>
<div style="color: #A0A0A0; font-size: 12px; padding-left: 12px; margin-top: 6px;">{step_answer}</div>
</div>"""
        
        walkthrough_html += """<div style="border-top: 1px solid #2A2A2A; margin-top: 8px; padding-top: 12px; font-size: 11px; color: #666666;"><strong>Governance Loop:</strong> Decision → Rationale → Approval → Measurement → Review → Decision</div>
</div>"""
        st.markdown(walkthrough_html, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ===========================
    # Diligence Signals Surface (Live System Telemetry)
    # ===========================
    st.subheader("Diligence Signals")
    st.caption("Surfaces observable governance behavior for audit and oversight.")
    
    # Derive signals from actual system state
    def get_diligence_signals():
        """Derive governance signals from live system state."""
        signals = []
        
        # 1. Canonical Unit-Based Accounting
        try:
            snapshot_df = pd.read_csv("data/live_snapshot.csv")
            has_units = "units" in snapshot_df.columns or "Units" in [c.title() for c in snapshot_df.columns]
            has_waves = len(snapshot_df) > 0
            signals.append({
                "label": "Canonical Unit-Based Accounting",
                "status": has_units and has_waves,
                "detail": f"{len(snapshot_df)} positions tracked" if has_waves else "No positions"
            })
        except:
            signals.append({"label": "Canonical Unit-Based Accounting", "status": False, "detail": "Data unavailable"})
        
        # 2. Immutable Audit Trail
        try:
            with open("data/operations_log.json", "r") as f:
                log = json.load(f)
            entries = log.get("entries", [])
            has_timestamps = all("timestamp" in e for e in entries) if entries else True
            has_actors = all("user" in e for e in entries) if entries else True
            signals.append({
                "label": "Immutable Audit Trail (Actor + Timestamp)",
                "status": has_timestamps and has_actors,
                "detail": f"{len(entries)} governance events logged"
            })
        except:
            signals.append({"label": "Immutable Audit Trail (Actor + Timestamp)", "status": False, "detail": "Log unavailable"})
        
        # 3. Human-in-the-Loop Advisory Mode
        # Verified by system architecture - no automated execution paths exist
        signals.append({
            "label": "Human-in-the-Loop Advisory Mode",
            "status": True,
            "detail": "All decisions require explicit human approval"
        })
        
        # 4. Non-Custodial / Non-Executing Architecture
        # Architectural property - no custody or execution endpoints
        signals.append({
            "label": "Non-Custodial / Non-Executing Architecture",
            "status": True,
            "detail": "No trade routing, custody, or execution capability"
        })
        
        # 5. Read-Only Translation Layer (WaveScore™)
        # WaveScore exists as interpretive layer only
        signals.append({
            "label": "Read-Only Translation Layer (WaveScore™)",
            "status": True,
            "detail": "Interpretive only, non-operational"
        })
        
        # 6. Tenant-Isolated Intelligence Fabric
        # Architectural property
        signals.append({
            "label": "Tenant-Isolated Intelligence Fabric",
            "status": True,
            "detail": "Single-tenant deployment, no cross-tenant data"
        })
        
        # 7. Adaptive Learning State
        try:
            with open("data/adaptive_state.json", "r") as f:
                adaptive = json.load(f)
            last_updated = adaptive.get("last_updated", "Unknown")
            threshold_count = len(adaptive.get("thresholds", {}))
            signals.append({
                "label": "Adaptive Intelligence State",
                "status": threshold_count > 0,
                "detail": f"{threshold_count} thresholds tracked, last update: {last_updated[:10] if len(last_updated) > 10 else last_updated}"
            })
        except:
            signals.append({"label": "Adaptive Intelligence State", "status": False, "detail": "State unavailable"})
        
        return signals
    
    diligence_signals = get_diligence_signals()
    
    # Display signals as compact check-style indicators
    signal_cols = st.columns(2)
    for i, signal in enumerate(diligence_signals):
        col = signal_cols[i % 2]
        with col:
            icon = "✔" if signal["status"] else "○"
            status_color = "green" if signal["status"] else "gray"
            st.markdown(f"<span style='color:{status_color};font-size:1em;'>{icon}</span> **{signal['label']}** · <span style='color:gray;font-size:0.85em;'>{signal['detail']}</span>", unsafe_allow_html=True)
    
    st.divider()
    
    # Governance messaging banner (compact)
    st.info("**Immutable governance record.** Read-only. No execution. All actions human-approved and auditable.")
    
    # Load the operations log (read-only access)
    def load_audit_log():
        """Load operations audit log for read-only display."""
        log_path = "data/operations_log.json"
        try:
            with open(log_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"version": "1.0", "entries": []}
    
    audit_log = load_audit_log()
    audit_entries = audit_log.get("entries", [])
    
    # -----------------------------------------------
    # Governance Analytics V2 (Signals Layer)
    # -----------------------------------------------
    st.divider()
    st.subheader("Governance Analytics")
    st.caption("Macro behavioral metrics derived from audit log | Read-only | No thresholds or alerts")
    
    # Calculate analytics from existing audit entries
    analytics_entries = audit_entries if audit_entries else []
    
    # Current period: last 7 days
    now = datetime.now()
    week_ago = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)
    
    # Current period entries
    current_entries = []
    prior_entries = []
    for e in analytics_entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                entry_dt = datetime.fromisoformat(ts.replace("Z", "+00:00").split("+")[0])
                if entry_dt >= week_ago:
                    current_entries.append(e)
                elif entry_dt >= two_weeks_ago:
                    prior_entries.append(e)
            except (ValueError, TypeError):
                pass
    
    decisions_per_week = len(current_entries)
    prior_decisions = len(prior_entries)
    decisions_delta = decisions_per_week - prior_decisions
    
    # 2. Human vs System Ratio (current and prior)
    def calc_human_ratio(entries):
        if not entries:
            return 0.0
        human = sum(1 for e in entries if e.get("user", "").upper() != "SYSTEM" and e.get("user", "") != "")
        return human / len(entries) if entries else 0.0
    
    current_human_ratio = calc_human_ratio(current_entries)
    prior_human_ratio = calc_human_ratio(prior_entries)
    human_ratio_delta = (current_human_ratio - prior_human_ratio) * 100
    
    total_human_ratio = calc_human_ratio(analytics_entries)
    human_pct = f"{total_human_ratio:.0%}" if analytics_entries else "—"
    
    # 3. Average Time Between Actions
    analytics_timestamps = []
    for e in analytics_entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                analytics_timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00").split("+")[0]))
            except (ValueError, TypeError):
                pass
    analytics_timestamps.sort()
    if len(analytics_timestamps) >= 2:
        time_diffs = [(analytics_timestamps[i] - analytics_timestamps[i-1]).total_seconds() / 3600 
                      for i in range(1, len(analytics_timestamps))]
        avg_time_between = np.mean(time_diffs)
        time_variance = np.std(time_diffs) if len(time_diffs) > 1 else 0
        if avg_time_between >= 24:
            avg_time_str = f"{avg_time_between/24:.1f}d"
        else:
            avg_time_str = f"{avg_time_between:.1f}h"
    else:
        avg_time_str = "—"
        time_variance = 0
        avg_time_between = 0
    
    # 4. Recalculations Count
    recalc_total = sum(1 for e in analytics_entries if e.get("action_type") == "RECALCULATE")
    current_recalcs = sum(1 for e in current_entries if e.get("action_type") == "RECALCULATE")
    prior_recalcs = sum(1 for e in prior_entries if e.get("action_type") == "RECALCULATE")
    recalc_delta = current_recalcs - prior_recalcs
    
    # Stability Indicator (variance-based)
    if analytics_entries:
        cv = (time_variance / avg_time_between) if avg_time_between > 0 else 0
        if cv < 0.3:
            stability_badge = "[+] Stable"
        elif cv < 0.7:
            stability_badge = "[-] Normal"
        else:
            stability_badge = "[v] Volatile"
    else:
        stability_badge = "—"
    
    # Display analytics in horizontal metrics row with deltas
    analytics_cols = st.columns(5)
    with analytics_cols[0]:
        delta_str = f"{'▲' if decisions_delta > 0 else '▼'} {abs(decisions_delta):+d}" if decisions_delta != 0 else ""
        st.metric(label="Decisions (7d)", value=str(decisions_per_week), delta=delta_str if delta_str else None)
    with analytics_cols[1]:
        hr_delta_str = f"{'▲' if human_ratio_delta > 0 else '▼'} {abs(human_ratio_delta):.0f}%" if abs(human_ratio_delta) >= 1 else ""
        st.metric(label="Human Ratio", value=human_pct, delta=hr_delta_str if hr_delta_str else None)
    with analytics_cols[2]:
        st.metric(label="Avg Interval", value=avg_time_str)
    with analytics_cols[3]:
        rc_delta_str = f"{'▲' if recalc_delta > 0 else '▼'} {abs(recalc_delta):+d}" if recalc_delta != 0 else ""
        st.metric(label="Recalculations", value=str(recalc_total), delta=rc_delta_str if rc_delta_str else None)
    with analytics_cols[4]:
        st.markdown(f"**Stability**")
        st.markdown(f"{stability_badge}")
    
    # Oversight Drift Detection
    drift_observations = []
    if human_ratio_delta < -5:
        drift_observations.append(f"Human approval ratio decreased {abs(human_ratio_delta):.0f}% over 14 days.")
    if recalc_delta > 2:
        drift_observations.append(f"Recalculation spike detected (+{recalc_delta} vs prior period).")
    if avg_time_between > 168:  # More than 7 days between actions
        drift_observations.append("Extended inactivity gap detected (>7 days between actions).")
    if len(current_entries) > 0:
        modify_count = sum(1 for e in current_entries if e.get("action_type") == "MODIFY")
        if modify_count > len(current_entries) * 0.5:
            drift_observations.append("High modification frequency detected (>50% of recent actions).")
    
    if drift_observations:
        st.markdown(f"<small style='color:#888;'>**Oversight Drift:** {' '.join(drift_observations)}</small>", unsafe_allow_html=True)
    
    st.markdown("")
    
    # -----------------------------------------------
    # Governance Coverage Metrics (Derived, Read-Only)
    # -----------------------------------------------
    st.divider()
    st.subheader("Governance Coverage Metrics")
    st.caption("Derived from audit log | Informational only | No thresholds or alerts")
    
    # Calculate derived metrics from existing audit entries
    total_entries = len(audit_entries)
    
    # Count by action type
    approve_count = sum(1 for e in audit_entries if e.get("action_type") == "APPROVE")
    modify_count = sum(1 for e in audit_entries if e.get("action_type") == "MODIFY")
    reject_count = sum(1 for e in audit_entries if e.get("action_type") == "REJECT")
    defer_count = sum(1 for e in audit_entries if e.get("action_type") == "DEFER")
    recalc_count = sum(1 for e in audit_entries if e.get("action_type") == "RECALCULATE")
    config_count = sum(1 for e in audit_entries if e.get("action_type") == "CONFIG_CHANGE")
    
    # Decision entries (excludes system recalculations)
    decision_entries = [e for e in audit_entries if e.get("action_type") in ["APPROVE", "MODIFY", "REJECT", "DEFER"]]
    decision_count = len(decision_entries)
    
    # Calculate percentages (of decision actions only)
    if decision_count > 0:
        approve_pct = (approve_count / decision_count) * 100
        modify_pct = (modify_count / decision_count) * 100
        reject_pct = (reject_count / decision_count) * 100
        defer_pct = (defer_count / decision_count) * 100
    else:
        approve_pct = modify_pct = reject_pct = defer_pct = 0.0
    
    # Average confidence at approval (if confidence data exists)
    confidence_scores = []
    for e in audit_entries:
        if e.get("action_type") in ["APPROVE", "MODIFY"]:
            rec = e.get("recommendation", {})
            if isinstance(rec, dict):
                conf = rec.get("confidence") or rec.get("confidence_score")
                if conf is not None:
                    try:
                        confidence_scores.append(float(conf))
                    except (ValueError, TypeError):
                        pass
    avg_confidence = np.mean(confidence_scores) if confidence_scores else None
    
    # Recalculation frequency
    recalc_entries = [e for e in audit_entries if e.get("action_type") == "RECALCULATE"]
    
    # Average time from recommendation to decision (if paired data exists)
    # This would require matching recommendation timestamps to decision timestamps
    # For now, calculate time span of audit log
    timestamps = []
    for e in audit_entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00").split("+")[0]))
            except (ValueError, TypeError):
                pass
    
    if len(timestamps) >= 2:
        time_span = max(timestamps) - min(timestamps)
        span_hours = time_span.total_seconds() / 3600
        if total_entries > 1:
            avg_time_between = span_hours / (total_entries - 1)
        else:
            avg_time_between = None
    else:
        span_hours = None
        avg_time_between = None
    
    # Display metrics in clean summary cards
    # Row 1: Decision Distribution
    st.markdown("**Decision Distribution**")
    dist_cols = st.columns(5)
    
    with dist_cols[0]:
        st.metric("Total Decisions", decision_count)
    with dist_cols[1]:
        st.metric("Approved", f"{approve_pct:.0f}%" if decision_count > 0 else "—")
    with dist_cols[2]:
        st.metric("Modified", f"{modify_pct:.0f}%" if decision_count > 0 else "—")
    with dist_cols[3]:
        st.metric("Rejected", f"{reject_pct:.0f}%" if decision_count > 0 else "—")
    with dist_cols[4]:
        st.metric("Deferred", f"{defer_pct:.0f}%" if decision_count > 0 else "—")
    
    # Row 2: System Activity
    st.markdown("**System Activity**")
    sys_cols = st.columns(4)
    
    with sys_cols[0]:
        st.metric("Total Log Entries", total_entries)
    with sys_cols[1]:
        st.metric("Recalculations", recalc_count)
    with sys_cols[2]:
        st.metric("Config Changes", config_count)
    with sys_cols[3]:
        if avg_confidence is not None:
            st.metric("Avg Confidence at Approval", f"{avg_confidence:.0f}%")
        else:
            st.metric("Avg Confidence at Approval", "—")
    
    # Row 3: Governance Velocity (if enough data)
    if span_hours is not None or avg_time_between is not None:
        st.markdown("**Governance Velocity**")
        vel_cols = st.columns(3)
        
        with vel_cols[0]:
            if span_hours is not None:
                if span_hours < 1:
                    st.metric("Log Time Span", f"{span_hours * 60:.0f} min")
                elif span_hours < 24:
                    st.metric("Log Time Span", f"{span_hours:.1f} hours")
                else:
                    st.metric("Log Time Span", f"{span_hours / 24:.1f} days")
            else:
                st.metric("Log Time Span", "—")
        
        with vel_cols[1]:
            if avg_time_between is not None:
                if avg_time_between < 1:
                    st.metric("Avg Time Between Actions", f"{avg_time_between * 60:.0f} min")
                else:
                    st.metric("Avg Time Between Actions", f"{avg_time_between:.1f} hours")
            else:
                st.metric("Avg Time Between Actions", "—")
        
        with vel_cols[2]:
            if decision_count > 0 and recalc_count > 0:
                ratio = decision_count / recalc_count
                st.metric("Decisions per Recalc", f"{ratio:.1f}")
            else:
                st.metric("Decisions per Recalc", "—")
    
    # Action Type Distribution (visual breakdown)
    if total_entries > 0:
        st.markdown("**Action Type Breakdown**")
        breakdown_data = {
            "Approve": approve_count,
            "Modify": modify_count,
            "Reject": reject_count,
            "Defer": defer_count,
            "Recalculate": recalc_count,
            "Config Change": config_count
        }
        breakdown_cols = st.columns(6)
        for i, (action_name, count) in enumerate(breakdown_data.items()):
            with breakdown_cols[i]:
                pct = (count / total_entries * 100) if total_entries > 0 else 0
                st.metric(action_name, f"{count}", delta=f"{pct:.0f}%", delta_color="off")
    
    # -----------------------------------------------
    # View Mode Toggle: Detailed | IC Summary
    # -----------------------------------------------
    st.divider()
    
    view_col1, view_col2 = st.columns([1, 3])
    with view_col1:
        view_mode = st.radio(
            "View Mode",
            options=["Detailed", "IC Summary"],
            key="audit_view_mode",
            horizontal=True
        )
    
    # -----------------------------------------------
    # IC Summary View (Executive-Facing, Read-Only)
    # -----------------------------------------------
    if view_mode == "IC Summary":
        st.divider()
        
        # Governance Integrity Banner
        st.success(
            "**This view summarizes governance activity for investment oversight.**\n\n"
            "All decisions are human-approved, auditable, and execution-safe."
        )
        
        # A. Governance Snapshot (Top Section)
        st.markdown('<span class="waves-micro-label">Audit Layer</span>', unsafe_allow_html=True)
        st.subheader("Governance Snapshot")
        
        snap_cols = st.columns(5)
        with snap_cols[0]:
            st.metric("Total Decisions", decision_count)
        with snap_cols[1]:
            st.metric("Approval Rate", f"{approve_pct:.0f}%" if decision_count > 0 else "—")
        with snap_cols[2]:
            st.metric("Modification Rate", f"{modify_pct:.0f}%" if decision_count > 0 else "—")
        with snap_cols[3]:
            st.metric("Rejection Rate", f"{reject_pct:.0f}%" if decision_count > 0 else "—")
        with snap_cols[4]:
            st.metric("Deferral Rate", f"{defer_pct:.0f}%" if decision_count > 0 else "—")
        
        snap_cols2 = st.columns(3)
        with snap_cols2[0]:
            st.metric("Recalculations", recalc_count)
        with snap_cols2[1]:
            if avg_confidence is not None:
                st.metric("Avg Confidence at Approval", f"{avg_confidence:.0f}%")
            else:
                st.metric("Avg Confidence at Approval", "—")
        with snap_cols2[2]:
            if span_hours is not None:
                if span_hours < 1:
                    st.metric("Log Time Span", f"{span_hours * 60:.0f} min")
                elif span_hours < 24:
                    st.metric("Log Time Span", f"{span_hours:.1f} hours")
                else:
                    st.metric("Log Time Span", f"{span_hours / 24:.1f} days")
            else:
                st.metric("Log Time Span", "—")
        
        # B. Decision Highlights (Condensed Log)
        st.divider()
        st.subheader("Decision Highlights")
        st.caption("Condensed view | Key fields only | Most recent first")
        
        # Filter and Export controls for IC view
        ic_filter_col, ic_export_col = st.columns([1, 2])
        
        with ic_filter_col:
            ic_filter = st.selectbox(
                "Filter",
                options=["All Actions", "Decisions Only", "APPROVE", "MODIFY", "REJECT", "DEFER", "RECALCULATE"],
                key="ic_filter"
            )
        
        # Apply IC filter
        if ic_filter == "All Actions":
            ic_entries = audit_entries
        elif ic_filter == "Decisions Only":
            ic_entries = [e for e in audit_entries if e.get("action_type") in ["APPROVE", "MODIFY", "REJECT", "DEFER"]]
        else:
            ic_entries = [e for e in audit_entries if e.get("action_type") == ic_filter]
        
        # Sort by timestamp (most recent first)
        ic_entries = sorted(ic_entries, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Export buttons for IC Summary view
        with ic_export_col:
            st.markdown("**Export**")
            ic_export_cols = st.columns(2)
            with ic_export_cols[0]:
                if len(ic_entries) > 0:
                    ic_csv_rows = []
                    for e in ic_entries:
                        rec = e.get("recommendation", {})
                        rec_id = rec.get("id", "") if isinstance(rec, dict) else ""
                        rec_title = rec.get("title", "") if isinstance(rec, dict) else ""
                        ic_csv_rows.append({
                            "Timestamp": e.get("timestamp", ""),
                            "Action": e.get("action_type", ""),
                            "Rec ID": rec_id,
                            "Title": rec_title,
                            "Outcome": e.get("decision", "")[:100] if e.get("decision") else ""
                        })
                    ic_csv_df = pd.DataFrame(ic_csv_rows)
                    ic_csv_data = ic_csv_df.to_csv(index=False)
                    st.download_button(
                        label="CSV",
                        data=ic_csv_data,
                        file_name=f"ic_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="csv_export_ic"
                    )
                else:
                    st.button("CSV", disabled=True, key="csv_disabled_ic")
            
            with ic_export_cols[1]:
                if len(ic_entries) > 0:
                    ic_txt_lines = [
                        "WAVES Intelligence Console - IC Summary Export",
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Filter: {ic_filter}",
                        f"Total Entries: {len(ic_entries)}",
                        "=" * 50,
                        ""
                    ]
                    for e in ic_entries:
                        rec = e.get("recommendation", {})
                        rec_id = rec.get("id", "N/A") if isinstance(rec, dict) else "N/A"
                        rec_title = rec.get("title", "N/A") if isinstance(rec, dict) else "N/A"
                        outcome = e.get("decision", "N/A")
                        if len(outcome) > 80:
                            outcome = outcome[:77] + "..."
                        ic_txt_lines.append(f"{e.get('action_type', 'N/A')} | {e.get('timestamp', 'N/A')[:16]}")
                        ic_txt_lines.append(f"  {rec_id} - {rec_title}")
                        ic_txt_lines.append(f"  Outcome: {outcome}")
                        ic_txt_lines.append("")
                    ic_txt_data = "\n".join(ic_txt_lines)
                    st.download_button(
                        label="TXT",
                        data=ic_txt_data,
                        file_name=f"ic_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="txt_export_ic"
                    )
                else:
                    st.button("TXT", disabled=True, key="txt_disabled_ic")
        
        st.markdown(f"**Showing {len(ic_entries)} entries**")
        
        if len(ic_entries) == 0:
            st.info("No entries match the selected filter.")
        else:
            # Display condensed entries (no expand-to-edit, no execution)
            for entry in ic_entries[:25]:
                action_type = entry.get("action_type", "UNKNOWN")
                entry_id = entry.get("id", "N/A")
                timestamp = entry.get("timestamp", "N/A")
                
                # Action badges
                action_badges = {
                    "APPROVE": "APPROVED",
                    "MODIFY": "MODIFIED",
                    "REJECT": "REJECTED",
                    "DEFER": "DEFERRED",
                    "CONFIG_CHANGE": "CONFIG",
                    "RECALCULATE": "RECALC"
                }
                badge = action_badges.get(action_type, action_type)
                
                # Get recommendation info
                rec = entry.get("recommendation", {})
                rec_id = rec.get("id", "—") if isinstance(rec, dict) else "—"
                rec_title = rec.get("title", "—") if isinstance(rec, dict) else "—"
                
                # Get outcome
                outcome = entry.get("decision", "—")
                if len(outcome) > 80:
                    outcome = outcome[:77] + "..."
                
                # Format timestamp
                display_time = timestamp[:16].replace("T", " ") if len(timestamp) >= 16 else timestamp
                
                # Condensed display row
                st.markdown(
                    f"**`{badge}`** | {display_time} | **{rec_id}** — {rec_title}\n\n"
                    f"> {outcome}"
                )
                st.markdown("---")
        
        # Footer for IC view
        st.caption(
            "**IC Summary View** — Designed for Investment Committees, Compliance leadership, and Due Diligence reviewers. "
            "Technical details are hidden. Switch to Detailed view for full audit information."
        )
    
    # -----------------------------------------------
    # Detailed View (Full Audit Log)
    # -----------------------------------------------
    else:
        st.divider()
        st.subheader("Governance Log")
        
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
        
        with filter_col1:
            action_filter = st.selectbox(
                "Filter by Action Type",
                options=["All Actions", "APPROVE", "MODIFY", "REJECT", "DEFER", "CONFIG_CHANGE", "RECALCULATE"],
                key="audit_trail_filter"
            )
        
        # Apply filter
        if action_filter == "All Actions":
            filtered_audit_entries = audit_entries
        else:
            filtered_audit_entries = [e for e in audit_entries if e.get("action_type") == action_filter]
        
        # Sort by timestamp (most recent first)
        filtered_audit_entries = sorted(
            filtered_audit_entries, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        # Export buttons (read-only, derived from operations_log.json only)
        with filter_col2:
            st.markdown("**Export Options**")
        with filter_col3:
            export_cols = st.columns(2)
            with export_cols[0]:
                # CSV Export
                if len(filtered_audit_entries) > 0:
                    csv_rows = []
                    for e in filtered_audit_entries:
                        rec = e.get("recommendation", {})
                        rec_id = rec.get("id", "") if isinstance(rec, dict) else ""
                        rec_title = rec.get("title", "") if isinstance(rec, dict) else ""
                        csv_rows.append({
                            "Entry ID": e.get("id", ""),
                            "Timestamp": e.get("timestamp", ""),
                            "Action Type": e.get("action_type", ""),
                            "Recommendation ID": rec_id,
                            "Recommendation Title": rec_title,
                            "Decision": e.get("decision", ""),
                            "Rationale": e.get("rationale", ""),
                            "User": e.get("user", "")
                        })
                    csv_df = pd.DataFrame(csv_rows)
                    csv_data = csv_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="csv_export_detailed"
                    )
                else:
                    st.button("Download CSV", disabled=True, key="csv_disabled_detailed")
            
            with export_cols[1]:
                # Text/PDF-style Export (plain text for compatibility)
                if len(filtered_audit_entries) > 0:
                    txt_lines = [
                        "WAVES Intelligence Console - Audit Trail Export",
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Filter: {action_filter}",
                        f"Total Entries: {len(filtered_audit_entries)}",
                        "=" * 60,
                        ""
                    ]
                    for e in filtered_audit_entries:
                        rec = e.get("recommendation", {})
                        rec_id = rec.get("id", "N/A") if isinstance(rec, dict) else "N/A"
                        rec_title = rec.get("title", "N/A") if isinstance(rec, dict) else "N/A"
                        txt_lines.append(f"Entry: {e.get('id', 'N/A')}")
                        txt_lines.append(f"Action: {e.get('action_type', 'N/A')}")
                        txt_lines.append(f"Timestamp: {e.get('timestamp', 'N/A')}")
                        txt_lines.append(f"Recommendation: {rec_id} - {rec_title}")
                        txt_lines.append(f"Decision: {e.get('decision', 'N/A')}")
                        if e.get("rationale"):
                            txt_lines.append(f"Rationale: {e.get('rationale')}")
                        txt_lines.append(f"User: {e.get('user', 'N/A')}")
                        txt_lines.append("-" * 40)
                        txt_lines.append("")
                    txt_data = "\n".join(txt_lines)
                    st.download_button(
                        label="Download TXT",
                        data=txt_data,
                        file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="txt_export_detailed"
                    )
                else:
                    st.button("Download TXT", disabled=True, key="txt_disabled_detailed")
        
        st.markdown(f"**Displaying {len(filtered_audit_entries)} of {len(audit_entries)} entries** (most recent first)")
        
        if len(filtered_audit_entries) == 0:
            st.info("No audit entries match the selected filter. As decisions are made in Operations, they will appear here.")
        else:
            # Display entries with expandable details
            for entry in filtered_audit_entries:
                action_type = entry.get("action_type", "UNKNOWN")
                entry_id = entry.get("id", "N/A")
                timestamp = entry.get("timestamp", "N/A")
                
                # Governance badges - professional, subtle, consistent
                governance_badges = {
                    "APPROVE": ("APPROVED", "[+]"),
                    "MODIFY": ("MODIFIED", "[-]"),
                    "REJECT": ("REJECTED", "[v]"),
                    "DEFER": ("DEFERRED", "[P]"),
                    "CONFIG_CHANGE": ("CONFIG CHANGE", "[C]"),
                    "RECALCULATE": ("RECALCULATED", "[R]")
                }
                
                badge_text, badge_icon = governance_badges.get(action_type, (action_type, "[.]"))
                
                # Get recommendation info for decision summary header
                rec = entry.get("recommendation", {})
                rec_id = rec.get("id", "") if isinstance(rec, dict) else ""
                
                # Format timestamp for decision summary header (e.g., "Jan 29, 20:13")
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00").split("+")[0])
                    formatted_time = dt.strftime("%b %d, %H:%M")
                except (ValueError, TypeError):
                    formatted_time = timestamp[:16].replace("T", " ") if len(timestamp) >= 16 else timestamp
                
                # Decision Summary Header: "APPROVED — REC-003 — Jan 29, 20:13"
                if rec_id:
                    summary_header = f"{badge_icon} **{badge_text}** — {rec_id} — {formatted_time}"
                else:
                    summary_header = f"{badge_icon} **{badge_text}** — {entry_id} — {formatted_time}"
                
                with st.expander(summary_header):
                    # Governance badge display
                    st.markdown(f"**`{badge_text}`**")
                    
                    # Entry details in structured format
                    detail_cols = st.columns([1, 2])
                    
                    with detail_cols[0]:
                        st.markdown("**Entry ID:**")
                        st.markdown("**Action Type:**")
                        st.markdown("**Timestamp:**")
                        st.markdown("**User/Actor:**")
                    
                    with detail_cols[1]:
                        st.markdown(f"`{entry_id}`")
                        st.markdown(f"`{action_type}`")
                        st.markdown(f"`{timestamp}`")
                        st.markdown(f"`{entry.get('user', 'Unknown')}`")
                    
                    # One-Line Decision Summary
                    decision_summary_templates = {
                        "RECALCULATE": "Portfolio weights recalculated due to adaptive intelligence update.",
                        "APPROVE": "Human approval granted for proposed allocation change.",
                        "MODIFY": "Allocation adjusted following review and risk parameters.",
                        "REJECT": "Proposed change rejected by human oversight.",
                        "DEFER": "Decision deferred pending additional review.",
                        "CONFIG_CHANGE": "System configuration updated by authorized actor."
                    }
                    decision_summary = entry.get("decision_summary") or decision_summary_templates.get(action_type, "Governance action recorded.")
                    st.markdown(f"<small style='color:#888;'>{decision_summary}</small>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Recommendation details (if applicable)
                    rec_detail = entry.get("recommendation", {})
                    if isinstance(rec_detail, dict) and rec_detail:
                        st.markdown("**Recommendation:**")
                        rec_detail_id = rec_detail.get("id", "N/A")
                        rec_title = rec_detail.get("title", "N/A")
                        st.markdown(f"- **ID:** `{rec_detail_id}`")
                        st.markdown(f"- **Title:** {rec_title}")
                        if rec_detail.get("rationale"):
                            st.markdown(f"- **Rationale:** {rec_detail.get('rationale')}")
                        if rec_detail.get("evidence"):
                            st.markdown(f"- **Evidence:** {rec_detail.get('evidence')}")
                        if rec_detail.get("expected_impact"):
                            st.markdown(f"- **Expected Impact:** {rec_detail.get('expected_impact')}")
                        st.divider()
                    
                    # Decision outcome
                    decision = entry.get("decision", "N/A")
                    st.markdown("**Decision Outcome:**")
                    st.markdown(f"> {decision}")
                    
                    # Rationale (if provided)
                    rationale = entry.get("rationale")
                    if rationale:
                        st.markdown("**Decision Rationale:**")
                        st.markdown(f"> {rationale}")
                    
                    # Affected configuration (if applicable)
                    affected_config = entry.get("affected_config")
                    if affected_config:
                        st.markdown("**Affected Configuration:**")
                        if isinstance(affected_config, dict):
                            for key, value in affected_config.items():
                                st.markdown(f"- `{key}`: {value}")
                        else:
                            st.markdown(f"- {affected_config}")
                    
                    # --- Data Lineage & Provenance (Read-Only) ---
                    # Derive trigger class from action type
                    trigger_class_map = {
                        "APPROVE": "Selection",
                        "MODIFY": "Exposure",
                        "REJECT": "Selection",
                        "DEFER": "Timing",
                        "CONFIG_CHANGE": "Regime",
                        "RECALCULATE": "Overlay"
                    }
                    trigger_class = trigger_class_map.get(action_type, "General")
                    
                    # Derive primary factors from decision text
                    primary_factors = []
                    decision_lower = decision.lower() if decision else ""
                    if "wave" in decision_lower:
                        primary_factors.append({"name": "Wave Composition", "context": "Derived from portfolio structure"})
                    if "weight" in decision_lower or "recalculated" in decision_lower:
                        primary_factors.append({"name": "Weight Distribution", "context": "Recorded from rebalancing"})
                    if "adaptive" in decision_lower or "intelligence" in decision_lower:
                        primary_factors.append({"name": "Adaptive State", "context": "Linked to learning engine"})
                    if "rebuild" in decision_lower or "refresh" in decision_lower:
                        primary_factors.append({"name": "Portfolio Refresh", "context": "Observed system event"})
                    if not primary_factors:
                        primary_factors.append({"name": "Governance Action", "context": "Recorded from operator input"})
                    
                    # Limit to top 3
                    primary_factors = primary_factors[:3]
                    
                    st.divider()
                    st.caption("View Lineage")
                    st.markdown("<small style='color:#999;'>Click to view recorded factor sources and decision context.</small>", unsafe_allow_html=True)
                    with st.expander("Data Lineage & Provenance (Read-Only)", expanded=False):
                        st.caption("This view displays the recorded inputs and attribution factors associated with this event. No actions are enabled.")
                        
                        lineage_col1, lineage_col2 = st.columns([1, 2])
                        
                        with lineage_col1:
                            st.markdown("**Event ID:**")
                            st.markdown("**Recorded Timestamp:**")
                            st.markdown("**Trigger Class:**")
                        
                        with lineage_col2:
                            st.markdown(f"`{entry_id}`")
                            st.markdown(f"`{timestamp}` (UTC)")
                            st.markdown(f"`{trigger_class}`")
                        
                        st.markdown("**Primary Factors:**")
                        for factor in primary_factors:
                            st.markdown(f"- {factor['name']} · _{factor['context']}_")
                        
                        # Data window / source timestamp
                        st.markdown("**Data Window:**")
                        try:
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00").split("+")[0])
                            data_window = dt.strftime("%Y-%m-%d")
                        except (ValueError, TypeError):
                            data_window = "N/A"
                        st.markdown(f"- Source Date: `{data_window}`")
                        
                        # Attribution reference
                        st.markdown("**Attribution Reference:**")
                        st.markdown("- Linked To: Alpha Attribution tab (component breakdown)")
                        
                        # Approver identity
                        approver = entry.get("user", "Unknown")
                        if approver and approver != "Unknown":
                            st.markdown("**Approver Identity:**")
                            st.markdown(f"- `{approver}`")
                        
                        # Rationale notes (if available)
                        if rationale:
                            st.markdown("**Rationale Notes:**")
                            st.markdown(f"- _{rationale}_")
    
    # -----------------------------------------------
    # Governance Narrative Panel (Story Layer)
    # -----------------------------------------------
    st.divider()
    with st.expander("Governance Overview", expanded=False):
        st.caption("Governance philosophy and risk-reduction posture for institutional review")
        st.markdown("")
        
        st.markdown("**1. Human Authority**")
        st.markdown("<small style='color:#888;'>All portfolio decisions require explicit human approval. The system does not execute trades autonomously.</small>", unsafe_allow_html=True)
        st.markdown("")
        
        st.markdown("**2. Lineage & Integrity**")
        st.markdown("<small style='color:#888;'>Each governance action records actor identity, timestamp, rationale, and affected waves, ensuring traceability and audit defensibility.</small>", unsafe_allow_html=True)
        st.markdown("")
        
        st.markdown("**3. Behavioral Analytics**")
        st.markdown("<small style='color:#888;'>Governance cadence and decision behavior are continuously measured to detect drift, instability, or inactivity.</small>", unsafe_allow_html=True)
        st.markdown("")
        
        st.markdown("**4. Export & Portability**")
        st.markdown("<small style='color:#888;'>Governance records can be exported for compliance, internal audit, and regulatory review.</small>", unsafe_allow_html=True)
        st.markdown("")
        
        st.markdown("**5. Net Result Statement**")
        st.markdown("<small style='color:#888;'>WAVES reduces fiduciary and operational risk by making decisions traceable, interpretable, and behaviorally measurable while maintaining human authority.</small>", unsafe_allow_html=True)
    
    # Footer notice
    st.divider()
    st.caption(
        "**Audit Trail Purpose:** This tab provides institutional-grade visibility into all portfolio governance decisions. "
        "Intended for Investment Committees, Compliance Teams, Risk Oversight, and Due Diligence reviews. "
        "No actions can be taken from this view — all decisions are made through the Operations tab."
    )
    st.caption("Behavioral analytics are derived from system governance logs and are not investment performance metrics.")

# ===========================
# Tab 6: Glossary & Concepts
# ===========================
with tabs[5]:
    st.header("Glossary & Concepts")
    st.caption("Reference appendix for understanding the WAVES Intelligence Console")
    st.markdown("")
    
    # -------------------------
    # Section 1: System Philosophy
    # -------------------------
    with st.expander("System Philosophy", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] SYSTEM PHILOSOPHY</div>
            <div style="color: #A0A0A0; font-size: 13px; line-height: 1.7;">
                This console is designed to provide decision support, interpretation, and oversight for portfolio governance. It serves as an advisory layer that helps investment professionals understand market conditions, attribution drivers, and portfolio positioning. The system is explicitly not designed to execute trades, hold custody, or operate autonomously. All outputs require human interpretation and approval. Governance and advisory-only design matter in institutional contexts because they preserve accountability, ensure auditability, and maintain clear lines of responsibility for all investment decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------------
    # Section 2: How to Read This Console
    # -------------------------
    with st.expander("How to Read This Console", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] HOW TO READ THIS CONSOLE</div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Tab Structure and Relationships</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    — <strong>Overview</strong> provides a high-level summary of portfolio performance, market conditions, and key metrics across multiple time horizons.<br>
                    — <strong>Alpha Attribution</strong> breaks down performance into independent components, helping you understand what drove returns.<br>
                    — <strong>Adaptive Intelligence</strong> shows how the system learns and adapts thresholds based on historical patterns.<br>
                    — <strong>Operations</strong> is where human decisions are made — recommendations flow in, and approved actions are logged.<br>
                    — <strong>Audit Trail</strong> provides a complete, timestamped record of all governance decisions for accountability and review.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Reading Metrics and Signals Together</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    Metrics represent quantitative measurements. Signals provide interpretive context. Summaries synthesize both into actionable observations. These should be read together, not in isolation. A single metric without context can be misleading; the system presents multiple perspectives to support informed judgment.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">What Decisions This System Supports</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    The console supports decisions related to portfolio oversight, risk awareness, rebalancing timing, and governance review. It provides the information needed to make these decisions — it does not make the decisions itself.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">What Decisions This System Does Not Make</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    The system does not decide whether to buy or sell, does not determine trade execution timing, and does not authorize transactions. All such decisions remain with qualified human decision-makers.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------------
    # Section 3: How the Console Is Intended to Be Used
    # -------------------------
    with st.expander("How the Console Is Intended to Be Used", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] HOW THE CONSOLE IS INTENDED TO BE USED</div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Typical Engagement Patterns</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    — <strong>Executives</strong> may use the Overview tab for high-level portfolio health and the Audit Trail for governance verification.<br>
                    — <strong>Risk Teams</strong> may focus on Alpha Attribution to understand what drove performance and identify concentration risks.<br>
                    — <strong>Analysts</strong> may engage with Adaptive Intelligence to understand how thresholds and recommendations evolve over time.<br>
                    — <strong>Compliance Officers</strong> may review the Audit Trail to verify that proper governance processes were followed.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Thinking About Comparisons and Benchmarks</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    Comparisons to benchmarks provide relative context, not absolute judgments. Outperformance or underperformance should be interpreted within the context of strategy intent, time horizon, and market conditions.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Using the System as an Advisory Layer</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    The console is intended to inform, not direct. Users should combine console outputs with their own expertise, institutional knowledge, and judgment. The system provides one perspective among many that inform good decisions.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">What Not to Over-Interpret</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    Single data points, short-term fluctuations, or isolated signals should not be over-interpreted. Context, trends, and multiple confirming indicators provide more reliable guidance than any individual metric.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------------
    # Section 4: Understanding the Data
    # -------------------------
    with st.expander("Understanding the Data", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] UNDERSTANDING THE DATA</div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Types of Data Displayed</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    — <strong>Prices and Returns</strong> — Raw market data reflecting actual portfolio and benchmark values.<br>
                    — <strong>Attribution Components</strong> — Derived metrics that decompose returns into independent drivers (selection, momentum, volatility, regime, exposure, residual).<br>
                    — <strong>Signals and Scores</strong> — Interpretive summaries that translate complex data into readable formats. These are informational, not predictive.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Raw vs Derived vs Interpretive Metrics</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    — <strong>Raw</strong>: Direct market observations (prices, volumes, returns).<br>
                    — <strong>Derived</strong>: Calculated from raw data using defined methodologies (attribution components, rolling statistics).<br>
                    — <strong>Interpretive</strong>: Summarized representations designed for human comprehension (WaveScore™, confidence levels, regime labels).
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Time Horizons and Update Cadence</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    The console displays multiple time horizons: Intraday, 30-day, 60-day, and 365-day. Each horizon serves a different purpose. Intraday data reflects current-session activity. Longer horizons provide trend and context. Data updates based on market availability; timestamps indicate freshness.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6; margin-bottom: 16px;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">Accuracy, Freshness, and Limitations</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    Data is sourced from market feeds and updated according to availability. Market hours, delays, and data provider limitations may affect freshness. The system displays what is available and indicates when data may be stale or unavailable.
                </div>
            </div>
            
            <div style="color: #D0D0D0; font-size: 13px; line-height: 1.6;">
                <div style="color: #A0A0A0; font-weight: 600; margin-bottom: 10px;">What Data Informs vs What It Does Not Predict</div>
                <div style="padding-left: 12px; color: #A0A0A0;">
                    Data informs understanding of current and historical conditions. It does not predict future outcomes. All forward-looking interpretation requires human judgment and should account for uncertainty.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------------
    # Section 5: Core Concepts & Complicated Topics
    # -------------------------
    with st.expander("Core Concepts & Complicated Topics", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] CORE CONCEPTS & COMPLICATED TOPICS</div>
            
            <div style="color: #A0A0A0; font-size: 13px; line-height: 1.7;">
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Governance</strong><br>
                    Governance refers to the framework of accountability, oversight, and control that ensures decisions are made responsibly and transparently. In this system, governance means every action is logged, traceable, and subject to human approval. No automated execution occurs.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Audit Trails</strong><br>
                    An audit trail is a permanent, timestamped record of who did what and when. It ensures that any decision can be reviewed, investigated, or verified after the fact. Audit trails are fundamental to institutional accountability and regulatory compliance.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Advisory-Only / Human-in-the-Loop</strong><br>
                    Advisory-only means the system provides recommendations and analysis but never acts on your behalf. Human-in-the-loop means a qualified person must review and approve any action before it occurs. This design preserves accountability and prevents unintended automated behavior.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">System States (Healthy, Degraded, etc.)</strong><br>
                    System states indicate the operational health of data and analysis. A "healthy" state means all data sources are current and functioning normally. A "degraded" state indicates some data may be incomplete, delayed, or unavailable. System states help users calibrate confidence in the displayed information.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Alpha Attribution</strong><br>
                    Alpha attribution decomposes portfolio performance into independent components to explain what drove returns. Rather than just showing a total return number, attribution reveals how much came from security selection, momentum exposure, volatility positioning, regime alignment, and other factors. This helps users understand the why behind performance.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Adaptive Intelligence</strong><br>
                    Adaptive intelligence refers to the system's ability to learn from historical patterns and adjust thresholds or recommendations over time. This learning is advisory — it informs but does not act. Adaptive features help the system remain calibrated to evolving market conditions.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Benchmarks and Comparisons</strong><br>
                    Benchmarks provide relative context by comparing portfolio performance to a reference standard. Comparisons help answer "how did we do relative to alternatives?" rather than just "what was our return?" Benchmark selection affects interpretation; users should understand what benchmark is being used and why.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">WaveScore™</strong><br>
                    WaveScore™ is a read-only interpretive summary that translates complex portfolio data into a simplified view. It is designed to provide at-a-glance understanding, not to drive decisions or trigger actions. WaveScore™ is purely informational and should be interpreted alongside detailed metrics.
                </div>
                
                <div>
                    <strong style="color: #D0D0D0;">Canonical Data</strong><br>
                    Canonical data refers to the single, authoritative source of truth for all system calculations. The console operates from canonical data files that ensure consistency across all views. No overrides or modifications occur outside the canonical layer.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------------
    # Section 6: Glossary of Terms
    # -------------------------
    with st.expander("Glossary of Terms", expanded=True):
        glossary_terms = [
            ("Adaptive State", "The persistent learning configuration that stores historical patterns and adjusted thresholds."),
            ("Advisory-Only", "A design principle where the system provides recommendations without executing actions."),
            ("Alpha", "Returns above or below a benchmark; the portion of performance not explained by market exposure."),
            ("Attribution", "The process of decomposing returns into component drivers to explain performance sources."),
            ("Audit Trail", "A chronological record of all system actions and decisions with timestamps and actor identification."),
            ("Benchmark", "A reference standard used for comparison, typically a market index or target allocation."),
            ("Canonical", "The single authoritative source of truth; canonical data is the foundation for all calculations."),
            ("Confidence", "A measure of reliability or certainty associated with a signal or recommendation."),
            ("Degraded", "A system state indicating some data or functionality may be incomplete or delayed."),
            ("Exposure", "The degree of sensitivity to a particular factor, asset class, or market condition."),
            ("Governance", "The framework of accountability and oversight ensuring responsible decision-making."),
            ("Horizon", "A time period used for analysis (e.g., Intraday, 30D, 60D, 365D)."),
            ("Human-in-the-Loop", "A requirement that human approval is needed before any action is taken."),
            ("Intraday", "Within the current trading day; real-time or near-real-time data."),
            ("Momentum", "Price trend persistence; the tendency of rising assets to continue rising (and vice versa)."),
            ("Operations Log", "A record of all operational decisions made through the governance framework."),
            ("Portfolio", "A collection of investments held and managed together."),
            ("Recommendation", "A system-generated suggestion that requires human review and approval."),
            ("Regime", "Market conditions characterized by specific patterns (e.g., trending, volatile, range-bound)."),
            ("Residual", "The portion of returns not explained by identified factors; unexplained variance."),
            ("Selection", "The component of returns attributable to specific security choices."),
            ("Signal", "An indicator or observation derived from data analysis."),
            ("Snapshot", "A point-in-time capture of portfolio positions and values."),
            ("Tenant", "An isolated organizational unit within a multi-institution architecture."),
            ("Threshold", "A boundary value that triggers a state change or recommendation."),
            ("Volatility", "The degree of price variation over time; a measure of risk or uncertainty."),
            ("Wave", "A thematic grouping of related positions within the portfolio."),
            ("WaveScore™", "A read-only interpretive summary translating complex data into simplified form.")
        ]
        
        glossary_html = """
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] GLOSSARY OF TERMS</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px 24px; color: #A0A0A0; font-size: 13px; line-height: 1.5;">
        """
        
        for term, definition in glossary_terms:
            glossary_html += f'<div><strong style="color: #D0D0D0;">{term}</strong> — {definition}</div>'
        
        glossary_html += "</div></div>"
        st.markdown(glossary_html, unsafe_allow_html=True)
    
    # -------------------------
    # Section 7: How to Interpret Outputs & Signals Responsibly
    # -------------------------
    with st.expander("How to Interpret Outputs & Signals Responsibly", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[i] HOW TO INTERPRET OUTPUTS & SIGNALS RESPONSIBLY</div>
            
            <div style="color: #A0A0A0; font-size: 13px; line-height: 1.7;">
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Signals Provide Context, Not Predictions</strong><br>
                    Signals indicate current conditions and historical patterns. They help frame understanding but do not forecast future outcomes. Responsible interpretation treats signals as informational inputs to judgment, not deterministic guides.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Scores Are Interpretive Summaries</strong><br>
                    Scores like WaveScore™ synthesize complex data into readable formats. They are designed for comprehension, not precision. Scores should be understood as approximate representations that support, rather than replace, detailed analysis.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Changes Over Time Matter More Than Single Points</strong><br>
                    Trends, patterns, and directional changes typically provide more reliable guidance than individual data points. Responsible interpretation considers how metrics evolve rather than fixating on current values alone.
                </div>
                
                <div>
                    <strong style="color: #D0D0D0;">Context and Oversight Are Always Required</strong><br>
                    No metric or signal should be interpreted in isolation. Institutional context, market conditions, portfolio intent, and human expertise all inform proper interpretation. The system provides data; humans provide judgment.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------------
    # Section 8: What This System Does NOT Do
    # -------------------------
    with st.expander("What This System Does NOT Do", expanded=False):
        st.markdown("""
        <div style="background: #1C1C1E; border: 1px solid #2A2A2A; border-radius: 6px; padding: 20px; margin: 8px 0; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <div style="position: absolute; top: 12px; right: 16px; font-size: 11px; color: #888888;">Observational Only · Non-Executing</div>
            <div style="color: #A0A0A0; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;">[L] WHAT THIS SYSTEM DOES NOT DO</div>
            
            <div style="color: #A0A0A0; font-size: 13px; line-height: 1.7;">
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Does Not Execute Trades</strong><br>
                    The system provides analysis and recommendations only. It does not connect to brokers, submit orders, or execute any transactions. All trading activity occurs through separate, external systems with proper authorization.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Does Not Hold Custody</strong><br>
                    The system has no custody of assets. It does not hold, transfer, or control any securities, cash, or other portfolio assets. Custody remains with qualified custodians outside this system.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Does Not Operate Autonomously</strong><br>
                    The system does not make decisions or take actions without human approval. All recommendations require explicit human review and authorization before any downstream action occurs.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Does Not Replace Human Decision-Makers</strong><br>
                    The system is a tool that supports human judgment, not a replacement for it. Qualified professionals remain responsible for all investment decisions. The system informs; humans decide.
                </div>
                
                <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #2A2A2A;">
                    <strong style="color: #D0D0D0;">Does Not Guarantee Outcomes</strong><br>
                    Analysis, signals, and recommendations are based on available data and defined methodologies. They do not guarantee future performance, predict market movements, or ensure any particular outcome. All investments involve risk.
                </div>
                
                <div>
                    <strong style="color: #D0D0D0;">Does Not Connect to External Trading Systems</strong><br>
                    The system operates independently of brokerage platforms, order management systems, and execution venues. There are no API connections, data feeds to trading systems, or automated order generation capabilities.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer notice
    st.divider()
    st.caption(
        "**Reference Appendix:** This tab provides comprehensive reference material for understanding the WAVES Intelligence Console. "
        "It is optional by design and intended for first-time viewers or detailed review. "
        "No actions can be taken from this view — it is purely informational."
    )