import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence – Mini Bloomberg Console",
    layout="wide"
)

# --------------------------------------------------------------------
# DEFAULT CSV PATH
# --------------------------------------------------------------------
# ✅ CHANGE THIS to the actual path on the machine running Streamlit.
# Example if you're running locally on your Mac:
# DEFAULT_CSV_PATH = "/Users/jason/Downloads/SP500_PORTFOLIO_FINAL.csv - Sheet17.csv"
DEFAULT_CSV_PATH = "SP500_PORTFOLIO_FINAL.csv - Sheet17.csv"

# --------------------------------------------------------------------
# WAVES DEFINITIONS (15 Waves)
# --------------------------------------------------------------------
WAVES = {
    "sp500": {
        "label": "S&P 500 Wave (LIVE Demo)",
        "benchmark": "S&P 500 Index",
        "style": "Core · Large Cap",
        "wave_type": "AI-Managed Wave"
    },
    "us_growth": {
        "label": "US Growth Wave",
        "benchmark": "Russell 1000 Growth",
        "style": "Growth · Large Cap",
        "wave_type": "AI-Managed Wave"
    },
    "us_value": {
        "label": "US Value Wave",
        "benchmark": "Russell 1000 Value",
        "style": "Value · Large Cap",
        "wave_type": "AI-Managed Wave"
    },
    "sm_mid_growth": {
        "label": "Small–Mid Cap Growth Wave",
        "benchmark": "SMID Growth Composite",
        "style": "Growth · SMID",
        "wave_type": "AI-Managed Wave"
    },
    "income": {
        "label": "Equity Income Wave",
        "benchmark": "US Dividend Composite",
        "style": "Income · Factor",
        "wave_type": "Income Wave"
    },
    "future_power": {
        "label": "Future Power & Energy Wave",
        "benchmark": "Clean Energy / Infrastructure",
        "style": "Thematic · Energy & Infra",
        "wave_type": "Thematic Wave"
    },
    "tech_leaders": {
        "label": "Tech Leaders Wave",
        "benchmark": "NASDAQ 100",
        "style": "Tech · Mega Cap",
        "wave_type": "AI-Managed Wave"
    },
    "ai_wave": {
        "label": "AI & Automation Wave",
        "benchmark": "AI / Robotics Composite",
        "style": "AI & Robotics",
        "wave_type": "Thematic Wave"
    },
    "quality_core": {
        "label": "Quality Core Wave",
        "benchmark": "Global Quality Composite",
        "style": "Quality · Core",
        "wave_type": "Core Wave"
    },
    "us_core": {
        "label": "US Core Equity Wave",
        "benchmark": "Total US Market",
        "style": "Core · Broad Market",
        "wave_type": "Core Wave"
    },
    "intl_dev": {
        "label": "International Developed Wave",
        "benchmark": "MSCI EAFE",
        "style": "Developed ex-US",
        "wave_type": "Core Wave"
    },
    "emerging": {
        "label": "Emerging Markets Wave",
        "benchmark": "MSCI EM",
        "style": "Emerging Markets",
        "wave_type": "Core Wave"
    },
    "div_growth": {
        "label": "Dividend Growth Wave",
        "benchmark": "US Dividend Growth Index",
        "style": "Dividend Growth",
        "wave_type": "Income Wave"
    },
    "sector_rotation": {
        "label": "Sector Rotation Wave",
        "benchmark": "Sector-Neutral Composite",
        "style": "Sector Rotation",
        "wave_type": "Tactical Wave"
    },
    "crypto_income": {
        "label": "Crypto Income Wave",
        "benchmark": "Crypto Yield Composite",
        "style": "Digital Assets · Income",
        "wave_type": "Crypto Wave"
    },
}

WAVE_KEYS = list(WAVES.keys())

# --------------------------------------------------------------------
# BRANDING CSS – (same as previous version)
# --------------------------------------------------------------------
CSS = """
<style>
/* App background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0b1220 0%, #020617 50%, #000000 100%);
    color: #e5e7eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #020617 65%, #030712 100%);
    border-right: 1px solid #111827;
}

/* Main padding */
.block-container {
    padding-top: 0.35rem;
    padding-bottom: 0.05rem;
    max-width: 1500px;
}

/* Typography */
h1, h2, h3, h4, h5, h6, label, p, span {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    color: #e5e7eb;
}

/* Title gradient */
.wave-title {
    font-size: 1.7rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    background: linear-gradient(90deg, #38bdf8, #60a5fa, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sub badges */
.badge {
    display:inline-flex;
    align-items:center;
    padding: 0.2rem 0.5rem;
    border-radius:999px;
    font-size:0.64rem;
    letter-spacing:0.12em;
    text-transform:uppercase;
    border:1px solid #1f2937;
    background: radial-gradient(circle at top left, rgba(56,189,248,0.25), rgba(15,23,42,0.96));
    color:#e5e7eb;
}
.badge-soft {
    background: rgba(15,23,42,0.88);
}

/* Compact metrics box (top-right) */
.metrics-box {
    background: radial-gradient(circle at top left, #022c22 0%, #020617 55%);
    border-radius: 10px;
    border: 1px solid rgba(45,212,191,0.7);
    box-shadow: 0 0 24px rgba(45,212,191,0.3);
    padding: 0.45rem 0.55rem 0.5rem 0.6rem;
    font-size: 0.7rem;
}
.metrics-header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:0.25rem;
}
.metrics-title {
    font-size:0.68rem;
    text-transform:uppercase;
    letter-spacing:0.16em;
    color:#a5b4fc;
}
.metrics-grid {
    display:grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-column-gap:0.5rem;
    grid-row-gap:0.25rem;
}
.metric-label-mini {
    font-size:0.6rem;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:#9ca3af;
}
.metric-value-mini {
    font-size:0.95rem;
    font-weight:600;
}
.metric-sub-mini {
    font-size:0.64rem;
    color:#6ee7b7;
}

/* Panel cards */
.section-card {
    background: rgba(15,23,42,0.96);
    border-radius:12px;
    padding:0.55rem 0.7rem 0.6rem 0.7rem;
    border:1px solid #111827;
    box-shadow:0 18px 48px rgba(0,0,0,0.9);
}
.section-title {
    font-size:0.84rem;
    font-weight:600;
}
.section-caption {
    font-size:0.7rem;
    color:#9ca3af;
}

/* Top-10 table */
.top10-table-container {
    margin-top:0.15rem;
    border-radius:8px;
    border:1px solid #1f2937;
    overflow:hidden;
}
.top10-table {
    width:100%;
    border-collapse:collapse;
    font-size:0.74rem;
}
.top10-table thead {
    background:#020617;
}
.top10-table th,
.top10-table td {
    padding:0.25rem 0.4rem;
    text-align:left;
    white-space:nowrap;
}
.top10-table th {
    font-size:0.65rem;
    text-transform:uppercase;
    letter-spacing:0.11em;
    color:#9ca3af;
    border-bottom:1px solid #111827;
}
.top10-table tbody tr:nth-child(even) {
    background:rgba(15,23,42,0.9);
}
.top10-table tbody tr:nth-child(odd) {
    background:rgba(15,23,42,0.7);
}
.top10-table tbody tr:hover {
    background:#101827;
}
.top10-ticker {
    font-weight:600;
}
.top10-link {
    color:#38bdf8;
    text-decoration:none;
}
.top10-link:hover {
    text-decoration:underline;
}
.top10-weight {
    font-variant-numeric:tabular-nums;
}
.top10-change-pos {
    color:#22c55e;
}
.top10-change-neg {
    color:#fb7185;
}

/* Row color for up/down */
.row-up td {
    color:#22c55e;
}
.row-down td {
    color:#fb7185;
}

/* Small footer strip */
.footer-note {
    font-size:0.7rem;
    color:#9ca3af;
}

/* Make headers compact */
header[data-testid="stHeader"] {
    height: 0rem;
    min-height: 0rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------
def find_column(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def format_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x * 100:,.1f}%"

def format_bps(x):
    if pd.isna(x):
        return "—"
    return f"{x:,.0f} bps"

def load_snapshot(uploaded_file):
    """
    Priority:
    1) If user uploaded a CSV in the sidebar, use that.
    2) Else try DEFAULT_CSV_PATH on the server/local machine.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "sidebar upload"

    if DEFAULT_CSV_PATH and os.path.exists(DEFAULT_CSV_PATH):
        return pd.read_csv(DEFAULT_CSV_PATH), f"default path: {DEFAULT_CSV_PATH}"

    return None, None

# --------------------------------------------------------------------
# SIDEBAR – WAVE, MODE, INFO + OPTIONAL OVERRIDE UPLOADER
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌊 WAVES Console")
    st.caption("Mini-Bloomberg view for WAVES Intelligence™ · W.A.V.E. Engine")

    wave_key = st.selectbox(
        "Select Wave",
        options=WAVE_KEYS,
        format_func=lambda k: WAVES[k]["label"],
        index=0
    )
    wave_meta = WAVES[wave_key]

    mode = st.radio(
        "Mode",
        options=["Standard", "Private Logic™"],
        index=0,
        help="Standard = disciplined, benchmark-linked. Private Logic™ = higher-octane, proprietary logic."
    )

    st.markdown("---")
    st.markdown("**Wave profile**")
    st.markdown(f"- Benchmark: `{wave_meta['benchmark']}`")
    st.markdown(f"- Style: `{wave_meta['style']}`")
    st.markdown(f"- Type: `{wave_meta['wave_type']}`")

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Override snapshot CSV (optional)",
        type=["csv"],
        help="Leave empty to auto-load the default Sheet 17 CSV. Use this only when testing another file."
    )

    st.caption(
        "Console is read-only – no live trades are placed."
    )

# --------------------------------------------------------------------
# HEADER STRIP (TOP OF SCREEN)
# --------------------------------------------------------------------
col_l, col_r = st.columns([1.8, 1.0])

with col_l:
    st.markdown(
        f"""
        <div>
            <div style="font-size:0.7rem; letter-spacing:0.22em; text-transform:uppercase; color:#9ca3af;">
                WAVES INTELLIGENCE™ · PORTFOLIO WAVE CONSOLE
            </div>
            <div class="wave-title">
                {wave_meta["label"]}
            </div>
            <div style="font-size:0.8rem; color:#9ca3af; margin-top:0.08rem;">
                Benchmark-aware, AI-directed Wave – rendered in a single screen, Bloomberg-style.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_r:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:flex-end; gap:0.35rem; margin-top:0.05rem;">
            <div class="badge-soft">
                Mode · <b>{mode}</b>
            </div>
            <div class="badge-soft">
                Benchmark · <b>{wave_meta["benchmark"]}</b>
            </div>
            <div class="badge">
                LIVE DEMO
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------
# LOAD DATA (NO BIG CENTER UPLOADER ANYMORE)
# --------------------------------------------------------------------
df_raw, data_source = load_snapshot(uploaded_file)

if df_raw is None:
    st.error(
        f"Could not find a CSV snapshot.\n\n"
        f"- Make sure DEFAULT_CSV_PATH points to a real file, or\n"
        f"- Upload a CSV in the sidebar override box."
    )
    st.stop()

df_raw = clean_columns(df_raw)

ticker_col = find_column(df_raw, ["Ticker", "Symbol"])
name_col = find_column(df_raw, ["Name", "Security", "Company Name"])
sector_col = find_column(df_raw, ["Sector"])
weight_col = find_column(df_raw, ["Wave_Wt_Final", "Weight", "Portfolio Weight", "Target Weight"])
dollar_col = find_column(df_raw, ["Dollar_Amount", "Position Value", "Market Value", "Value"])
alpha_bps_col = find_column(df_raw, ["Alpha_bps", "Alpha (bps)", "Alpha_bps_12m"])
change_col = find_column(df_raw, ["Change_1d", "Return_1d", "1D Return", "Today_Return", "Day_Change"])

df = df_raw.copy()

if weight_col is None:
    df["__weight__"] = 1.0 / max(len(df), 1)
    weight_col = "__weight__"

weights = df[weight_col].astype(float)
total_weight = weights.sum() if weights.sum() > 0 else 1.0
weights_norm = weights / total_weight

# TOP 10 HOLDINGS
df["__w_norm__"] = weights_norm
df_sorted = df.sort_values("__w_norm__", ascending=False)
top10 = df_sorted.head(10).copy()

# --------------------------------------------------------------------
# METRICS – COMPACT BOX IN TOP RIGHT (UNDER HEADER)
# --------------------------------------------------------------------
n_holdings = len(df)
equity_weight = 1.0  # placeholder; wire to actual equity/cash later
cash_weight = 0.0
largest_pos = float(weights_norm.max()) if len(weights_norm) > 0 else 0.0

if alpha_bps_col and alpha_bps_col in df.columns:
    alpha_est = float(df[alpha_bps_col].mean())
else:
    alpha_est = np.nan

metrics_html = f"""
<div class="metrics-box">
    <div class="metrics-header">
        <div class="metrics-title">Wave Snapshot</div>
        <div style="font-size:0.62rem; color:#9ca3af;">{data_source or "default"}</div>
    </div>
    <div class="metrics-grid">
        <div>
            <div class="metric-label-mini">Total holdings</div>
            <div class="metric-value-mini">{n_holdings:,}</div>
        </div>
        <div>
            <div class="metric-label-mini">Largest position</div>
            <div class="metric-value-mini">{format_pct(largest_pos)}</div>
        </div>
        <div>
            <div class="metric-label-mini">Equity vs cash</div>
            <div class="metric-value-mini">{format_pct(equity_weight)} / {format_pct(cash_weight)}</div>
            <div class="metric-sub-mini">Wave-level risk budget</div>
        </div>
        <div>
            <div class="metric-label-mini">Alpha capture (est)</div>
            <div class="metric-value-mini">{format_bps(alpha_est) if not np.isnan(alpha_est) else "N/A"}</div>
            <div class="metric-sub-mini">Wave-average vs benchmark</div>
        </div>
    </div>
</div>
"""

# Put metrics right under header, full-width row aligned to right
_, metrics_col = st.columns([1.8, 1.0])
with metrics_col:
    st.markdown(metrics_html, unsafe_allow_html=True)

# --------------------------------------------------------------------
# MAIN TERMINAL LAYOUT – SAME AS BEFORE (top-10 + charts + bottom strip)
# --------------------------------------------------------------------
# … (keep the rest of the code you already have for:
#     - left/right columns
#     - top-10 table with red/green
#     - charts
#     - bottom mode overview)
# --------------------------------------------------------------------