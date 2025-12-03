import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence – Mini Bloomberg Console",
    layout="wide"
)

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
# BRANDING CSS – DARK MINI-BLOOMBERG LOOK (COMPRESSED)
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
    padding-bottom: 0.1rem;
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

/* Metric row */
.metric-row {
    display:flex;
    gap:0.6rem;
    margin-top:0.35rem;
    margin-bottom:0.25rem;
}
.metric-card {
    flex:1;
    background: radial-gradient(circle at top left, #0f172a, #020617);
    border-radius:10px;
    border:1px solid #1f2937;
    padding:0.4rem 0.6rem 0.42rem 0.6rem;
    box-shadow:0 14px 32px rgba(0,0,0,0.8);
}
.metric-label {
    font-size:0.6rem;
    letter-spacing:0.15em;
    text-transform:uppercase;
    color:#9ca3af;
}
.metric-value {
    font-size:1.05rem;
    font-weight:600;
    margin-top:1px;
}
.metric-sub {
    font-size:0.68rem;
    color:#9ca3af;
    margin-top:0.08rem;
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

/* Dataframe styling */
.dataframe thead tr th {
    background-color:#020617 !important;
    color:#e5e7eb !important;
    font-size:0.7rem;
}
.dataframe tbody tr:nth-child(even) {
    background-color:rgba(15,23,42,0.7) !important;
}
.dataframe tbody tr td {
    font-size:0.72rem;
}
table.dataframe a {
    color:#38bdf8;
    text-decoration:none;
    font-weight:500;
}
table.dataframe a:hover {
    text-decoration:underline;
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

# --------------------------------------------------------------------
# SIDEBAR – WAVE, MODE, INFO
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌊 WAVES Console")
    st.caption("Mini-Bloomberg view for WAVES Intelligence™.")

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
    st.caption(
        "Upload any Wave snapshot CSV on the right. "
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
# FILE UPLOAD
# --------------------------------------------------------------------
st.markdown("")
upload_col, _ = st.columns([1.8, 1.0])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a Wave snapshot CSV",
        type=["csv"],
        help="Use your latest Wave snapshot export from Google Sheets or your data engine."
    )

if uploaded_file is None:
    st.info("Upload a CSV file to activate the console.")
    st.stop()

# --------------------------------------------------------------------
# DATA PREP
# --------------------------------------------------------------------
df_raw = pd.read_csv(uploaded_file)
df_raw = clean_columns(df_raw)

ticker_col = find_column(df_raw, ["Ticker", "Symbol"])
name_col = find_column(df_raw, ["Name", "Security", "Company Name"])
sector_col = find_column(df_raw, ["Sector"])
weight_col = find_column(df_raw, ["Wave_Wt_Final", "Weight", "Portfolio Weight", "Target Weight"])
dollar_col = find_column(df_raw, ["Dollar_Amount", "Position Value", "Market Value", "Value"])
alpha_bps_col = find_column(df_raw, ["Alpha_bps", "Alpha (bps)", "Alpha_bps_12m"])

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
# METRIC STRIP (COMPACT)
# --------------------------------------------------------------------
n_holdings = len(df)
equity_weight = 1.0  # placeholder; wire this to equity/cash later
cash_weight = 0.0
largest_pos = float(weights_norm.max()) if len(weights_norm) > 0 else 0.0

if alpha_bps_col and alpha_bps_col in df.columns:
    alpha_est = float(df[alpha_bps_col].mean())
else:
    alpha_est = np.nan

st.markdown('<div class="metric-row">', unsafe_allow_html=True)

# Total holdings
st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">TOTAL HOLDINGS</div>
        <div class="metric-value">{n_holdings:,}</div>
        <div class="metric-sub">Underlying positions in the selected Wave</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Equity / cash
st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">EQUITY VS CASH</div>
        <div class="metric-value">{format_pct(equity_weight)} / {format_pct(cash_weight)}</div>
        <div class="metric-sub">Wave-level risk budget (demo assumes fully invested)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Largest weight
st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">LARGEST POSITION</div>
        <div class="metric-value">{format_pct(largest_pos)}</div>
        <div class="metric-sub">Single-name concentration vs diversified Wave</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Alpha estimate
alpha_text = format_bps(alpha_est) if not np.isnan(alpha_est) else "N/A"
st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">ALPHA CAPTURE (EST)</div>
        <div class="metric-value">{alpha_text}</div>
        <div class="metric-sub">If alpha column exists, this is Wave-average alpha</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# MAIN TERMINAL LAYOUT – ONE SCREEN
# Left: Top 10 Table
# Right: Charts stack
# --------------------------------------------------------------------
left, right = st.columns([1.5, 1.25])

# ---------- LEFT PANEL: TOP 10 HOLDINGS TABLE -----------------------
with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2rem;">
            <div class="section-title">Top 10 holdings</div>
            <div class="section-caption">Ranked by final Wave weight</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_cols = []

    if ticker_col:
        display_cols.append(ticker_col)
    if name_col and name_col not in display_cols:
        display_cols.append(name_col)
    if sector_col and sector_col not in display_cols:
        display_cols.append(sector_col)

    display_cols.append("__w_norm__")
    if dollar_col and dollar_col not in display_cols:
        display_cols.append(dollar_col)

    top_view = top10[display_cols].copy()
    top_view.rename(
        columns={
            "__w_norm__": "Weight",
        },
        inplace=True,
    )

    # Hyperlink ticker column
    if ticker_col:
        link_col = "Link"
        top_view[link_col] = top_view[ticker_col].apply(
            lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">Quote</a>'
            if pd.notna(t) else ""
        )
        # Format weight as %
        if "Weight" in top_view.columns:
            top_view["Weight"] = top_view["Weight"].astype(float).apply(format_pct)

        st.markdown(
            top_view.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    else:
        if "Weight" in top_view.columns:
            top_view["Weight"] = top_view["Weight"].astype(float).apply(format_pct)
        st.dataframe(top_view, use_container_width=True, height=340)

    st.markdown(
        """
        <div class="footer-note" style="margin-top:0.25rem;">
            Tip: click <b>Quote</b> to jump to live market data for any ticker
            while keeping the Wave console pinned.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT PANEL: CHARTS STACK --------------------------------
with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.18rem;">
            <div class="section-title">Wave analytics</div>
            <div class="section-caption">Top-10 profile · Sector mix · Weight distribution</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: Top-10 bar chart
    if ticker_col:
        bar_data = pd.DataFrame({
            "Ticker": top10[ticker_col],
            "Weight": top10["__w_norm__"].astype(float),
        })
        fig_bar = px.bar(
            bar_data,
            x="Weight",
            y="Ticker",
            orientation="h",
            title="Top-10 by Wave weight",
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="Weight",
            yaxis_title="",
            margin=dict(l=10, r=10, t=26, b=10),
            height=190,
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No ticker column found – add a Ticker/Symbol column for holdings charts.", icon="ℹ️")

    # Row 2: Sector donut + distribution mini-chart
    c1, c2 = st.columns([1.2, 1.0])

    # Sector donut
    with c1:
        if sector_col:
            sec_data = (
                df.groupby(df[sector_col])["__w_norm__"]
                .sum()
                .reset_index()
                .rename(columns={sector_col: "Sector", "__w_norm__": "Weight"})
                .sort_values("Weight", ascending=False)
            )

            fig_sect = px.pie(
                sec_data,
                values="Weight",
                names="Sector",
                hole=0.55,
                title="Sector mix (full Wave)",
            )
            fig_sect.update_layout(
                showlegend=True,
                legend_orientation="v",
                legend_y=0.5,
                legend_x=1.05,
                margin=dict(l=0, r=50, t=24, b=0),
                height=205,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig_sect, use_container_width=True)
        else:
            st.info("No 'Sector' column detected – add one to see sector allocation.", icon="ℹ️")

    # Weight distribution mini-chart
    with c2:
        dist_data = df_sorted[["__w_norm__"]].copy()
        dist_data["Rank"] = np.arange(1, len(dist_data) + 1)

        fig_line = px.area(
            dist_data,
            x="Rank",
            y="__w_norm__",
            title="Weight decay curve",
        )
        fig_line.update_traces(mode="lines", line_shape="spline")
        fig_line.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="Holding rank",
            yaxis_title="Weight",
            margin=dict(l=10, r=10, t=24, b=10),
            height=205,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, tickformat=".0%"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# BOTTOM STRIP – MODE EXPLANATION (STILL SAME SCREEN, VERY COMPACT)
# --------------------------------------------------------------------
st.markdown("")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1.15rem;">
        <div style="flex:1;">
            <div class="section-title">Mode overview</div>
            <div class="section-caption">
                How <b>{mode}</b> would steer the {wave_meta["label"]} Wave in production.
            </div>
            <div class="footer-note" style="margin-top:0.25rem;">
                <b>Standard mode</b> keeps the Wave tightly aligned to its benchmark
                (<code>{wave_meta['benchmark']}</code>) with controlled tracking error,
                strict beta discipline, and lower turnover.
                <br/><br/>
                <b>Private Logic™</b> layers in proprietary leadership, regime-switching,
                and SmartSafe™ overlays to push harder for risk-adjusted alpha while still
                staying within institutional guardrails. This demo only changes the
                narrative – live Waves would change exposures and trading plans behind the scenes.
            </div>
        </div>
        <div style="flex:0.9;">
            <div class="section-title">Console status</div>
            <ul class="footer-note">
                <li>Read-only: no real orders are routed from this screen.</li>
                <li>All analytics calculated directly from the uploaded CSV snapshot.</li>
                <li>Every Wave and mode can be exported to a full institutional console.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)