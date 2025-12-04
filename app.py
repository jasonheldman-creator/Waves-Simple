import os
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

# Default snapshot CSV path (adjust name if needed)
DEFAULT_SNAPSHOT_PATH = "SP500_PORTFOLIO_FINAL_Sheet17.csv"

# --------------------------------------------------------------------
# BRANDING CSS – DARK MINI-BLOOMBERG LOOK
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
    padding-top: 0.6rem;
    padding-bottom: 0.4rem;
    max-width: 1500px;
}

/* Typography */
h1, h2, h3, h4, h5, h6, label, p, span {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    color: #e5e7eb;
}

/* Title gradient */
.wave-title {
    font-size: 1.85rem;
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
    padding: 0.25rem 0.55rem;
    border-radius:999px;
    font-size:0.68rem;
    letter-spacing:0.12em;
    text-transform:uppercase;
    border:1px solid #1f2937;
    background: radial-gradient(circle at top left, rgba(56,189,248,0.25), rgba(15,23,42,0.96));
    color:#e5e7eb;
}
.badge-soft {
    display:inline-flex;
    align-items:center;
    padding: 0.18rem 0.45rem;
    border-radius:999px;
    font-size:0.64rem;
    letter-spacing:0.12em;
    text-transform:uppercase;
    border:1px solid #111827;
    background: rgba(15,23,42,0.88);
    color:#e5e7eb;
}

/* Metric row */
.metric-strip {
    position:absolute;
    top:0.2rem;
    right:0rem;
}
.metric-grid {
    display:grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap:0.4rem;
}
.metric-card {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    border-radius:14px;
    border:1px solid #22c55e33;
    padding:0.5rem 0.75rem;
    box-shadow:0 18px 40px rgba(0,0,0,0.8);
}
.metric-label {
    font-size:0.62rem;
    letter-spacing:0.18em;
    text-transform:uppercase;
    color:#9ca3af;
}
.metric-value {
    font-size:1.05rem;
    font-weight:600;
    margin-top:0.08rem;
}
.metric-sub {
    font-size:0.68rem;
    color:#9ca3af;
    margin-top:0.08rem;
}

/* Panel cards */
.section-card {
    background: rgba(15,23,42,0.96);
    border-radius:14px;
    padding:0.7rem 0.8rem 0.75rem 0.85rem;
    border:1px solid #111827;
    box-shadow:0 22px 60px rgba(0,0,0,0.9);
}
.section-title {
    font-size:0.88rem;
    font-weight:600;
}
.section-caption {
    font-size:0.72rem;
    color:#9ca3af;
}

/* Dataframe / table */
table.dataframe {
    width: 100%;
    border-collapse: collapse;
}
table.dataframe thead tr th {
    background-color:#020617 !important;
    color:#e5e7eb !important;
    font-size:0.72rem;
    padding:0.3rem 0.35rem;
    border-bottom:1px solid #111827;
}
table.dataframe tbody tr:nth-child(even) {
    background-color:rgba(15,23,42,0.7) !important;
}
table.dataframe tbody tr td {
    font-size:0.75rem;
    padding:0.26rem 0.35rem;
    border-bottom:1px solid #020617;
}
table.dataframe a {
    color:#38bdf8;
    text-decoration:none;
    font-weight:500;
}
table.dataframe a:hover {
    text-decoration:underline;
}

/* Change cell colors (positive / negative) */
span.change-pos {
    color:#22c55e;
    font-weight:500;
}
span.change-neg {
    color:#ef4444;
    font-weight:500;
}

/* Small footer strip */
.footer-note {
    font-size:0.72rem;
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

def format_change_html(x):
    if pd.isna(x):
        return "—"
    # Try to interpret as decimal if small
    val = float(x)
    if abs(val) < 1.5:
        pct = val * 100.0
    else:
        pct = val
    sign = "+" if pct > 0 else ""
    cls = "change-pos" if pct > 0 else "change-neg" if pct < 0 else ""
    return f'<span class="{cls}">{sign}{pct:.2f}%</span>' if cls else f"{pct:.2f}%"

def google_finance_link(ticker: str) -> str:
    """
    Build a Google Finance URL.
    For demo we assume US listed; Google usually resolves even without exchange suffix.
    """
    if not isinstance(ticker, str) or ticker.strip() == "":
        return ""
    t = ticker.strip().upper()
    # Crypto style like BTC-USD works without suffix
    if "-" in t:
        return f"https://www.google.com/finance/quote/{t}"
    # Default: let Google resolve by ticker
    return f"https://www.google.com/finance/quote/{t}:NASDAQ"

def load_snapshot(uploaded_file):
    """
    Priority:
    1) Sidebar uploaded CSV (override)
    2) Default snapshot CSV on disk
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "sidebar upload"

    if os.path.exists(DEFAULT_SNAPSHOT_PATH):
        return pd.read_csv(DEFAULT_SNAPSHOT_PATH), DEFAULT_SNAPSHOT_PATH

    return None, None

# --------------------------------------------------------------------
# SIDEBAR – WAVE, MODE, INFO
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
    st.markdown("**Override snapshot CSV (optional)**")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["csv"],
        label_visibility="collapsed",
        help="Leave empty to use the default live snapshot file."
    )

    st.caption("Console is read-only – no live trades are placed.")

# --------------------------------------------------------------------
# HEADER STRIP (TOP OF SCREEN)
# --------------------------------------------------------------------
col_l, col_r = st.columns([1.8, 1.0])

with col_l:
    st.markdown(
        f"""
        <div style="position:relative; padding-right:14rem;">
            <div style="font-size:0.78rem; letter-spacing:0.22em; text-transform:uppercase; color:#9ca3af;">
                WAVES INTELLIGENCE™ · PORTFOLIO WAVE CONSOLE
            </div>
            <div class="wave-title">
                {wave_meta["label"]}
            </div>
            <div style="font-size:0.84rem; color:#9ca3af; margin-top:0.12rem;">
                Benchmark-aware, AI-directed Wave – rendered in a single screen, Bloomberg-style.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_r:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:flex-end; gap:0.4rem; margin-top:0.1rem;">
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
# LOAD DATA
# --------------------------------------------------------------------
df_raw, data_source = load_snapshot(uploaded_file)
if df_raw is None:
    st.warning(
        "No snapshot CSV found. Upload a file in the sidebar or place "
        f"`{DEFAULT_SNAPSHOT_PATH}` next to `app.py`.",
        icon="⚠️",
    )
    st.stop()

df_raw = clean_columns(df_raw)

# Column detection
ticker_col = find_column(df_raw, ["Ticker", "Symbol"])
name_col = find_column(df_raw, ["Name", "Security", "Company Name"])
sector_col = find_column(df_raw, ["Sector"])
weight_col = find_column(df_raw, ["Wave_Wt_Final", "Weight", "Portfolio Weight", "Target Weight"])
dollar_col = find_column(df_raw, ["Dollar_Amount", "Position Value", "Market Value", "Value"])
alpha_bps_col = find_column(df_raw, ["Alpha_bps", "Alpha (bps)", "Alpha_bps_12m"])
change_col = find_column(df_raw, ["Change_1d", "1D Change", "Return_1d", "1D_Return", "Change"])

df = df_raw.copy()

# If we have ticker + weight, collapse duplicates by ticker
if ticker_col and weight_col:
    group_keys = [ticker_col]
    if name_col:
        group_keys.append(name_col)
    if sector_col:
        group_keys.append(sector_col)

    agg_map = {weight_col: "sum"}
    if dollar_col:
        agg_map[dollar_col] = "sum"
    if alpha_bps_col:
        agg_map[alpha_bps_col] = "mean"
    if change_col:
        agg_map[change_col] = "mean"

    df = df.groupby(group_keys, as_index=False).agg(agg_map)

# Ensure weight exists
if weight_col is None:
    df["__weight__"] = 1.0 / max(len(df), 1)
    weight_col = "__weight__"

weights = df[weight_col].astype(float)
total_weight = weights.sum() if weights.sum() > 0 else 1.0
weights_norm = weights / total_weight

df["__w_norm__"] = weights_norm
df_sorted = df.sort_values("__w_norm__", ascending=False)
top10 = df_sorted.head(10).copy()

# --------------------------------------------------------------------
# METRIC STRIP (COMPACT, TOP-RIGHT)
# --------------------------------------------------------------------
n_holdings = len(df)
equity_weight = 1.0  # demo assumes fully invested
cash_weight = 0.0
largest_pos = float(weights_norm.max()) if len(weights_norm) > 0 else 0.0
alpha_est = float(df[alpha_bps_col].mean()) if alpha_bps_col and alpha_bps_col in df.columns else np.nan
alpha_text = format_bps(alpha_est) if not np.isnan(alpha_est) else "N/A"

metric_html = f"""
<div class="metric-strip">
  <div class="metric-grid">
    <div class="metric-card">
      <div class="metric-label">TOTAL HOLDINGS</div>
      <div class="metric-value">{n_holdings:,}</div>
      <div class="metric-sub">Positions in the selected Wave</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">EQUITY VS CASH</div>
      <div class="metric-value">{format_pct(equity_weight)} / {format_pct(cash_weight)}</div>
      <div class="metric-sub">Wave-level risk budget (demo)</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">LARGEST POSITION</div>
      <div class="metric-value">{format_pct(largest_pos)}</div>
      <div class="metric-sub">Single-name concentration</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">ALPHA CAPTURE (EST)</div>
      <div class="metric-value">{alpha_text}</div>
      <div class="metric-sub">If alpha column exists in CSV</div>
    </div>
  </div>
</div>
"""
st.markdown(metric_html, unsafe_allow_html=True)

st.markdown("")  # tiny spacer

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
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
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
    if change_col and change_col not in display_cols:
        display_cols.append(change_col)

    top_view = top10[display_cols].copy()
    rename_map = {"__w_norm__": "Weight"}
    if change_col:
        rename_map[change_col] = "1D Change"
    top_view.rename(columns=rename_map, inplace=True)

    # Format columns
    if "Weight" in top_view.columns:
        top_view["Weight"] = top_view["Weight"].astype(float).apply(format_pct)
    if dollar_col and dollar_col in top_view.columns:
        top_view[dollar_col] = top_view[dollar_col].astype(float).map(lambda x: f"${x:,.0f}")
    if "1D Change" in top_view.columns:
        top_view["1D Change"] = top_view["1D Change"].apply(format_change_html)

    # Add Google Finance link column
    if ticker_col:
        link_col = "Link"
        top_view[link_col] = top_view[ticker_col].apply(
            lambda t: f'<a href="{google_finance_link(t)}" target="_blank">Google</a>'
            if pd.notna(t) else ""
        )

    st.markdown(
        top_view.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer-note" style="margin-top:0.35rem;">
            Positive 1D moves render in green; negative in red.
            Click <b>Google</b> for a full Google Finance profile without leaving the console.
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
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
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
            margin=dict(l=10, r=10, t=32, b=10),
            height=230,
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
                margin=dict(l=0, r=60, t=40, b=0),
                height=260,
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
            margin=dict(l=10, r=10, t=40, b=10),
            height=260,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, tickformat=".0%"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# BOTTOM STRIP – MODE EXPLANATION (STILL SAME SCREEN)
# --------------------------------------------------------------------
st.markdown("")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1.5rem;">
        <div style="flex:1;">
            <div class="section-title">Mode overview</div>
            <div class="section-caption">
                How <b>{mode}</b> would steer the {wave_meta["label"]} Wave in production.
            </div>
            <div class="footer-note" style="margin-top:0.4rem;">
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
                <li>All analytics calculated directly from the uploaded or live snapshot CSV.</li>
                <li>Every Wave and mode can be exported to a full institutional console.</li>
                <li>Quote links now open in <b>Google Finance</b> for a cleaner research view.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)