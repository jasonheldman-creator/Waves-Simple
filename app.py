import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# yfinance is optional – app should still run without it
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

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
# 🔁 CHANGE THIS to the actual path on the machine running Streamlit.
# Example on your Mac:
# DEFAULT_CSV_PATH = "/Users/jason/Downloads/SP500_PORTFOLIO_FINAL - Sheet17.csv"
DEFAULT_CSV_PATH = "SP500_PORTFOLIO_FINAL - Sheet17.csv"

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
# HELPERS
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
        return pd.read_csv(DEFAULT_CSV_PATH), f"default: {os.path.basename(DEFAULT_CSV_PATH)}"

    return None, None

def load_index_series(symbol: str, period: str = "6mo"):
    """
    Fetch a simple OHLCV series from yfinance and return a DataFrame
    with Date + Close. If anything fails, return None.
    """
    try:
        data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if data.empty:
            return None
        data = data.reset_index()[["Date", "Close"]]
        return data
    except Exception:
        return None

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

    st.caption("Console is read-only – no live trades are placed.")

# --------------------------------------------------------------------
# HEADER STRIP
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
# LOAD DATA
# --------------------------------------------------------------------
df_raw, data_source = load_snapshot(uploaded_file)

if df_raw is None:
    st.error(
        "Could not find a CSV snapshot.\n\n"
        "• Make sure DEFAULT_CSV_PATH points to your Sheet 17 file, or\n"
        "• Upload a CSV in the sidebar override box."
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
# METRICS – COMPACT BOX IN TOP RIGHT
# --------------------------------------------------------------------
n_holdings = len(df)
equity_weight = 1.0   # placeholder; wire to equity/cash columns later
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
        <div style="font-size:0.62rem; color:#9ca3af;">{data_source or "snapshot"}</div>
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

_, metrics_col = st.columns([1.8, 1.0])
with metrics_col:
    st.markdown(metrics_html, unsafe_allow_html=True)

# --------------------------------------------------------------------
# ROW 1 – MARKET CHARTS: S&P 500 + VIX
# --------------------------------------------------------------------
sp_col, vix_col = st.columns([1.6, 1.0])

with sp_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.15rem;">
            <div class="section-title">S&P 500 Index</div>
            <div class="section-caption">^GSPC · Last 6 months</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sp_df = load_index_series("^GSPC", period="6mo")
    if sp_df is not None:
        fig_sp = px.area(
            sp_df,
            x="Date",
            y="Close",
            title="",
        )
        fig_sp.update_traces(mode="lines", line_shape="spline")
        fig_sp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="",
            yaxis_title="Index level",
            margin=dict(l=10, r=10, t=4, b=10),
            height=200,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True),
        )
        st.plotly_chart(fig_sp, use_container_width=True)
    else:
        st.info("Unable to load ^GSPC data (check yfinance or internet connection).", icon="ℹ️")
    st.markdown("</div>", unsafe_allow_html=True)

with vix_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.15rem;">
            <div class="section-title">VIX Volatility Index</div>
            <div class="section-caption">^VIX · Last 6 months</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    vix_df = load_index_series("^VIX", period="6mo")
    if vix_df is not None:
        fig_vix = px.line(
            vix_df,
            x="Date",
            y="Close",
            title="",
        )
        fig_vix.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="",
            yaxis_title="Index level",
            margin=dict(l=10, r=10, t=4, b=10),
            height=200,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True),
        )
        st.plotly_chart(fig_vix, use_container_width=True)
    else:
        st.info("Unable to load ^VIX data (check yfinance or internet connection).", icon="ℹ️")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# ROW 2 – TOP 10 TABLE (LEFT) + ANALYTICS (RIGHT)
# --------------------------------------------------------------------
left, right = st.columns([1.5, 1.25])

# ---------- LEFT: TOP 10 HOLDINGS TABLE -----------------------------
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
    if change_col and change_col not in display_cols:
        display_cols.append(change_col)

    top_view = top10[display_cols].copy()

    # Build custom HTML table for per-row coloring
    header_cells = []
    if ticker_col:
        header_cells.append("<th>Ticker</th>")
    if name_col:
        header_cells.append("<th>Name</th>")
    if sector_col:
        header_cells.append("<th>Sector</th>")
    header_cells.append("<th>Weight</th>")
    if dollar_col:
        header_cells.append("<th>Value</th>")
    if change_col:
        header_cells.append("<th>1D</th>")

    rows_html = []
    for _, row in top_view.iterrows():
        t = str(row[ticker_col]) if ticker_col and not pd.isna(row[ticker_col]) else ""
        n = str(row[name_col]) if name_col and not pd.isna(row[name_col]) else ""
        s = str(row[sector_col]) if sector_col and not pd.isna(row[sector_col]) else ""
        w = float(row["__w_norm__"])
        w_str = format_pct(w)

        val_str = ""
        if dollar_col and not pd.isna(row[dollar_col]):
            val_str = f"${float(row[dollar_col]):,.0f}"

        chg_str = ""
        chg_class = ""
        row_class = ""
        if change_col and not pd.isna(row[change_col]):
            chg = float(row[change_col])
            chg_str = f"{chg*100:+.2f}%"
            if chg >= 0:
                chg_class = "top10-change-pos"
                row_class = "row-up"
            else:
                chg_class = "top10-change-neg"
                row_class = "row-down"

        if t:
            link_html = f'<a href="https://finance.yahoo.com/quote/{t}' \
                        f'" target="_blank" class="top10-link">Quote</a>'
            ticker_html = f'<span class="top10-ticker">{t}</span> · {link_html}'
        else:
            ticker_html = "—"

        cells = []
        if ticker_col:
            cells.append(f"<td>{ticker_html}</td>")
        if name_col:
            cells.append(f"<td>{n}</td>")
        if sector_col:
            cells.append(f"<td>{s}</td>")
        cells.append(f'<td class="top10-weight">{w_str}</td>')
        if dollar_col:
            cells.append(f"<td>{val_str}</td>")
        if change_col:
            cells.append(f'<td class="{chg_class}">{chg_str}</td>')

        rows_html.append(f'<tr class="{row_class}">{"".join(cells)}</tr>')

    table_html = f"""
    <div class="top10-table-container">
        <table class="top10-table">
            <thead>
                <tr>
                    {''.join(header_cells)}
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    <div class="footer-note" style="margin-top:0.25rem;">
        Positive 1D moves render in <b>green</b>, negatives in <b>red</b>. 
        Click <b>Quote</b> for live market data without leaving the console.
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: ANALYTICS STACK ----------------------------------
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

    # Top-10 bar chart
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
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="Weight",
            yaxis_title="",
            margin=dict(l=10, r=10, t=4, b=10),
            height=170,
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No ticker column found – add a Ticker/Symbol column for holdings charts.", icon="ℹ️")

    # Sector + weight decay row
    c1, c2 = st.columns([1.2, 1.0])

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
            )
            fig_sect.update_layout(
                showlegend=True,
                legend_orientation="v",
                legend_y=0.5,
                legend_x=1.05,
                margin=dict(l=0, r=50, t=4, b=0),
                height=190,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig_sect, use_container_width=True)
        else:
            st.info("No 'Sector' column detected – add one to see sector allocation.", icon="ℹ️")

    with c2:
        dist_data = df_sorted[["__w_norm__"]].copy()
        dist_data["Rank"] = np.arange(1, len(dist_data) + 1)
        fig_line = px.area(
            dist_data,
            x="Rank",
            y="__w_norm__",
        )
        fig_line.update_traces(mode="lines", line_shape="spline")
        fig_line.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="Holding rank",
            yaxis_title="Weight",
            margin=dict(l=10, r=10, t=4, b=10),
            height=190,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, tickformat=".0%"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# BOTTOM STRIP – MODE OVERVIEW
# --------------------------------------------------------------------
st.markdown("")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1.1rem;">
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
                staying within institutional guardrails.
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