import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Simple Console",
    layout="wide",
)

# --------------------------------------------------------------------
# Dark theme + branding
# --------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Overall app background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0f172a 0%, #020617 55%);
    color: #e5e7eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #1f2937;
}

/* Remove default padding at top */
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 1.5rem;
    max-width: 1450px;
}

/* Headings & text */
h1, h2, h3, h4, h5, h6, label, p, span {
    color: #e5e7eb !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

/* Metric strip */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin: 0.75rem 0 0.25rem 0;
}

.metric-card {
    flex: 1;
    background: radial-gradient(circle at top left, #0f172a, #020617);
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    border: 1px solid #1f2937;
    box-shadow: 0 16px 32px rgba(0,0,0,0.55);
}

.metric-label {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #9ca3af;
}

.metric-value {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 2px;
}

/* Section cards (Bloomberg-style panels) */
.section-card {
    background: rgba(15,23,42,0.92);
    border-radius: 14px;
    padding: 0.75rem 0.9rem 0.8rem 0.9rem;
    border: 1px solid #1f2937;
    box-shadow: 0 18px 40px rgba(0,0,0,0.65);
}

.section-header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:0.35rem;
}

.section-title {
    font-size: 0.92rem;
    font-weight: 600;
}

.section-caption {
    font-size: 0.75rem;
    color: #9ca3af;
}

/* Dataframe tweaks */
.dataframe thead tr th {
    background-color: #020617 !important;
    color: #e5e7eb !important;
}
.dataframe tbody tr:nth-child(even) {
    background-color: rgba(15,23,42,0.6) !important;
}

/* Links inside table */
table.dataframe a {
    color: #38bdf8;
    text-decoration: none;
    font-weight: 500;
}
table.dataframe a:hover {
    text-decoration: underline;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def find_column(df, candidates):
    """Return the first existing column from candidates (case-insensitive), else None."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def clean_columns(df):
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
# Sidebar (controls)
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")

    smart_mode = st.radio(
        "SmartSafe™ mode",
        ["Neutral", "Defensive", "Max Safe"],
        index=0,
        help="Read-only demo: these do not place any trades."
    )

    st.markdown("#### Human overrides")

    equity_tilt = st.slider(
        "Equity tilt (human override, %)",
        min_value=-30, max_value=30, value=0, step=1,
    )

    growth_tilt = st.slider(
        "Growth style tilt (bps)",
        min_value=-300, max_value=300, value=0, step=10,
    )

    value_tilt = st.slider(
        "Value style tilt (bps)",
        min_value=-300, max_value=300, value=0, step=10,
    )

    st.caption("This console is read-only — no live orders are placed.")

# --------------------------------------------------------------------
# Header strip
# --------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:0.78rem; letter-spacing:0.22em; text-transform:uppercase; color:#60a5fa;">
                WAVES INTELLIGENCE™ · LIVE DEMO
            </div>
            <div style="font-size:2.0rem; font-weight:720; margin-top:0.15rem;">
                WAVES SIMPLE CONSOLE
            </div>
            <div style="font-size:0.9rem; color:#9ca3af; margin-top:0.1rem;">
                Bloomberg-style Wave dashboard · one screen · instant CSV analytics.
            </div>
        </div>
        <div style="
            padding:0.45rem 0.9rem;
            border-radius:999px;
            border:1px solid #1f2937;
            background:rgba(15,23,42,0.95);
            font-size:0.78rem;
            color:#9ca3af;">
            AI-managed Wave · CSV-driven · No external data calls
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# File upload
# --------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a Wave snapshot CSV",
    type=["csv"],
    help="Use your Google Sheets / Wave snapshot export.",
)

if not uploaded_file:
    st.info("Upload a CSV file to begin.")
    st.stop()

# --------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------
raw_df = pd.read_csv(uploaded_file)
raw_df = clean_columns(raw_df)

ticker_col = find_column(raw_df, ["Ticker", "Symbol"])
name_col = find_column(raw_df, ["Name", "Security", "Company Name"])
sector_col = find_column(raw_df, ["Sector"])
weight_col = find_column(raw_df, ["Wave_Wt_Final", "Weight", "Portfolio Weight", "Target Weight"])
dollar_col = find_column(raw_df, ["Dollar_Amount", "Position Value", "Market Value", "Value"])

df = raw_df.copy()

# Fallback weights if none supplied
if weight_col is None:
    df["__weight__"] = 1.0 / len(df)
    weight_col = "__weight__"

weights = df[weight_col].astype(float)
total_weight = weights.sum() if weights.sum() > 0 else 1.0
weights_norm = weights / total_weight

# --------------------------------------------------------------------
# Key stats strip (top of screen)
# --------------------------------------------------------------------
n_holdings = len(df)
equity_weight = 1.00  # demo assumption
cash_weight = 0.00
largest_pos = float(weights_norm.max()) if len(weights_norm) > 0 else 0.0

st.markdown('<div class="metric-row">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Total holdings</div>
        <div class="metric-value">{n_holdings:,}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Equity weight</div>
        <div class="metric-value">{format_pct(equity_weight)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Cash weight</div>
        <div class="metric-value">{format_pct(cash_weight)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Largest position</div>
        <div class="metric-value">{format_pct(largest_pos)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# --------------------------------------------------------------------
# Main terminal layout (Preview left, Charts right)
# --------------------------------------------------------------------
left, right = st.columns([1.45, 1.1])

# ---- LEFT: Portfolio preview table ---------------------------------
with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">Portfolio preview</div>
            <div class="section-caption">Snapshot of all holdings with quick links</div>
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
    if weight_col and weight_col not in display_cols:
        display_cols.append(weight_col)
    if dollar_col and dollar_col not in display_cols:
        display_cols.append(dollar_col)

    preview_df = df[display_cols].copy() if display_cols else df.copy()

    # Hyperlink column
    if ticker_col:
        preview_df["Link"] = preview_df[ticker_col].apply(
            lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">Quote</a>'
            if pd.notna(t) else ""
        )
        st.markdown(
            preview_df.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(preview_df, use_container_width=True, height=450)

    st.markdown(
        f"""
        <div style="font-size:0.8rem; color:#9ca3af; margin-top:0.4rem;">
            Wave snapshot: <b>{n_holdings:,}</b> holdings · equity {format_pct(equity_weight)},
            cash {format_pct(cash_weight)}, largest position {format_pct(largest_pos)}.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---- RIGHT: Charts panel -------------------------------------------
with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">Analytics</div>
            <div class="section-caption">Sector allocation & top holdings (weight)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(1)

    # Sector chart
    if sector_col:
        sector_data = (
            pd.DataFrame({
                "Sector": df[sector_col],
                "Weight": weights_norm
            })
            .groupby("Sector", as_index=False)["Weight"]
            .sum()
            .sort_values("Weight", ascending=False)
        )
        fig_sector = px.bar(
            sector_data,
            x="Sector",
            y="Weight",
            title="Sector allocation",
        )
        fig_sector.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="",
            yaxis_title="Weight",
            margin=dict(l=10, r=10, t=40, b=10),
            height=250,
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.info("No 'Sector' column detected — add one to see sector allocation.", icon="ℹ️")

    # Top holdings chart
    if ticker_col:
        top_n = 15
        top_data = (
            pd.DataFrame({
                "Ticker": df[ticker_col],
                "Weight": weights_norm
            })
            .sort_values("Weight", ascending=False)
            .head(top_n)
        )
        fig_top = px.bar(
            top_data,
            x="Ticker",
            y="Weight",
            title=f"Top {top_n} holdings",
        )
        fig_top.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="",
            yaxis_title="Weight",
            margin=dict(l=10, r=10, t=40, b=10),
            height=260,
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No 'Ticker' column detected — add one to see top holdings.", icon="ℹ️")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Bottom strip: SmartSafe + overrides (compact)
# --------------------------------------------------------------------
st.markdown("")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header">
        <div class="section-title">SmartSafe™ & human override status</div>
        <div class="section-caption">Read-only demo of the Waves Intelligence™ control layer</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c, col_d = st.columns([1.2, 1, 1, 1])

with col_a:
    st.markdown("**SmartSafe™ mode**")
    st.markdown(f"`{smart_mode}`")

with col_b:
    st.markdown("**Equity tilt override**")
    st.markdown(f"{equity_tilt:+d}%")

with col_c:
    st.markdown("**Growth style tilt**")
    st.markdown(format_bps(growth_tilt))

with col_d:
    st.markdown("**Value style tilt**")
    st.markdown(format_bps(value_tilt))

st.markdown(
    """
    <div style="font-size:0.78rem; color:#9ca3af; margin-top:0.5rem;">
        In the full WAVES Intelligence™ stack, these controls would adjust Wave-level risk targets,
        apply style tilts inside guardrails, and stream updated targets into the trading engine.
        In this demo, settings are informational only and do not modify the uploaded CSV.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)