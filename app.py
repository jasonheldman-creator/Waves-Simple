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
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Headings & text */
h1, h2, h3, h4, h5, h6, label, p, span {
    color: #e5e7eb !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.metric-card {
    flex: 1;
    background: radial-gradient(circle at top left, #0f172a, #020617);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    border: 1px solid #1f2937;
    box-shadow: 0 18px 40px rgba(0,0,0,0.55);
}

.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #9ca3af;
}

.metric-value {
    font-size: 1.45rem;
    font-weight: 600;
    margin-top: 2px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: #020617;
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    border: 1px solid #1f2937;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #22d3ee, #3b82f6);
    border-color: #38bdf8;
}

/* Dataframe tweaks */
.dataframe thead tr th {
    background-color: #020617 !important;
    color: #e5e7eb !important;
}
.dataframe tbody tr:nth-child(even) {
    background-color: rgba(15,23,42,0.6) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------
# Helper functions
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
# Sidebar controls
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")

    smart_mode = st.radio(
        "SmartSafe™ mode",
        ["Neutral", "Defensive", "Max Safe"],
        index=0,
        help="Read-only demo: these do not place any trades yet.",
    )

    st.markdown("#### Human overrides")

    equity_tilt = st.slider(
        "Equity tilt (human override, %)",
        min_value=-30,
        max_value=30,
        value=0,
        step=1,
        help="Shifts the Wave toward or away from equities."
    )

    growth_tilt = st.slider(
        "Growth style tilt (bps)",
        min_value=-300,
        max_value=300,
        value=0,
        step=10,
    )

    value_tilt = st.slider(
        "Value style tilt (bps)",
        min_value=-300,
        max_value=300,
        value=0,
        step=10,
    )

    st.caption("This console is read-only — no live orders are placed.")

# --------------------------------------------------------------------
# Header
# --------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
        <div>
            <div style="font-size:0.78rem; letter-spacing:0.22em; text-transform:uppercase; color:#60a5fa;">
                WAVES INTELLIGENCE™
            </div>
            <div style="font-size:2.1rem; font-weight:720; margin-top:0.25rem;">
                WAVES SIMPLE CONSOLE
            </div>
            <div style="font-size:0.92rem; color:#9ca3af; margin-top:0.2rem;">
                Upload a Wave snapshot CSV to view live portfolio analytics, charts, and human overrides.
            </div>
        </div>
        <div style="
            padding:0.5rem 0.9rem;
            border-radius:999px;
            border:1px solid #1f2937;
            background:rgba(15,23,42,0.9);
            font-size:0.8rem;
            color:#9ca3af;">
            Demo mode · AI-managed Wave · No external data calls
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# --------------------------------------------------------------------
# File upload
# --------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a Wave snapshot CSV",
    type=["csv"],
    help="Use your Google Sheets / portfolio export for the selected Wave.",
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
# Key stats
# --------------------------------------------------------------------
n_holdings = len(df)
equity_weight = 1.00  # demo – treat all as equity
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

# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------
tab_preview, tab_charts, tab_overrides = st.tabs(
    ["Preview & stats", "Charts", "Overrides & targets"]
)

# ---------- Preview & stats ---------------------------------------------------
with tab_preview:
    st.subheader("Portfolio preview")

    # Build a neat table with optional hyperlink column
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

    if ticker_col:
        preview_df["Link"] = preview_df[ticker_col].apply(
            lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">Open</a>'
            if pd.notna(t) else ""
        )
        st.markdown(
            preview_df.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(preview_df, use_container_width=True)

    # Simple summary text
    st.markdown(
        f"""
        **Wave snapshot:** {n_holdings:,} holdings ·
        equity {format_pct(equity_weight)}, cash {format_pct(cash_weight)},  
        largest single position {format_pct(largest_pos)}.
        """
    )

# ---------- Charts ------------------------------------------------------------
with tab_charts:
    st.subheader("Top holdings & sector charts")

    col1, col2 = st.columns(2)

    # Sector chart
    with col1:
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
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("No sector column detected – add a 'Sector' column to see this chart.")

    # Top holdings chart
    with col2:
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
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No ticker column detected – add a 'Ticker' column to see this chart.")

# ---------- Overrides & targets ----------------------------------------------
with tab_overrides:
    st.subheader("SmartSafe™ & human overrides (demo)")

    st.markdown(
        f"""
        **Current SmartSafe™ mode:** `{smart_mode}`  
        **Equity tilt override:** `{equity_tilt:+d}%`  
        **Growth tilt override:** `{format_bps(growth_tilt)}`  
        **Value tilt override:** `{format_bps(value_tilt)}`
        """
    )

    st.markdown(
        """
        In the full WAVES Intelligence™ console, these settings would:
        - Adjust target risk and equity exposure per Wave  
        - Tilt the portfolio toward growth or value leaders within defined guardrails  
        - Route final targets into the trading engine and compliance layer  

        In this demo, overrides are **view-only** – they do not modify the uploaded CSV.
        """
    )