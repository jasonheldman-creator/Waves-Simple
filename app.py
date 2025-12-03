import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==================================================
# PAGE SETUP – DARK MODE
# ==================================================
st.set_page_config(
    page_title="🌊 Waves Simple Console",
    layout="wide"
)

DARK_BG = "#0d1117"
CARD_BG = "#111827"
ACCENT = "#3b82f6"
TEXT_MAIN = "#e5e7eb"
TEXT_MUTED = "#9ca3af"

custom_css = f"""
<style>
    body {{
        background-color: {DARK_BG};
        color: {TEXT_MAIN};
    }}
    .stApp {{
        background-color: {DARK_BG};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }}
    h1, h2, h3, h4, h5, h6, label {{
        color: {TEXT_MAIN} !important;
    }}
    .metric-card {{
        background: radial-gradient(circle at top left, #1f2937, #020617);
        border-radius: 14px;
        padding: 1rem 1.3rem;
        border: 1px solid #1f2937;
        box-shadow: 0 24px 60px rgba(0,0,0,0.6);
    }}
    .metric-label {{
        font-size: 0.78rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: {TEXT_MUTED};
    }}
    .metric-value {{
        font-size: 1.7rem;
        font-weight: 600;
        margin-top: 4px;
        color: {TEXT_MAIN};
    }}
    .metric-sub {{
        font-size: 0.8rem;
        color: {TEXT_MUTED};
        margin-top: 0.15rem;
    }}
    .stDataFrame table {{
        background-color: #020617 !important;
        color: {TEXT_MAIN} !important;
        border-radius: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 1rem;
        color: {TEXT_MAIN};
        padding: 0.7rem 1.2rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #111827 !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==================================================
# HEADER & SIDEBAR
# ==================================================
st.title("🌊 WAVES SIMPLE CONSOLE")
st.markdown(
    "Upload a **Wave snapshot CSV** to explore the portfolio, "
    "visualize exposures, and preview SmartSafe™ & human overrides."
)

st.sidebar.header("⚙️ Console Controls")

smartsafe_mode = st.sidebar.radio(
    "SmartSafe™ mode",
    ["Neutral", "Defensive", "Max Safe"],
    index=0
)

# Recommended cash based on mode
mode_to_cash = {
    "Neutral": 0,
    "Defensive": 15,
    "Max Safe": 30,
}
recommended_cash = mode_to_cash[smartsafe_mode]

cash_override = st.sidebar.slider(
    "Cash buffer (SmartSafe™ %, removes from equities)",
    0, 50, recommended_cash, step=1
)

st.sidebar.caption(
    "SmartSafe™ creates a portfolio-level cash buffer and "
    "scales down all holdings proportionally. This console is read-only; "
    "no live trades are placed."
)

equity_tilt = st.sidebar.slider("Equity tilt (bps – informational only)", -300, 300, 0, step=25)
growth_tilt = st.sidebar.slider("Growth vs Value tilt (bps – informational only)", -300, 300, 0, step=25)

# ==================================================
# FILE UPLOAD
# ==================================================
uploaded_file = st.file_uploader("Upload your Wave snapshot CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# ==================================================
# COLUMN DETECTION HELPERS
# ==================================================
def detect_columns(df: pd.DataFrame):
    cols = df.columns

    def pick(*names, default=None):
        for n in names:
            if n in cols:
                return n
        return default

    ticker = pick("Ticker", "Symbol", default=cols[0])
    sector = pick("Sector", "GICS_Sector", default=None)
    weight = pick("Wave_Wt_Final", "Weight", "Weight_%", default=None)
    dollar = pick("Dollar_Amount", "Position_Dollar", "Dollar", default=None)
    price = pick("Price", default=None)
    score = pick("Momentum_Score", "Score", default=None)

    return {
        "ticker": ticker,
        "sector": sector,
        "weight": weight,
        "dollar": dollar,
        "price": price,
        "score": score,
    }

colmap = detect_columns(df_raw)

# Base weight series
if colmap["weight"] is not None:
    base_w = df_raw[colmap["weight"]].astype(float)
    # If not roughly in [0,1], normalise
    if base_w.sum() > 1.5:
        base_w = base_w / base_w.sum()
else:
    if colmap["dollar"] is not None:
        base_w = df_raw[colmap["dollar"]].astype(float)
        base_w = base_w / base_w.sum()
    else:
        # Equal weight fallback
        base_w = pd.Series(np.ones(len(df_raw)) / len(df_raw), index=df_raw.index)

df = df_raw.copy()
df["__Base_Weight__"] = base_w

# Apply SmartSafe cash override: shrink remaining weights
cash_pct = cash_override / 100.0
scaled_w = df["__Base_Weight__"] * (1 - cash_pct)
if scaled_w.sum() > 0:
    scaled_w = scaled_w / scaled_w.sum() * (1 - cash_pct)  # keep explicit cash slice
df["__Adj_Weight__"] = scaled_w

# ==================================================
# SUMMARY METRICS
# ==================================================
total_holdings = len(df)
total_equity_weight = (1 - cash_pct) * 100
top_idx = df["__Base_Weight__"].idxmax()
top_name = df.loc[top_idx, colmap["ticker"]]
top_weight = df.loc[top_idx, "__Base_Weight__"] * 100

if colmap["dollar"] is not None:
    total_dollar = df[colmap["dollar"]].sum()
    largest_pos = df[colmap["dollar"]].max()
else:
    total_dollar = None
    largest_pos = None

# ==================================================
# LAYOUT – SUMMARY ROW
# ==================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">TOTAL HOLDINGS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_holdings}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-sub">Unique tickers in this Wave</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">EQUITY ALLOCATION</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_equity_weight:0.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-sub">Cash buffer: {cash_override:.0f}% via SmartSafe™</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">TOP HOLDING</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{top_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-sub">Base weight: {top_weight:0.2f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">STYLE SIGNALS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{equity_tilt:+.0f} / {growth_tilt:+.0f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-sub">Equity / Growth tilts (bps, informational)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================================================
# TABS
# ==================================================
tab_overview, tab_charts, tab_overrides, tab_diag = st.tabs(
    ["📊 Overview", "📈 Charts", "🛡 SmartSafe & Overrides", "🔍 Diagnostics"]
)

# ==================================================
# OVERVIEW TAB
# ==================================================
with tab_overview:
    st.subheader("Portfolio preview")

    # Show main table with base & adjusted weights
    display_cols = [colmap["ticker"]]
    for key in ["sector", "price", "dollar", "score"]:
        if colmap[key] is not None:
            display_cols.append(colmap[key])

    display_cols = list(dict.fromkeys(display_cols))  # dedupe
    df_view = df[display_cols].copy()
    df_view["Base_Wt_%"] = df["__Base_Weight__"] * 100
    df_view["Adj_Wt_%"] = df["__Adj_Weight__"] * 100

    st.dataframe(
        df_view.sort_values("Base_Wt_%", ascending=False),
        use_container_width=True,
        height=420
    )

    st.subheader("Key stats")
    k1, k2, k3 = st.columns(3)

    if total_dollar is not None:
        k1.metric("Total portfolio notional", f"${total_dollar:,.0f}")
        k2.metric("Largest position (notional)", f"${largest_pos:,.0f}")
    else:
        k1.metric("Total portfolio weight", "100.0%")
        k2.metric("Largest position weight", f"{top_weight:0.2f}%")

    if colmap["sector"] is not None:
        sector_counts = df[colmap["sector"]].nunique()
        k3.metric("Distinct sectors", sector_counts)
    else:
        k3.metric("Distinct sectors", "N/A")

# ==================================================
# CHARTS TAB
# ==================================================
with tab_charts:
    st.subheader("Portfolio charts")

    chart_choice = st.selectbox(
        "Choose a chart",
        ["Top 10 Holdings (bar)", "Weight Distribution (pie)", "Sector Exposure", "Position Size Distribution"]
    )

    ticker_col = colmap["ticker"]
    sector_col = colmap["sector"]

    # Top 10 holdings by adjusted weight
    if chart_choice == "Top 10 Holdings (bar)":
        top10 = df.nlargest(10, "__Adj_Weight__")
        fig = px.bar(
            top10,
            x="__Adj_Weight__",
            y=ticker_col,
            orientation="h",
            text="__Adj_Weight__",
            title="Top 10 Holdings – Adjusted for SmartSafe™"
        )
        fig.update_traces(texttemplate="%{text:.2%}")
        fig.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Pie chart – adjusted weights
    if chart_choice == "Weight Distribution (pie)":
        fig = px.pie(
            df,
            names=ticker_col,
            values="__Adj_Weight__",
            title="Weight Distribution (post-SmartSafe™)"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Sector exposure
    if chart_choice == "Sector Exposure":
        if sector_col is None:
            st.error("No Sector column found in the CSV.")
        else:
            sector_df = (
                df.groupby(sector_col)["__Adj_Weight__"]
                .sum()
                .reset_index()
                .sort_values("__Adj_Weight__", ascending=True)
            )
            fig = px.bar(
                sector_df,
                x="__Adj_Weight__",
                y=sector_col,
                orientation="h",
                text="__Adj_Weight__",
                title="Sector Exposure – Adjusted Weights"
            )
            fig.update_traces(texttemplate="%{text:.2%}")
            fig.update_layout(template="plotly_dark", height=560, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Position size distribution
    if chart_choice == "Position Size Distribution":
        fig = px.histogram(
            df,
            x="__Adj_Weight__",
            nbins=30,
            title="Distribution of Position Sizes (Adjusted Weights)"
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Weight",
            yaxis_title="Count of positions",
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# SMARTSAFE & OVERRIDES TAB
# ==================================================
with tab_overrides:
    st.subheader("SmartSafe™ & human override preview")

    st.markdown(
        f"""
        **Mode:** `{smartsafe_mode}`  
        **Cash buffer in use:** **{cash_override:.0f}%**  
        Equity holdings are scaled down proportionally; this is a **preview only**,  
        no live orders are placed.
        """
    )

    # Show top 15 with base vs adjusted comparison
    comp = df[[ticker_col]].copy()
    comp["Base_Wt_%"] = df["__Base_Weight__"] * 100
    comp["Adj_Wt_%"] = df["__Adj_Weight__"] * 100
    comp = comp.sort_values("Base_Wt_%", ascending=False).head(15)

    st.markdown("#### Top 15 holdings – base vs SmartSafe™ adjusted")
    st.dataframe(comp, use_container_width=True, height=420)

    fig = px.bar(
        comp.melt(id_vars=[ticker_col], value_vars=["Base_Wt_%", "Adj_Wt_%"],
                  var_name="Type", value_name="Weight"),
        x="Weight",
        y=ticker_col,
        color="Type",
        barmode="group",
        orientation="h",
        title="Base vs Adjusted Weights (Top 15)"
    )
    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "➡️ In a later version, this tab can be wired to **order simulation**, "
        "**risk bands**, and **WaveScore™** thresholds. For now, it serves as a "
        "transparent, auditable override layer for institutionals."
    )

# ==================================================
# DIAGNOSTICS TAB
# ==================================================
with tab_diag:
    st.subheader("Diagnostics & column mapping")

    st.write("**Detected columns** (based on your CSV header names):")
    st.json(colmap)

    st.markdown("#### Raw columns")
    st.write(list(df_raw.columns))

    st.markdown("#### First 10 rows (raw)")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.caption(
        "Use this tab if something isn’t rendering correctly. "
        "Franklin (or any acquirer) can see exactly how the console is reading their CSV."
    )