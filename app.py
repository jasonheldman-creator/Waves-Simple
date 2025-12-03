import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==================================================
# PAGE SETUP – DARK SINGLE-SCREEN CONSOLE
# ==================================================
st.set_page_config(
    page_title="🌊 Waves Simple Console",
    layout="wide"
)

DARK_BG = "#020617"
CARD_BG = "#0b1120"
ACCENT = "#3b82f6"
TEXT_MAIN = "#e5e7eb"
TEXT_MUTED = "#9ca3af"

custom_css = f"""
<style>
    .stApp {{
        background-color: {DARK_BG};
    }}
    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 1.8rem;
    }}
    h1, h2, h3, h4, h5, h6, label {{
        color: {TEXT_MAIN} !important;
    }}
    .metric-card {{
        background: radial-gradient(circle at top left, #1f2937, #020617);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        border: 1px solid #111827;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    }}
    .metric-label {{
        font-size: 0.72rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: {TEXT_MUTED};
    }}
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 4px;
        color: {TEXT_MAIN};
    }}
    .metric-sub {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-top: 0.1rem;
    }}
    .stDataFrame table {{
        background-color: #020617 !important;
        color: {TEXT_MAIN} !important;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==================================================
# SIDEBAR – MODES & SLIDERS
# ==================================================
st.sidebar.header("⚙️ Console Controls")

smartsafe_mode = st.sidebar.radio(
    "SmartSafe™ mode",
    ["Neutral", "Defensive", "Max Safe"],
    index=0
)

mode_to_cash = {"Neutral": 0, "Defensive": 15, "Max Safe": 30}
default_cash = mode_to_cash[smartsafe_mode]

cash_override = st.sidebar.slider(
    "Cash buffer (SmartSafe™ %)",
    0, 50, default_cash, step=1
)

st.sidebar.caption(
    "SmartSafe™ creates a portfolio-level cash slice and scales down "
    "all positions proportionally. This console is read-only; no live trades."
)

# ==================================================
# HEADER
# ==================================================
st.title("🌊 WAVES SIMPLE CONSOLE")
st.markdown(
    "Upload a **Wave snapshot CSV** to view the portfolio, charts, and "
    "SmartSafe™-adjusted weights on a single institutional-style screen."
)

uploaded_file = st.file_uploader("Upload your Wave snapshot CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# ==================================================
# COLUMN DETECTION
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
    weight = pick("Wave_Wt_Final", "Weight", "Weight_%", "Index_Weight", default=None)
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

ticker_col = colmap["ticker"]
sector_col = colmap["sector"]

# ==================================================
# BASE & ADJUSTED WEIGHTS
# ==================================================
if colmap["weight"] is not None:
    base_w = df_raw[colmap["weight"]].astype(float)
    if base_w.sum() > 1.5:  # looks like percent, normalize
        base_w = base_w / base_w.sum()
else:
    if colmap["dollar"] is not None:
        base_w = df_raw[colmap["dollar"]].astype(float)
        base_w = base_w / base_w.sum()
    else:
        base_w = pd.Series(np.ones(len(df_raw)) / len(df_raw), index=df_raw.index)

df = df_raw.copy()
df["Base_Weight"] = base_w

cash_pct = cash_override / 100.0
adj_w = df["Base_Weight"] * (1 - cash_pct)
if adj_w.sum() > 0:
    adj_w = adj_w / adj_w.sum() * (1 - cash_pct)

df["Adj_Weight"] = adj_w

# ==================================================
# SUMMARY NUMBERS
# ==================================================
total_holdings = len(df)
equity_weight = (1 - cash_pct) * 100

top_idx = df["Base_Weight"].idxmax()
top_ticker = df.loc[top_idx, ticker_col]
top_weight = df.loc[top_idx, "Base_Weight"] * 100

if colmap["dollar"] is not None:
    total_notional = df[colmap["dollar"]].sum()
    largest_pos = df[colmap["dollar"]].max()
else:
    total_notional = None
    largest_pos = None

# ==================================================
# HYPERLINKS TO QUOTES
# ==================================================
df["Quote_Link"] = df[ticker_col].apply(
    lambda t: f"https://finance.yahoo.com/quote/{str(t).strip()}"
)

# ==================================================
# LAYOUT – ALL ON ONE SCREEN
# ==================================================

# --------- ROW 1: METRIC CARDS + TOP-10 BAR CHART ----------
top_left, top_right = st.columns([2, 3])

with top_left:
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)

    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">TOTAL HOLDINGS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_holdings}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-sub">Tickers in this Wave</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">EQUITY ALLOCATION</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{equity_weight:0.1f}%</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-sub">Cash buffer via SmartSafe™: {cash_override:.0f}%</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">TOP HOLDING</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{top_ticker}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub">Base weight: {top_weight:0.2f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">SMARTSAFE™ MODE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{smartsafe_mode}</div>', unsafe_allow_html=True)
        sub = "Preview only – no live orders placed."
        st.markdown(f'<div class="metric-sub">{sub}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with top_right:
    st.subheader("Top 10 holdings (SmartSafe™-adjusted)")
    top10 = df.nlargest(10, "Adj_Weight")
    fig_top10 = px.bar(
        top10,
        x="Adj_Weight",
        y=ticker_col,
        orientation="h",
        text="Adj_Weight",
        height=360,
    )
    fig_top10.update_traces(texttemplate="%{text:.2%}")
    fig_top10.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Weight",
        yaxis_title="",
    )
    st.plotly_chart(fig_top10, use_container_width=True)

st.markdown("---")

# --------- ROW 2: TABLE + CHART PANEL ----------
bottom_left, bottom_right = st.columns([3, 2])

with bottom_left:
    st.subheader("Portfolio table (with live quote links)")

    # Build compact view
    view_cols = [ticker_col]
    for key in ["sector", "price", "dollar", "score"]:
        c = colmap[key]
        if c is not None and c not in view_cols:
            view_cols.append(c)

    df_view = df[view_cols + ["Base_Weight", "Adj_Weight", "Quote_Link"]].copy()
    df_view["Base_Wt_%"] = df_view["Base_Weight"] * 100
    df_view["Adj_Wt_%"] = df_view["Adj_Weight"] * 100
    df_view = df_view.drop(columns=["Base_Weight", "Adj_Weight"])

    st.dataframe(
        df_view.sort_values("Adj_Wt_%", ascending=False),
        use_container_width=True,
        height=430,
        column_config={
            "Quote_Link": st.column_config.LinkColumn(
                "Quote",
                help="Open Yahoo Finance for this ticker",
                display_text="Open"
            ),
            "Base_Wt_%": st.column_config.NumberColumn("Base Wt (%)", format="%.2f"),
            "Adj_Wt_%": st.column_config.NumberColumn("Adj Wt (%)", format="%.2f"),
        },
    )

with bottom_right:
    st.subheader("Exposures & distribution")

    if sector_col is not None:
        sector_df = (
            df.groupby(sector_col)["Adj_Weight"]
            .sum()
            .reset_index()
            .sort_values("Adj_Weight", ascending=True)
        )
        fig_sector = px.bar(
            sector_df,
            x="Adj_Weight",
            y=sector_col,
            orientation="h",
            text="Adj_Weight",
            height=260,
        )
        fig_sector.update_traces(texttemplate="%{text:.2%}")
        fig_sector.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Weight",
            yaxis_title="",
            title="Sector exposure (Adj.)",
            title_x=0.0,
            title_y=0.98,
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.info("No sector column detected in this CSV.")

    fig_hist = px.histogram(
        df,
        x="Adj_Weight",
        nbins=30,
        height=220,
    )
    fig_hist.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Adjusted weight",
        yaxis_title="Count of positions",
        title="Position size distribution",
        title_x=0.0,
        title_y=0.98,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# --------- DIAGNOSTICS (COLLAPSIBLE BUT SAME SCREEN) ----------
with st.expander("🔍 Diagnostics (for you / Franklin)"):
    st.write("**Detected column mapping:**")
    st.json(colmap)
    st.write("Raw columns:", list(df_raw.columns))
    st.write("First 8 rows:")
    st.dataframe(df_raw.head(8), use_container_width=True)