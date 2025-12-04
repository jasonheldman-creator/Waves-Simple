import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# DARK MODE + GLOBAL STYLE
# -----------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ – Portfolio Wave Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body, .stApp {
    background-color: #0D1117 !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', sans-serif;
}
.sidebar .sidebar-content {
    background-color: #0D1117 !important;
}
h1, h2, h3, h4, h5 {
    color: #58A6FF !important;
}
.dataframe th {
    color: #58A6FF !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 10 OFFICIAL WAVES (LIVE DEMO)
# -----------------------------
WAVE_CONFIG = {
    "SPX": {
        "name": "S&P 500 Core Equity Wave",
        "benchmark": "SPY",
        "style": "Core – Large Cap",
        "csv": "https://docs.google.com/spreadsheets/d/e/2PACX-1vT7VpPdWSUSyZP9CVXZwTgqx7a7mMD2aQMRqSESqZgiagh8wSeEm3RAWHvLlWmJtLqYrqj7UVjQIpq9/pub?gid=711820877&single=true&output=csv"
    },
    "QQQ": {
        "name": "US Growth & Innovation Wave",
        "benchmark": "QQQ",
        "style": "Core – Large Cap",
        "csv": None  # Waiting for live CSV
    },
    "SMGX": {
        "name": "Small Cap Growth Wave",
        "benchmark": "IJR",
        "style": "Small Cap",
        "csv": None
    },
    "MID": {
        "name": "Mid Cap Wave",
        "benchmark": "MDY",
        "style": "Mid Cap",
        "csv": None
    },
    "VAL": {
        "name": "Value Wave",
        "benchmark": "VTV",
        "style": "Value",
        "csv": None
    },
    "DIV": {
        "name": "Equity Income Wave",
        "benchmark": "SCHD",
        "style": "Dividend / Income",
        "csv": None
    },
    "RWA": {
        "name": "RWA Income Wave",
        "benchmark": "RWR",
        "style": "Real-World Assets",
        "csv": None
    },
    "FIN": {
        "name": "Financials Leadership Wave",
        "benchmark": "XLF",
        "style": "Sector – Financials",
        "csv": None
    },
    "ENE": {
        "name": "Future Energy Leadership Wave",
        "benchmark": "XLE",
        "style": "Sector – Energy",
        "csv": None
    },
    "AI": {
        "name": "AI Leadership Wave",
        "benchmark": "QQQ",
        "style": "Innovation / AI",
        "csv": None
    }
}

# -----------------------------
# LOAD HOLDINGS
# -----------------------------
def load_holdings(csv_url):
    if csv_url is None:
        return None
    
    try:
        df = pd.read_csv(csv_url)
    except Exception:
        return None

    df.columns = [c.strip().lower() for c in df.columns]

    # MUST have ticker + weight
    required = ["ticker", "weight"]
    for req in required:
        if req not in df.columns:
            return None

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["ticker", "weight"])
    df = df.sort_values(by="weight", ascending=False)

    return df

# -----------------------------
# SIDEBAR – SELECTED WAVE
# -----------------------------
selected = st.sidebar.selectbox(
    "Select Wave",
    list(WAVE_CONFIG.keys()),
    format_func=lambda x: f"{x} – {WAVE_CONFIG[x]['name']}"
)

config = WAVE_CONFIG[selected]

# -----------------------------
# PAGE HEADER
# -----------------------------
st.markdown(f"""
# WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE  
## {config['name']} (LIVE Demo)
Mode: **Standard** · Benchmark: **{config['benchmark']}** · Style: **{config['style']}** · Type: **AI-Managed Wave**
""")

# -----------------------------
# LOAD THE DATA
# -----------------------------
df = load_holdings(config["csv"])

if df is None:
    st.warning(f"""
### Holdings for **{config['name']}** haven’t been wired in yet.  
**{selected}** is live today; this Wave slot is ready for its CSV link next.
""")
    st.stop()

# -----------------------------
# MAIN DISPLAY – 3 COLUMNS
# -----------------------------
col1, col2 = st.columns([1.2, 1])

# -----------------------------
# TOP 10 TABLE
# -----------------------------
with col1:
    st.subheader("Top 10 holdings")
    st.dataframe(df[["ticker", "weight"]].head(10), use_container_width=True)

# -----------------------------
# WAVE SNAPSHOT BOX
# -----------------------------
with col2:
    st.subheader("Wave snapshot")

    total_holdings = len(df)
    largest = df.iloc[0]["weight"]

    st.markdown(f"""
    **Wave:** `{selected}` – {config['name']}  
    **Total holdings:** {total_holdings}  
    **Largest position:** {largest:.2f}%  
    **Equity/Cash mix:** 100% / 0%  
    """)

# -----------------------------
# CHARTS
# -----------------------------
st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Top 10 profile – Wave weight distribution")
    fig = px.bar(df.head(10), x="ticker", y="weight")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Holding rank curve")
    fig2 = px.line(df.reset_index(), x=df.index, y="weight")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)