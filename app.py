import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Page setup – DARK MODE theme
# --------------------------------------------------
st.set_page_config(
    page_title="🌊 Waves Simple Console",
    layout="wide"
)

# ---- Custom Dark CSS ----
custom_css = """
<style>
    body {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .stApp {
        background-color: #0d1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6, label {
        color: #e6edf3 !important;
    }
    .stDataFrame table {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        color: #e6edf3;
        padding: 0.7rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌊 **WAVES SIMPLE CONSOLE**")
st.markdown("Upload a **Wave snapshot CSV** to view your portfolio.")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
st.sidebar.radio("SmartSafe™ mode", ["Neutral", "Defensive", "Max Safe"])
st.sidebar.slider("Equity tilt (human override, %)", -20, 20, 0)
st.sidebar.slider("Growth style tilt (bps)", -300, 300, 0)
st.sidebar.slider("Value style tilt (bps)", -300, 300, 0)
st.sidebar.caption("This console is read-only — no live trades occur.")

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Check for common fields
ticker_col = "Ticker" if "Ticker" in df.columns else df.columns[0]
weight_col = "Wave_Wt_Final" if "Wave_Wt_Final" in df.columns else None
sector_col = "Sector" if "Sector" in df.columns else None
dollar_col = "Dollar_Amount" if "Dollar_Amount" in df.columns else None

# --------------------------------------------------
# TABS
# --------------------------------------------------
tabs = st.tabs(["📊 Preview & Stats", "📈 Charts", "🎯 Overrides & Targets"])

# ========== PREVIEW TAB ==========
with tabs[0]:
    st.subheader("Portfolio preview")
    st.dataframe(df, use_container_width=True)

    # Simple Key Stats
    st.subheader("Key stats")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total holdings", len(df))
    col2.metric("Equity weight", "100.0%")
    if dollar_col:
        col3.metric("Largest position", f"{df[dollar_col].max():,.1f}")
    else:
        col3.metric("Largest position", "N/A")

# ========== CHARTS TAB ==========
with tabs[1]:
    st.subheader("📈 Portfolio Charts")

    chart_type = st.selectbox(
        "Choose a chart",
        ["Top 10 Holdings", "Weight Distribution (Pie)", "Sector Exposure"]
    )

    # ---- Top 10 Holdings Chart ----
    if chart_type == "Top 10 Holdings":
        if not dollar_col:
            st.error("Dollar_Amount column not found in CSV.")
        else:
            top10 = df.nlargest(10, dollar_col)

            fig = px.bar(
                top10,
                x=dollar_col,
                y=ticker_col,
                orientation="h",
                title="Top 10 Holdings by Dollar Amount",
                text=dollar_col,
            )
            fig.update_layout(
                template="plotly_dark",
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Pie Chart ----
    if chart_type == "Weight Distribution (Pie)":
        if not weight_col:
            st.error("Wave_Wt_Final column not found in CSV.")
        else:
            fig = px.pie(
                df,
                names=ticker_col,
                values=weight_col,
                title="Portfolio Weight Breakdown",
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ---- Sector Exposure ----
    if chart_type == "Sector Exposure":
        if not sector_col or not weight_col:
            st.error("Sector or weight column missing.")
        else:
            sector_df = df.groupby(sector_col)[weight_col].sum().reset_index()
            sector_df = sector_df.sort_values(weight_col, ascending=True)

            fig = px.bar(
                sector_df,
                x=weight_col,
                y=sector_col,
                orientation="h",
                title="Sector Exposure (% Weight)",
                text=weight_col,
            )
            fig.update_layout(
                template="plotly_dark",
                height=550,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

# ========== OVERRIDES TAB ==========
with tabs[2]:
    st.subheader("🎯 Human Overrides & Targets (coming next)")
    st.write("We will add tilt controls, SmartSafe integration, and WaveScore adjustments here.")