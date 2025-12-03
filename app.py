import streamlit as st
import pandas as pd

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
    .navbar {
        background-color: #111418 !important;
    }
    .sidebar .sidebar-content {
        background-color: #111418 !important;
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
    col3.metric("Largest position", f"{df['Dollar_Amount'].max():,.1f}")

# ========== CHARTS TAB (placeholder until ready) ==========
with tabs[1]:
    st.subheader("Charts (coming next)")
    st.write("We will add interactive weight charts, sector bars, and exposure visuals here.")

# ========== OVERRIDES TAB (placeholder) ==========
with tabs[2]:
    st.subheader("Human Overrides & Targets (coming next)")
    st.write("We will add tilt controls, SmartSafe integration, and WaveScore adjustments here.")