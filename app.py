import streamlit as st
import pandas as pd

# Basic page setup
st.set_page_config(
    page_title="WAVES Simple Dashboard",
    layout="wide"
)

st.title("🌊 WAVES Simple Dashboard")
st.write(
    """
    This is a lightweight version of your WAVES dashboard running on Streamlit.  
    If you can see this screen, the app is deployed and working correctly.

    Next step: we can wire this to your live holdings / benchmark data.
    """
)

# Sidebar info
st.sidebar.header("Status")
st.sidebar.success("App is running without OpenAI or infinite loops ✅")

st.sidebar.markdown("---")
st.sidebar.write(
    "You can upload a CSV below to preview your wave holdings or any market data."
)

# CSV uploader
st.subheader("Upload a CSV file (optional)")
uploaded_file = st.file_uploader(
    "Upload a CSV file with your data (e.g., holdings, prices, or benchmark series).",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data")
        st.dataframe(df, use_container_width=True)

        # Some simple summary stats
        st.write("### Summary statistics")
        st.write(df.describe(include="all"))
    except Exception as e:
        st.error(f"Couldn't read the CSV file: {e}")

st.markdown("---")
st.caption(
    "WAVES Simple • This is a placeholder dashboard. "
    "We can replace this with your full performance engine once everything is stable."
)

  