import streamlit as st
import pandas as pd

st.set_page_config(page_title="Waves Simple Console")

st.title("🌊 WAVES SIMPLE CONSOLE")

uploaded_file = st.file_uploader("Upload your CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("CSV Preview")
    st.dataframe(df)
else:
    st.info("Upload a CSV file to begin.")
