import streamlit as st
st.set_page_config(page_title="WAVES Intelligence™", layout="wide")
st.markdown("### ✅ WAVES BOOT GUARD: app.py loaded")

try:
    # redeploy trigger (no functional change)
    pass
except Exception as e:
    st.error("❌ Boot failed — showing exception")
    st.exception(e)
    st.stop()
