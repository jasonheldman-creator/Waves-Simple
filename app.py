import streamlit as st

st.set_page_config(page_title="WAVES – Recovery Mode")

st.title("WAVES Recovery Mode")
st.write("app.py is alive.")

# Import and run app_min safely
try:
    import app_min
    if hasattr(app_min, "main"):
        app_min.main()
    else:
        st.warning("app_min.main() not found")
except Exception as e:
    st.error("app_min import failed")
    st.exception(e)