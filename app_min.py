import streamlit as st

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

def main():
    st.title("WAVES – Recovery Mode")
    st.success("app_min.main() is now running")

    st.divider()
    st.write("🔍 Import diagnostics starting...")

    try:
        import waves
        st.success("✅ waves module imported successfully")
    except Exception as e:
        st.error("❌ waves import failed")
        st.exception(e)

if __name__ == "__main__":
    main()