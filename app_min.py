import streamlit as st

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

def main():
    st.title("WAVES — Recovery Mode")
    st.success("app_min.main() is now running")

    st.divider()
    st.write("🔍 Import diagnostics starting...")

    try:
        import waves
        st.success("✅ waves module imported successfully")
    except Exception as e:
        st.error("❌ waves import failed")
        st.exception(e)
        return  # stop here if import fails

    # 🔬 SAFE READ-ONLY INSPECTION
    st.divider()
    st.write("🧪 waves module inspection")

    try:
        st.write("waves module file:", waves.__file__)
        st.write(
            "waves module attributes (sample):",
            sorted(dir(waves))[:25]
        )
        st.success("waves module inspection completed")
    except Exception as e:
        st.error("waves inspection failed")
        st.exception(e)

if __name__ == "__main__":
    main()