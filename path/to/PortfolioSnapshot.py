import streamlit as st

# Global variable to keep track of render counter
if 'render_counter' not in st.session_state:
    st.session_state.render_counter = 0

# Increment render counter on every render
st.session_state.render_counter += 1

# Display the counter on the Portfolio Snapshot banner
st.header(f"Portfolio Snapshot - Rendered {st.session_state.render_counter} times")