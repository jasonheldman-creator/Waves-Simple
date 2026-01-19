def compute_portfolio_snapshot():
    try:
        # Your logic to compute the portfolio snapshot here
        st.session_state['portfolio_snapshot_debug'] = snapshot
    except Exception as e:
        st.error(f"Error occurred: {e}")
    finally:
        # Code to execute regardless of whether an exception occurred
        pass