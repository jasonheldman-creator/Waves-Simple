This PR intentionally removes all f-strings from st.sidebar.info() calls, even where they are technically valid, because they have proven fragile in Streamlit when combined with emojis, mobile edits, or conditional rendering.

Sidebar info banners should use plain string literals or explicit string concatenation only.

All sidebar info messages must be single-line, with no triple-quoted strings, no embedded newlines, and no formatted expressions.

This PR is a preventative hardening pass, not a behavioral change — logic, session state, and rendering behavior must remain identical.

Add a brief comment in app.py documenting this rule so future edits do not reintroduce the failure mode.