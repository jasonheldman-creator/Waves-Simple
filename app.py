# app.py
"""
WAVES Intelligence™ — Emergency Safe Mode Console

This version intentionally does NOT import the engine.
It is guaranteed to render so the app never shows a blank screen
during live meetings. After the meeting, we can reconnect the
full Vector console + engine.
"""

import streamlit as st

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Safe Mode — Live engine temporarily offline for maintenance.")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Autonomous Wealth Engine Overview")
    st.markdown(
        """
        **WAVES Intelligence™** runs a live, autonomous portfolio engine:

        - Multiple **Waves** (S&P Wave, Growth Wave, Future Power & Energy, Clean Transit-Infrastructure, etc.)
        - Each Wave holds a **basket of stocks** with smart weights and rebalancing rules
        - The engine continuously tracks:
          - 1-Day, 30-Day, and 60-Day returns
          - **Alpha captured** vs. a benchmark (SPY, QQQ, BTC, etc.)
          - **Risk metrics** like volatility and drawdown
        - All positions and performance roll into:
          - A **top-10 holdings view** for each Wave
          - A **benchmark comparison chart**
          - A unified **All-Waves dashboard**
        """
    )

with col2:
    st.subheader("What you’re seeing in production")
    st.markdown(
        """
        Right now the app is running in **Safe Mode**:

        - The full live data engine is **intact in the repo**
        - We’ve temporarily disconnected it from the UI
        - This avoids any upstream data-provider issues (e.g., Yahoo Finance)
          from interrupting the demo.
        
        After this meeting, we simply reconnect:
        - `waves_engine.py` → live price feeds
        - The full **Vector 1 console** (tabs, alpha matrix, risk, mode comparison)
        """
    )

st.markdown("---")

st.subheader("How the engine works (talk track)")

st.markdown(
    """
    1. **CSV Inputs**  
       - `wave_weights.csv` defines each Wave and its stocks/weights.  
       - An S&P universe file provides the **full 500-stock core**.

    2. **Dynamic S&P Wave**  
       - We build a **market-cap S&P core**.  
       - Then overlay your custom tilts (e.g., overweight AI, underweight rate-sensitive names).  
       - That creates an **S&P Wave designed to beat the ETF** without changing the client’s “S&P” story.

    3. **Live Price Engine**  
       - A Python engine pulls prices for every ticker.  
       - It computes **daily returns**, **cumulative performance**, and **alpha vs. benchmark**.  
       - It logs a full **audit trail** for compliance.

    4. **Institutional Console (Vector 1)**  
       - For each Wave, the console shows:
         - 1-Day / 30-Day / 60-Day **return & alpha**
         - A **performance chart vs. benchmark**
         - **Top 10 holdings** with one-click quote links
         - An **All-Waves dashboard** that ranks Waves by alpha
    """
)

st.markdown("---")
st.caption(
    "WAVES Intelligence™ • Autonomous Wealth Engine • Safe Mode view — "
    "live engine ready to reconnect after data-provider issues are cleared."
)