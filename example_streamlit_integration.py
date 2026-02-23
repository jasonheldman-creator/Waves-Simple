#!/usr/bin/env python3
"""
Example integration of alpha attribution into a Streamlit app.

This shows how to add the alpha attribution table to app.py
"""

import streamlit as st
import pandas as pd

# Placeholder for actual imports (would be in app.py)
def example_alpha_attribution_tab():
    """
    Example tab/section for alpha attribution in Streamlit app.
    
    To integrate into app.py:
    1. Add this as a new tab in the main interface
    2. Import waves_engine and alpha_attribution modules
    3. Use wave selection from sidebar
    """
    
    st.header("🎯 Alpha Attribution Decomposition")
    
    st.markdown("""
    **Precise, reconciled decomposition of Wave alpha into five components:**
    
    1️⃣ **Exposure & Timing Alpha** - Dynamic exposure scaling and entry/exit timing  
    2️⃣ **Regime & VIX Overlay Alpha** - VIX gating and risk-off defensive positioning  
    3️⃣ **Momentum & Trend Alpha** - Momentum-based weight tilting and trend following  
    4️⃣ **Volatility & Risk Control Alpha** - Volatility targeting and drawdown limits  
    5️⃣ **Asset Selection Alpha** - Security selection and portfolio construction (residual)
    
    **Reconciliation Enforced:** All components sum exactly to realized Wave alpha.
    """)
    
    # Configuration controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_wave = st.selectbox(
            "Select Wave",
            ["US MegaCap Core Wave", "AI & Cloud MegaCap Wave", "S&P 500 Wave"],
            key="attr_wave_select"
        )
    
    with col2:
        mode = st.selectbox(
            "Mode",
            ["Standard", "Alpha-Minus-Beta", "Private Logic"],
            key="attr_mode_select"
        )
    
    with col3:
        days = st.selectbox(
            "Period",
            [90, 180, 365],
            index=2,
            key="attr_days_select"
        )
    
    if st.button("🔄 Compute Attribution", key="compute_attribution"):
        with st.spinner("Computing alpha attribution..."):
            try:
                # In actual app.py, import and use real functions:
                # import waves_engine as we
                # daily_df, summary = we.compute_alpha_attribution(
                #     wave_name=selected_wave,
                #     mode=mode,
                #     days=days
                # )
                
                # For this example, show the expected output structure
                st.success("✅ Attribution computed successfully")
                
                # Show summary
                st.subheader("Attribution Summary")
                
                # Example summary table (would come from actual computation)
                summary_data = {
                    "Component": [
                        "1️⃣ Exposure & Timing",
                        "2️⃣ Regime & VIX",
                        "3️⃣ Momentum & Trend",
                        "4️⃣ Volatility Control",
                        "5️⃣ Asset Selection (Residual)",
                        "**Total Alpha**"
                    ],
                    "Cumulative Alpha": [
                        "+1.25%",
                        "+0.35%",
                        "+0.42%",
                        "+0.08%",
                        "+2.15%",
                        "**+4.25%**"
                    ],
                    "Contribution to Total": [
                        "29.4%",
                        "8.2%",
                        "9.9%",
                        "1.9%",
                        "50.6%",
                        "**100.0%**"
                    ]
                }
                
                st.table(pd.DataFrame(summary_data))
                
                # Reconciliation check
                st.info("""
                **✅ Reconciliation Check Passed**
                - Total Realized Alpha: +4.25%
                - Sum of Components: +4.25%
                - Reconciliation Error: 0.0000%
                """)
                
                # Daily attribution sample
                st.subheader("Daily Attribution Sample (Last 10 Days)")
                
                # Example daily data (would come from actual computation)
                daily_example = {
                    "Date": ["2025-12-20", "2025-12-19", "2025-12-18", "2025-12-17", "2025-12-16"],
                    "VIX": [19.4, 21.3, 17.9, 25.6, 18.2],
                    "Regime": ["Neutral", "Downtrend", "Uptrend", "Panic", "Neutral"],
                    "Exposure%": [112, 95, 118, 82, 105],
                    "Safe%": [8, 15, 5, 30, 10],
                    "ExposTimα": ["+0.15%", "-0.08%", "+0.22%", "-0.25%", "+0.08%"],
                    "RegVIXα": ["+0.05%", "+0.12%", "+0.01%", "+0.35%", "+0.03%"],
                    "MomTrndα": ["+0.10%", "-0.05%", "+0.15%", "-0.10%", "+0.12%"],
                    "VolCtrlα": ["+0.02%", "+0.03%", "-0.01%", "+0.05%", "+0.01%"],
                    "AssetSelα": ["+0.01%", "+0.02%", "+0.08%", "-0.05%", "+0.06%"],
                    "Totalα": ["+0.33%", "+0.05%", "+0.45%", "+0.00%", "+0.30%"]
                }
                
                st.dataframe(
                    pd.DataFrame(daily_example),
                    hide_index=True,
                    use_container_width=True
                )
                
                # Download option
                st.download_button(
                    label="📥 Download Full Daily Attribution (CSV)",
                    data="Date,VIX,Regime,ExposureTimingα,RegimeVIXα,MomentumTrendα,VolatilityControlα,AssetSelectionα,TotalAlpha\n...",
                    file_name=f"{selected_wave.replace(' ', '_')}_attribution_{days}d.csv",
                    mime="text/csv",
                    key="download_attribution"
                )
                
                # Visualization option (expandable)
                with st.expander("📊 Component Contribution Chart"):
                    st.info("Chart showing relative contribution of each alpha component over time")
                    # In actual implementation, would include plotly chart:
                    # fig = create_attribution_chart(daily_df)
                    # st.plotly_chart(fig, use_container_width=True)
                
                # Detailed explanation (expandable)
                with st.expander("ℹ️ Component Definitions"):
                    st.markdown("""
                    **1️⃣ Exposure & Timing Alpha**
                    - Value from dynamically adjusting exposure above/below baseline
                    - Sources: Entry/exit timing, drawdown avoidance
                    - Example: Higher exposure during market rally
                    
                    **2️⃣ Regime & VIX Overlay Alpha**
                    - Value from shifting to safe assets during high VIX/risk-off
                    - Sources: VIX gating, regime-based risk management
                    - Example: Increased safe allocation when VIX > 25
                    
                    **3️⃣ Momentum & Trend Alpha**
                    - Value from overweighting winners, underweighting losers
                    - Sources: Momentum tilts, trend confirmation, rotations
                    - Example: Overweighting NVDA during strong uptrend
                    
                    **4️⃣ Volatility & Risk Control Alpha**
                    - Value from scaling exposure to maintain target volatility
                    - Sources: Volatility targeting, drawdown limits
                    - Example: Reducing exposure when volatility spikes
                    
                    **5️⃣ Asset Selection Alpha (Residual)**
                    - Value from security selection after all other effects
                    - Sources: Stock picks, sector allocation, rebalancing
                    - This is the residual that ensures perfect reconciliation
                    """)
                
            except Exception as e:
                st.error(f"Error computing attribution: {str(e)}")
                st.exception(e)
    
    # Help section
    with st.expander("❓ Help & Documentation"):
        st.markdown("""
        ### How to Use Alpha Attribution
        
        1. **Select Wave:** Choose the Wave to analyze
        2. **Select Mode:** Operating mode affects base exposure and risk management
        3. **Select Period:** Analysis window (90, 180, or 365 days)
        4. **Compute:** Click button to run attribution analysis
        
        ### Reconciliation Guarantee
        
        All five components are computed from actual realized returns and sum **exactly** to total alpha:
        
        ```
        Total Alpha = Wave Return - Benchmark Return
        Total Alpha = Σ (All 5 Components)
        ```
        
        Reconciliation error is monitored and must be < 0.01% for valid attribution.
        
        ### Data Sources
        
        - Same comprehensive return series used in WaveScore and performance metrics
        - Daily diagnostics: VIX, regime, exposure, safe allocation
        - All values traceable to actual trading day returns
        
        ### Additional Resources
        
        - See `ALPHA_ATTRIBUTION_DOCUMENTATION.md` for technical details
        - Run `demo_alpha_attribution.py` for standalone examples
        - Run `test_alpha_attribution.py` to validate reconciliation
        """)


def main():
    """
    Example main function showing how this would integrate into app.py
    """
    st.set_page_config(page_title="Alpha Attribution Demo", layout="wide")
    
    st.title("🌊 WAVES Intelligence™ - Alpha Attribution")
    
    # In actual app.py, this would be one tab among many:
    # tabs = st.tabs(["Overview", "Performance", "Alpha Attribution", "Diagnostics"])
    # with tabs[2]:
    #     example_alpha_attribution_tab()
    
    # For this standalone demo:
    example_alpha_attribution_tab()
    
    # Footer
    st.markdown("---")
    st.caption("WAVES Intelligence™ Alpha Attribution Engine v1.0 - Governance-Ready Attribution")


if __name__ == "__main__":
    main()
