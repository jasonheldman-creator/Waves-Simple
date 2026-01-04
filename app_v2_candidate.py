#!/usr/bin/env python3
"""
WAVES Intelligence™ Institutional Console - V2 CANDIDATE (EXPERIMENTAL)

⚠️ WARNING: This is an EXPERIMENTAL version of the WAVES console.
⚠️ This file is NOT the production entry point.
⚠️ For production use, please use app.py

This candidate version was proposed in PR #73 as a complete rewrite.
To preserve the existing production features, this version has been
saved as app_v2_candidate.py for evaluation and testing.

Usage: streamlit run app_v2_candidate.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

# Import real engines
import waves_engine as we
import vector_truth as vt
import decision_engine as de

# Page configuration
st.set_page_config(
    page_title="WAVES Intelligence™ — V2 CANDIDATE (EXPERIMENTAL)",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# UTILITIES
# ============================================================

def format_pct(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage."""
    if pd.isna(value) or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_num(value: float, decimals: int = 2) -> str:
    """Format number with decimals."""
    if pd.isna(value) or not np.isfinite(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def compute_metrics(nav_df: pd.DataFrame) -> Dict[str, float]:
    """Compute standard performance metrics from NAV dataframe."""
    if nav_df.empty or len(nav_df) < 2:
        return {}
    
    wave_ret = nav_df["wave_ret"]
    bm_ret = nav_df["bm_ret"]
    
    # Returns
    def period_return(ret_series: pd.Series, days: int) -> float:
        if len(ret_series) < days:
            return np.nan
        period_ret = ret_series.iloc[-days:]
        return float((1 + period_ret).prod() - 1)
    
    r1d_w = float(wave_ret.iloc[-1]) if len(wave_ret) > 0 else np.nan
    r1d_b = float(bm_ret.iloc[-1]) if len(bm_ret) > 0 else np.nan
    r30_w = period_return(wave_ret, 30)
    r30_b = period_return(bm_ret, 30)
    r60_w = period_return(wave_ret, 60)
    r60_b = period_return(bm_ret, 60)
    r365_w = period_return(wave_ret, 365)
    r365_b = period_return(bm_ret, 365)
    
    # Alpha
    a1d = r1d_w - r1d_b
    a30 = r30_w - r30_b
    a60 = r60_w - r60_b
    a365 = r365_w - r365_b
    
    # Risk
    vol_w = float(wave_ret.std() * np.sqrt(252))
    vol_b = float(bm_ret.std() * np.sqrt(252))
    
    excess = wave_ret - bm_ret
    te = float(excess.std() * np.sqrt(252))
    ir = float(a365 / te) if te > 0 and np.isfinite(a365) else np.nan
    
    # Drawdown
    nav_w = nav_df["wave_nav"]
    running_max = nav_w.cummax()
    dd = (nav_w / running_max) - 1.0
    mdd = float(dd.min())
    
    return {
        "r1d_w": r1d_w, "r1d_b": r1d_b, "a1d": a1d,
        "r30_w": r30_w, "r30_b": r30_b, "a30": a30,
        "r60_w": r60_w, "r60_b": r60_b, "a60": a60,
        "r365_w": r365_w, "r365_b": r365_b, "a365": a365,
        "vol_w": vol_w, "vol_b": vol_b, "te": te, "ir": ir, "mdd": mdd,
    }


def create_nav_chart(nav_df: pd.DataFrame, wave_name: str) -> go.Figure:
    """Create NAV performance chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=nav_df.index,
        y=nav_df["wave_nav"],
        mode='lines',
        name=wave_name,
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=nav_df.index,
        y=nav_df["bm_nav"],
        mode='lines',
        name='Benchmark',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{wave_name} vs Benchmark (NAV)",
        xaxis_title="Date",
        yaxis_title="NAV (Base = 1.0)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚠️ WAVES V2 CANDIDATE")
st.sidebar.markdown("**EXPERIMENTAL VERSION**")
st.sidebar.warning("This is NOT the production console. Use app.py for production.")

# Wave selector
all_waves = we.get_all_waves()
selected_wave = st.sidebar.selectbox(
    "Select Wave",
    all_waves,
    index=0 if "US MegaCap Core Wave" not in all_waves else all_waves.index("US MegaCap Core Wave")
)

# Mode selector
modes = we.get_modes()
selected_mode = st.sidebar.selectbox(
    "Operating Mode",
    modes,
    index=0
)

# Time window
days_options = [90, 180, 365]
selected_days = st.sidebar.selectbox(
    "Analysis Window (Days)",
    days_options,
    index=2
)

st.sidebar.markdown("---")
st.sidebar.caption("WAVES Intelligence™ V2 Candidate")
st.sidebar.caption("⚠️ EXPERIMENTAL - NOT FOR PRODUCTION USE")

# ============================================================
# MAIN CONTENT - TABS
# ============================================================

st.title("⚠️ WAVES Intelligence™ V2 Candidate (EXPERIMENTAL)")
st.warning("**EXPERIMENTAL VERSION** - This is NOT the production console. For production use, run: `streamlit run app.py`")
st.caption(f"**{selected_wave}** — {selected_mode} Mode — {selected_days}D Window")

# Create tabs
tabs = st.tabs([
    "📊 Overview",
    "📈 Performance",
    "🏆 Rankings",
    "🎯 Attribution",
    "🔬 Diagnostics",
    "🧠 Decision Intelligence"
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tabs[0]:
    st.header("📊 Overview")
    
    with st.spinner("Loading wave data..."):
        try:
            nav_df = we.compute_history_nav(selected_wave, mode=selected_mode, days=selected_days)
            metrics = compute_metrics(nav_df)
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("365D Return", format_pct(metrics.get("r365_w", np.nan)))
                st.metric("365D Alpha", format_pct(metrics.get("a365", np.nan)))
            
            with col2:
                st.metric("60D Return", format_pct(metrics.get("r60_w", np.nan)))
                st.metric("60D Alpha", format_pct(metrics.get("a60", np.nan)))
            
            with col3:
                st.metric("Volatility (Ann.)", format_pct(metrics.get("vol_w", np.nan)))
                st.metric("Info Ratio", format_num(metrics.get("ir", np.nan)))
            
            with col4:
                st.metric("Max Drawdown", format_pct(metrics.get("mdd", np.nan)))
                st.metric("Tracking Error", format_pct(metrics.get("te", np.nan)))
            
            st.markdown("---")
            
            # Holdings
            st.subheader("Current Holdings")
            holdings_df = we.get_wave_holdings(selected_wave)
            if not holdings_df.empty:
                holdings_df["Weight"] = holdings_df["Weight"].apply(lambda x: format_pct(x))
                st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            else:
                st.info("No holdings data available")
            
            # Benchmark
            st.subheader("Benchmark Composition")
            bm_table = we.get_benchmark_mix_table()
            wave_bm = bm_table[bm_table["Wave"] == selected_wave]
            if not wave_bm.empty:
                wave_bm = wave_bm[["Ticker", "Name", "Weight"]].copy()
                wave_bm["Weight"] = wave_bm["Weight"].apply(lambda x: format_pct(x))
                st.dataframe(wave_bm, use_container_width=True, hide_index=True)
            else:
                st.info("Benchmark data unavailable")
                
        except (KeyError, ValueError, RuntimeError) as e:
            st.error(f"Error loading overview data: {str(e)}")
            st.exception(e)
        except Exception as e:
            st.error(f"Unexpected error loading overview: {str(e)}")
            st.exception(e)

# ============================================================
# TAB 2: PERFORMANCE
# ============================================================

with tabs[1]:
    st.header("📈 Performance Analysis")
    
    with st.spinner("Computing performance metrics..."):
        try:
            nav_df = we.compute_history_nav(selected_wave, mode=selected_mode, days=selected_days)
            metrics = compute_metrics(nav_df)
            
            # NAV Chart
            fig = create_nav_chart(nav_df, selected_wave)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Performance Table
            st.subheader("Returns Summary")
            
            perf_data = {
                "Period": ["1D", "30D", "60D", "365D"],
                "Wave Return": [
                    format_pct(metrics.get("r1d_w", np.nan)),
                    format_pct(metrics.get("r30_w", np.nan)),
                    format_pct(metrics.get("r60_w", np.nan)),
                    format_pct(metrics.get("r365_w", np.nan)),
                ],
                "Benchmark Return": [
                    format_pct(metrics.get("r1d_b", np.nan)),
                    format_pct(metrics.get("r30_b", np.nan)),
                    format_pct(metrics.get("r60_b", np.nan)),
                    format_pct(metrics.get("r365_b", np.nan)),
                ],
                "Alpha": [
                    format_pct(metrics.get("a1d", np.nan)),
                    format_pct(metrics.get("a30", np.nan)),
                    format_pct(metrics.get("a60", np.nan)),
                    format_pct(metrics.get("a365", np.nan)),
                ],
            }
            
            st.table(pd.DataFrame(perf_data))
            
            st.markdown("---")
            
            # Risk Metrics
            st.subheader("Risk Metrics")
            
            risk_data = {
                "Metric": [
                    "Volatility (Annualized)",
                    "Tracking Error",
                    "Information Ratio",
                    "Max Drawdown",
                    "Benchmark Volatility"
                ],
                "Value": [
                    format_pct(metrics.get("vol_w", np.nan)),
                    format_pct(metrics.get("te", np.nan)),
                    format_num(metrics.get("ir", np.nan)),
                    format_pct(metrics.get("mdd", np.nan)),
                    format_pct(metrics.get("vol_b", np.nan)),
                ]
            }
            
            st.table(pd.DataFrame(risk_data))
            
        except Exception as e:
            st.error(f"Error computing performance: {str(e)}")
            st.exception(e)

# ============================================================
# TAB 3: RANKINGS
# ============================================================

with tabs[2]:
    st.header("🏆 Wave Rankings")
    
    st.info("Computing multi-wave comparison...")
    
    with st.spinner("Ranking all waves..."):
        try:
            ranking_data = []
            
            for wave in all_waves[:10]:  # Limit to top 10 for performance
                try:
                    nav_df = we.compute_history_nav(wave, mode=selected_mode, days=selected_days)
                    metrics = compute_metrics(nav_df)
                    
                    ranking_data.append({
                        "Wave": wave,
                        "365D Return": metrics.get("r365_w", np.nan),
                        "365D Alpha": metrics.get("a365", np.nan),
                        "Volatility": metrics.get("vol_w", np.nan),
                        "Info Ratio": metrics.get("ir", np.nan),
                        "Max Drawdown": metrics.get("mdd", np.nan),
                    })
                except Exception as e:
                    st.warning(f"Skipped {wave}: {str(e)}")
                    continue
            
            if ranking_data:
                ranking_df = pd.DataFrame(ranking_data)
                ranking_df = ranking_df.sort_values("365D Alpha", ascending=False)
                
                # Format for display
                display_df = ranking_df.copy()
                display_df["365D Return"] = display_df["365D Return"].apply(format_pct)
                display_df["365D Alpha"] = display_df["365D Alpha"].apply(format_pct)
                display_df["Volatility"] = display_df["Volatility"].apply(format_pct)
                display_df["Info Ratio"] = display_df["Info Ratio"].apply(format_num)
                display_df["Max Drawdown"] = display_df["Max Drawdown"].apply(format_pct)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # WaveScore explanation
                with st.expander("ℹ️ About WaveScore Rankings"):
                    st.markdown("""
                    **WaveScore** ranks waves based on:
                    - **365D Alpha**: Excess return vs benchmark
                    - **Information Ratio**: Risk-adjusted alpha quality
                    - **Max Drawdown**: Downside protection
                    - **Volatility**: Total risk profile
                    
                    Waves are sorted by 365D Alpha (highest to lowest).
                    """)
            else:
                st.warning("No ranking data available")
                
        except Exception as e:
            st.error(f"Error generating rankings: {str(e)}")
            st.exception(e)

# ============================================================
# TAB 4: ATTRIBUTION
# ============================================================

with tabs[3]:
    st.header("🎯 Alpha Attribution")
    
    with st.spinner("Computing attribution..."):
        try:
            nav_df = we.compute_history_nav(selected_wave, mode=selected_mode, days=selected_days, include_diagnostics=True)
            metrics = compute_metrics(nav_df)
            
            # Build Vector Truth Report
            total_excess = metrics.get("a365", np.nan)
            
            # Extract diagnostics if available
            alpha_series = None
            regime_series = None
            if hasattr(nav_df, 'attrs') and 'diagnostics' in nav_df.attrs:
                diag_df = nav_df.attrs['diagnostics']
                if not diag_df.empty and 'regime' in diag_df.columns:
                    regime_series = diag_df['regime'].tolist()
                    # Use wave excess returns as alpha series
                    excess_ret = (nav_df["wave_ret"] - nav_df["bm_ret"]).tolist()
                    alpha_series = excess_ret
            
            # Build report
            report = vt.build_vector_truth_report(
                wave_name=selected_wave,
                timeframe_label=f"{selected_days}D",
                total_excess_return=total_excess,
                capital_weighted_alpha=total_excess,  # Simplified for now
                exposure_adjusted_alpha=total_excess,  # Simplified for now
                alpha_series=alpha_series,
                regime_series=regime_series,
                benchmark_snapshot_id="auto-composite-v1",
                benchmark_drift_status="stable"
            )
            
            # Display Vector Truth
            st.subheader("Vector™ Truth Layer")
            
            # Compute reliability metrics
            data_rows = len(nav_df) if not nav_df.empty else 0
            regime_coverage = {}
            if regime_series:
                from collections import Counter
                regime_counts = Counter(regime_series)
                regime_coverage = {
                    "risk_on": regime_counts.get("uptrend", 0) + regime_counts.get("neutral", 0),
                    "risk_off": regime_counts.get("downtrend", 0) + regime_counts.get("panic", 0)
                }
            
            reliability = vt.compute_alpha_reliability_metrics(
                window_days=selected_days,
                bm_drift="stable",
                data_rows=data_rows,
                regime_coverage=regime_coverage,
                alpha_inflation_risk=report.reconciliation.inflation_risk
            )
            
            # Show reliability panel
            reliability_md = vt.render_alpha_reliability_panel(reliability)
            st.markdown(reliability_md)
            
            st.markdown("---")
            
            # Show main attribution
            attribution_confidence = reliability.get("attribution_confidence", "Low")
            truth_md = vt.format_vector_truth_markdown(report, attribution_confidence)
            st.markdown(truth_md)
            
            # Show detailed attribution in expander
            with st.expander("🔍 Detailed Alpha Attribution (Advanced)"):
                attr_detail_md = vt.render_vector_truth_alpha_attribution(report)
                st.markdown(attr_detail_md)
            
            # Alpha Attribution Table (if high confidence)
            if attribution_confidence == "High":
                st.markdown("---")
                st.subheader("Attribution Components Breakdown")
                
                breakdown = vt.extract_alpha_attribution_breakdown(report)
                
                comp_data = {
                    "Component": [
                        "Exposure & Timing",
                        "VIX & Regime Overlays",
                        "Asset Selection (Exposure-Adjusted)",
                        "Total Excess Return",
                        "Residual Strategy Return"
                    ],
                    "Value": [
                        format_pct(breakdown.get("exposure_timing")),
                        format_pct(breakdown.get("vix_regime_overlays")),
                        format_pct(breakdown.get("asset_selection")),
                        format_pct(breakdown.get("total_excess")),
                        format_pct(breakdown.get("residual_strategy"))
                    ]
                }
                
                st.table(pd.DataFrame(comp_data))
            
        except Exception as e:
            st.error(f"Error computing attribution: {str(e)}")
            st.exception(e)

# ============================================================
# TAB 5: DIAGNOSTICS
# ============================================================

with tabs[4]:
    st.header("🔬 System Diagnostics")
    
    with st.spinner("Loading diagnostics..."):
        try:
            # Get VIX/Regime diagnostics
            diag_df = we.get_vix_regime_diagnostics(selected_wave, mode=selected_mode, days=selected_days)
            
            if not diag_df.empty:
                st.subheader("VIX & Regime Overlay Activity")
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_vix = diag_df["vix"].mean()
                    st.metric("Avg VIX", format_num(avg_vix, 1))
                
                with col2:
                    avg_exposure = diag_df["exposure"].mean()
                    st.metric("Avg Exposure", format_pct(avg_exposure))
                
                with col3:
                    avg_safe = diag_df["safe_fraction"].mean()
                    st.metric("Avg Safe %", format_pct(avg_safe))
                
                st.markdown("---")
                
                # Recent diagnostics table
                st.subheader("Recent Activity (Last 10 Days)")
                
                recent_diag = diag_df.tail(10).copy()
                display_cols = ["regime", "vix", "exposure", "safe_fraction", "aggregated_risk_state"]
                
                # Format for display
                recent_diag["vix"] = recent_diag["vix"].apply(lambda x: format_num(x, 1))
                recent_diag["exposure"] = recent_diag["exposure"].apply(format_pct)
                recent_diag["safe_fraction"] = recent_diag["safe_fraction"].apply(format_pct)
                
                st.dataframe(
                    recent_diag[display_cols],
                    use_container_width=True
                )
                
                # Download option
                csv = diag_df.to_csv()
                st.download_button(
                    label="📥 Download Full Diagnostics (CSV)",
                    data=csv,
                    file_name=f"{selected_wave}_{selected_mode}_{selected_days}d_diagnostics.csv",
                    mime="text/csv"
                )
                
                # Strategy Attribution (if available)
                with st.expander("📊 Strategy Attribution (Advanced)"):
                    try:
                        strat_attr = we.get_strategy_attribution(selected_wave, selected_mode, selected_days)
                        
                        if strat_attr.get("ok"):
                            summary = strat_attr["summary"]
                            
                            st.markdown("**Strategy Impact Summary**")
                            st.json(summary)
                        else:
                            st.info("Strategy attribution not available")
                    except (AttributeError, KeyError, RuntimeError) as e:
                        st.info(f"Strategy attribution not available: {str(e)}")
                    except Exception:
                        st.info("Strategy attribution feature not available in this version")
                
            else:
                st.info("No diagnostics data available")
                
        except Exception as e:
            st.error(f"Error loading diagnostics: {str(e)}")
            st.exception(e)

# ============================================================
# TAB 6: DECISION INTELLIGENCE
# ============================================================

with tabs[5]:
    st.header("🧠 Decision Intelligence")
    
    with st.spinner("Generating intelligence..."):
        try:
            nav_df = we.compute_history_nav(selected_wave, mode=selected_mode, days=selected_days)
            metrics = compute_metrics(nav_df)
            
            # Build context for decision engine
            ctx = {
                "wave_name": selected_wave,
                "mode": selected_mode,
                "r1d": metrics.get("r1d_w", np.nan),
                "r30": metrics.get("r30_w", np.nan),
                "r60": metrics.get("r60_w", np.nan),
                "r365": metrics.get("r365_w", np.nan),
                "a1d": metrics.get("a1d", np.nan),
                "a30": metrics.get("a30", np.nan),
                "a60": metrics.get("a60", np.nan),
                "a365": metrics.get("a365", np.nan),
                "vol_w": metrics.get("vol_w", np.nan),
                "te": metrics.get("te", np.nan),
                "ir": metrics.get("ir", np.nan),
                "mdd": metrics.get("mdd", np.nan),
            }
            
            # Daily Wave Activity
            activity = de.build_daily_wave_activity(ctx)
            
            st.subheader("📰 Daily Wave Activity")
            st.markdown(f"**{activity['headline']}**")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**What Changed:**")
                for item in activity["what_changed"]:
                    st.markdown(f"• {item}")
            
            with col2:
                st.markdown("**Why:**")
                for item in activity["why"]:
                    st.markdown(f"• {item}")
            
            st.markdown("---")
            
            st.markdown("**Results:**")
            for item in activity["results"]:
                st.markdown(f"• {item}")
            
            st.markdown("---")
            
            st.markdown("**Checks:**")
            for item in activity["checks"]:
                st.markdown(f"• {item}")
            
            st.markdown("---")
            
            # Actions / Watch / Notes
            decisions = de.generate_decisions(ctx)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🎯 Actions**")
                for action in decisions["actions"]:
                    st.warning(f"• {action}")
            
            with col2:
                st.markdown("**👀 Watch**")
                for watch in decisions["watch"]:
                    st.info(f"• {watch}")
            
            with col3:
                st.markdown("**📝 Notes**")
                for note in decisions["notes"]:
                    st.success(f"• {note}")
            
        except Exception as e:
            st.error(f"Error generating decision intelligence: {str(e)}")
            st.exception(e)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("WAVES Intelligence™ V2 Candidate — ⚠️ EXPERIMENTAL VERSION")
st.caption("This is NOT the production console. Use app.py for production.")
st.caption("This is not investment advice. For institutional use only.")
