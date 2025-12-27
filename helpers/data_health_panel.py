"""
Data Health Panel
Provides visibility into system health, cache performance, and data availability.
Enhanced with degraded data diagnostics and ticker failure tracking.
"""

import streamlit as st
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List


def load_ticker_failures() -> pd.DataFrame:
    """
    Load ticker failures from the analytics pipeline output.
    
    Returns:
        DataFrame with ticker failure records or empty DataFrame
    """
    try:
        ticker_failures_path = os.path.join('data', 'waves', 'ticker_failures.csv')
        if os.path.exists(ticker_failures_path):
            df = pd.read_csv(ticker_failures_path)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_validation_report() -> pd.DataFrame:
    """
    Load validation report from the analytics pipeline.
    
    Returns:
        DataFrame with validation results or empty DataFrame
    """
    try:
        validation_path = os.path.join('data', 'waves', 'validation_report.csv')
        if os.path.exists(validation_path):
            df = pd.read_csv(validation_path)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def render_degraded_data_diagnostics():
    """
    Render diagnostics panel for degraded data and ticker failures.
    Shows detailed information about which waves have degraded data and why.
    """
    st.subheader("🔍 Degraded Data Diagnostics")
    st.caption("Detailed view of data quality issues and ticker failures")
    
    try:
        # Load validation report
        validation_df = load_validation_report()
        
        if not validation_df.empty:
            # Filter for degraded waves
            degraded_mask = validation_df['data_status'].isin(['PARTIAL', 'NO_DATA'])
            degraded_waves = validation_df[degraded_mask]
            
            if not degraded_waves.empty:
                st.warning(f"⚠️ {len(degraded_waves)} wave(s) have degraded data")
                
                # Display degraded waves summary
                summary_cols = ['wave_id', 'data_status', 'data_quality', 'ticker_failures', 'issue_count']
                available_cols = [col for col in summary_cols if col in degraded_waves.columns]
                st.dataframe(
                    degraded_waves[available_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show detailed issues for each degraded wave
                with st.expander("📋 Detailed Issues by Wave"):
                    for _, row in degraded_waves.iterrows():
                        wave_id = row['wave_id']
                        data_status = row.get('data_status', 'UNKNOWN')
                        data_quality = row.get('data_quality', '')
                        
                        st.markdown(f"**{wave_id}** - Status: `{data_status}`")
                        if data_quality:
                            st.caption(f"Quality: {data_quality}")
                        st.markdown("---")
            else:
                st.success("✅ All waves have OK data status")
        else:
            st.info("ℹ️ No validation report available. Run analytics pipeline to generate diagnostics.")
        
        # Load and display ticker failures
        st.markdown("---")
        st.markdown("### Ticker Failures")
        
        ticker_failures_df = load_ticker_failures()
        
        if not ticker_failures_df.empty:
            st.error(f"❌ {len(ticker_failures_df)} ticker failure(s) detected")
            
            # Group by wave to show affected waves
            if 'wave_id' in ticker_failures_df.columns:
                failures_by_wave = ticker_failures_df.groupby('wave_id').size().reset_index(name='failure_count')
                failures_by_wave = failures_by_wave.sort_values('failure_count', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Affected Waves**")
                    st.dataframe(
                        failures_by_wave,
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    st.markdown("**All Failed Tickers**")
                    if 'ticker' in ticker_failures_df.columns:
                        unique_tickers = ticker_failures_df['ticker'].unique()
                        st.write(", ".join(sorted(unique_tickers)))
            
            # Detailed failure list
            with st.expander("📊 Detailed Ticker Failures"):
                st.dataframe(
                    ticker_failures_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            # CSV Download button
            csv = ticker_failures_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Ticker Failures CSV",
                data=csv,
                file_name=f"ticker_failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_ticker_failures"
            )
        else:
            st.success("✅ No ticker failures detected")
        
        # Full validation report download
        if not validation_df.empty:
            st.markdown("---")
            st.markdown("### Full Validation Report")
            
            csv = validation_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Validation Report CSV",
                data=csv,
                file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_validation_report"
            )
        
    except Exception as e:
        st.error(f"❌ Error displaying degraded data diagnostics: {str(e)}")


def render_data_health_panel():
    """
    Render the Data Health panel showing system status and metrics.
    Enhanced with fail-safe error handling to prevent crashes.
    """
    st.subheader("📊 Data Health Panel")
    st.caption("System health metrics and data availability status")
    
    try:
        # Import health status function
        from helpers.ticker_sources import get_ticker_health_status, test_ticker_fetch
        
        # Get health status (with fail-safe)
        try:
            health = get_ticker_health_status()
        except Exception as e:
            # Fail-safe: Show degraded status but don't crash
            st.warning(f"⚠️ Unable to fetch health status: {str(e)}")
            health = {
                'overall_status': 'unknown',
                'resilience_available': False,
                'timestamp': datetime.now().isoformat()
            }
        
        # Display overall status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = health.get('overall_status', 'unknown')
            status_emoji = "🟢" if status == 'healthy' else "🟡" if status == 'degraded' else "⚪"
            st.metric("Overall Status", f"{status_emoji} {status.upper()}")
        
        with col2:
            resilience = health.get('resilience_available', False)
            resilience_emoji = "✅" if resilience else "⚠️"
            st.metric("Resilience Features", f"{resilience_emoji} {'Active' if resilience else 'Inactive'}")
        
        with col3:
            timestamp = health.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                    st.metric("Last Check", time_str)
                except Exception:
                    st.metric("Last Check", "N/A")
            else:
                st.metric("Last Check", "N/A")
        
        # Circuit Breaker Status
        if health.get('circuit_breakers'):
            st.markdown("---")
            st.markdown("### Circuit Breaker Status")
            
            circuit_breakers = health['circuit_breakers']
            if isinstance(circuit_breakers, dict) and 'error' not in circuit_breakers:
                for name, state in circuit_breakers.items():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.text(name)
                    
                    with col2:
                        cb_state = state.get('state', 'unknown')
                        if cb_state == 'closed':
                            st.success("🟢 Closed")
                        elif cb_state == 'open':
                            st.error("🔴 Open")
                        elif cb_state == 'half_open':
                            st.warning("🟡 Half-Open")
                        else:
                            st.info("⚪ Unknown")
                    
                    with col3:
                        failures = state.get('failure_count', 0)
                        st.metric("Failures", failures, label_visibility="collapsed")
                    
                    with col4:
                        available = state.get('is_available', True)
                        st.text("✅" if available else "❌")
            else:
                st.info("No circuit breakers active or error loading status")
        
        # Cache Statistics
        if health.get('cache_stats'):
            st.markdown("---")
            st.markdown("### Cache Statistics")
            
            cache_stats = health['cache_stats']
            if isinstance(cache_stats, dict) and 'error' not in cache_stats:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total = cache_stats.get('total_entries', 0)
                    st.metric("Total Entries", total)
                
                with col2:
                    expired = cache_stats.get('expired_entries', 0)
                    st.metric("Expired Entries", expired)
                
                with col3:
                    size_bytes = cache_stats.get('total_size_bytes', 0)
                    size_kb = size_bytes / 1024
                    st.metric("Cache Size", f"{size_kb:.1f} KB")
                
                cache_dir = cache_stats.get('cache_dir', 'N/A')
                st.caption(f"Cache directory: `{cache_dir}`")
            else:
                st.info("Cache statistics unavailable")
        
        # Ticker Fetch Test
        st.markdown("---")
        st.markdown("### Ticker Fetch Test")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            test_ticker = st.text_input("Test Ticker Symbol", value="AAPL", key="health_test_ticker")
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("Test Fetch", key="health_test_button"):
                with st.spinner(f"Testing {test_ticker}..."):
                    result = test_ticker_fetch(test_ticker)
                    
                    if result['success']:
                        st.success(f"✅ Success! Latency: {result['latency_ms']}ms")
                        if result['data']:
                            price = result['data'].get('price')
                            change = result['data'].get('change_pct')
                            if price is not None:
                                st.info(f"Price: ${price:.2f}, Change: {change:+.2f}%")
                    else:
                        st.error(f"❌ Failed! Error: {result.get('error', 'Unknown error')}")
        
    except ImportError:
        st.warning("⚠️ Data health tracking not available - helper modules not found")
    except Exception as e:
        st.error(f"❌ Error displaying health panel: {str(e)}")


def render_compact_health_indicator():
    """
    Render a compact health indicator suitable for sidebar or header.
    """
    try:
        from helpers.ticker_sources import get_ticker_health_status
        
        health = get_ticker_health_status()
        status = health.get('overall_status', 'unknown')
        
        if status == 'healthy':
            st.success("🟢 System Healthy")
        elif status == 'degraded':
            st.warning("🟡 System Degraded")
        else:
            st.error("🔴 System Issues")
        
        # Show circuit breaker count
        circuit_breakers = health.get('circuit_breakers', {})
        if isinstance(circuit_breakers, dict) and 'error' not in circuit_breakers:
            open_count = sum(1 for state in circuit_breakers.values() 
                           if state.get('state') == 'open')
            if open_count > 0:
                st.caption(f"⚠️ {open_count} circuit breaker(s) open")
        
    except Exception:
        st.info("ℹ️ Health status unavailable")
