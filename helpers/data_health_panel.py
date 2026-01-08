"""
Data Health Panel
Provides visibility into system health, cache performance, and data availability.
Enhanced with failed ticker diagnostics and reporting.

NOTE: This module requires Streamlit to be installed. Import streamlit is lazy-loaded
      to allow other helpers modules to be imported without Streamlit.
"""

from datetime import datetime
from typing import Dict, Any
import os


def render_data_health_panel():
    """
    Render the Data Health panel showing system status and metrics.
    Enhanced with fail-safe error handling and ticker diagnostics.
    """
    # Lazy import of streamlit - only imported when this function is called
    try:
        import streamlit as st
    except ImportError:
        raise RuntimeError(
            "Streamlit is required for this function. "
            "Please install it to use Streamlit-related functionality: pip install streamlit"
        )
    
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
            # Show active wave ticker count if available, otherwise show resilience status
            active_ticker_count = health.get('active_wave_ticker_count', 0)
            if active_ticker_count > 0:
                st.metric("Active Wave Tickers", active_ticker_count)
            else:
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
        
        # Display note about active wave filtering
        if health.get('active_wave_ticker_count', 0) > 0:
            st.caption("ℹ️ System health monitored based on tickers from active waves only")
        
        # ========================================================================
        # PRICE_BOOK PROOF Section
        # ========================================================================
        st.markdown("---")
        st.markdown("### 📈 PRICE_BOOK PROOF")
        st.caption("Verification that price cache is loaded and usable")
        
        try:
            # Import price_loader functions
            from helpers.price_loader import (
                get_price_book_debug_summary, 
                CACHE_PATH,
                load_cache
            )
            import pandas as pd
            
            # Load price_book
            try:
                price_book = load_cache()
            except Exception as e:
                st.error(f"❌ Failed to load price_book: {str(e)}")
                price_book = None
            
            # Get debug summary
            debug_summary = get_price_book_debug_summary(price_book)
            
            # Check if file exists
            file_exists = os.path.exists(CACHE_PATH)
            file_size_kb = 0
            file_size_mb = 0
            if file_exists:
                try:
                    file_size_bytes = os.path.getsize(CACHE_PATH)
                    file_size_kb = file_size_bytes / 1024
                    file_size_mb = file_size_bytes / (1024 * 1024)
                except Exception as e:
                    st.warning(f"⚠️ Could not get file size: {str(e)}")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows (Days)", debug_summary['rows'])
            
            with col2:
                st.metric("Cols (Tickers)", debug_summary['cols'])
            
            with col3:
                st.metric("Non-null Cells", f"{debug_summary['non_null_cells']:,}")
            
            with col4:
                file_status = "✅ Exists" if file_exists else "❌ Missing"
                st.metric("Cache File", file_status)
            
            # Display date range
            if debug_summary['start_date'] and debug_summary['end_date']:
                st.caption(f"📅 Date Range: {debug_summary['start_date']} to {debug_summary['end_date']}")
            else:
                st.caption("📅 Date Range: N/A")
            
            # Display file size
            if file_exists:
                if file_size_mb >= 1:
                    st.caption(f"💾 File Size: {file_size_mb:.2f} MB")
                else:
                    st.caption(f"💾 File Size: {file_size_kb:.2f} KB")
                st.caption(f"📂 Path: `{CACHE_PATH}`")
            else:
                st.caption(f"📂 Path: `{CACHE_PATH}` (missing)")
            
            # Display sample tickers (first 10)
            if debug_summary['sample_tickers']:
                ticker_list = ', '.join(debug_summary['sample_tickers'])
                st.caption(f"🎯 Sample Tickers (first 10): {ticker_list}")
            
            # Show warnings/errors
            is_empty = debug_summary['is_empty']
            is_stale = False
            
            # Check if end_date is stale (>7 days old)
            if debug_summary['end_date']:
                try:
                    end_date_obj = datetime.strptime(debug_summary['end_date'], '%Y-%m-%d')
                    days_old = (datetime.now() - end_date_obj).days
                    if days_old > 7:
                        is_stale = True
                except Exception:
                    pass
            
            # Display error/warning banner if needed
            if not file_exists or is_empty:
                st.error("⚠️ PRICE_BOOK EMPTY – portfolio + alphas will be N/A. Check prices_cache.parquet in repo and load path.")
            elif is_stale:
                st.warning(f"⚠️ PRICE_BOOK DATA STALE – Last updated: {debug_summary['end_date']} ({days_old} days old)")
            else:
                st.success("✅ PRICE_BOOK loaded successfully and up-to-date")
        
        except ImportError as e:
            st.warning(f"⚠️ PRICE_BOOK verification not available - import error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error verifying PRICE_BOOK: {str(e)}")
        
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
        
        # Failed Ticker Diagnostics
        try:
            from helpers.ticker_diagnostics import get_diagnostics_tracker
            
            st.markdown("---")
            st.markdown("### Failed Ticker Diagnostics")
            
            tracker = get_diagnostics_tracker()
            stats = tracker.get_summary_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Failures", stats['total_failures'])
            
            with col2:
                st.metric("Unique Tickers", stats['unique_tickers'])
            
            with col3:
                st.metric("Fatal", stats['fatal_count'])
            
            with col4:
                st.metric("Non-Fatal", stats['non_fatal_count'])
            
            if stats['total_failures'] > 0:
                # Show breakdown by failure type
                st.markdown("#### Failure Type Breakdown")
                for failure_type, count in stats['by_type'].items():
                    st.text(f"• {failure_type}: {count}")
                
                # Export button
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Export Failed Tickers Report", key="export_failed_tickers"):
                        try:
                            filepath = tracker.export_to_csv()
                            st.success(f"✅ Report exported to: `{filepath}`")
                            
                            # Provide download option if file exists
                            if os.path.exists(filepath):
                                with open(filepath, 'r') as f:
                                    csv_data = f.read()
                                st.download_button(
                                    label="Download Report",
                                    data=csv_data,
                                    file_name=os.path.basename(filepath),
                                    mime="text/csv",
                                    key="download_failed_tickers"
                                )
                        except Exception as e:
                            st.error(f"❌ Failed to export report: {str(e)}")
                
                # Show recent failures
                st.markdown("#### Recent Failures (Last 10)")
                failures = tracker.get_all_failures()
                recent_failures = sorted(failures, key=lambda x: x.last_seen or datetime.min, reverse=True)[:10]
                
                if recent_failures:
                    for failure in recent_failures:
                        with st.expander(f"🔴 {failure.ticker_original} ({failure.failure_type.value})"):
                            st.text(f"Wave: {failure.wave_name or 'N/A'}")
                            st.text(f"Normalized: {failure.ticker_normalized}")
                            st.text(f"Source: {failure.source}")
                            st.text(f"Fatal: {'Yes' if failure.is_fatal else 'No'}")
                            st.text(f"Error: {failure.error_message}")
                            st.text(f"Suggested Fix: {failure.suggested_fix}")
                            if failure.last_seen:
                                st.caption(f"Last seen: {failure.last_seen.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info("✅ No ticker failures recorded")
        
        except ImportError:
            st.info("ℹ️ Ticker diagnostics not available - module not imported")
        except Exception as e:
            st.warning(f"⚠️ Unable to load ticker diagnostics: {str(e)}")
        
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
    # Lazy import of streamlit - only imported when this function is called
    try:
        import streamlit as st
    except ImportError:
        raise RuntimeError(
            "Streamlit is required for this function. "
            "Please install it to use Streamlit-related functionality: pip install streamlit"
        )
    
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


def render_data_readiness_panel():
    """
    Render data readiness metrics panel showing wave operational status.
    """
    # Lazy import of streamlit - only imported when this function is called
    try:
        import streamlit as st
    except ImportError:
        raise RuntimeError(
            "Streamlit is required for this function. "
            "Please install it to use Streamlit-related functionality: pip install streamlit"
        )
    
    st.subheader("🌊 Data Readiness")
    st.caption("Historical price data coverage and wave operational status")
    
    try:
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Check if prices.csv exists
        prices_path = 'data/prices.csv'
        if not os.path.exists(prices_path):
            st.warning("⚠️ No price data found. Run: `python scripts/enable_full_data.py`")
            return
        
        # Load prices data with error handling
        try:
            df = pd.read_csv(prices_path)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert invalid dates to NaT
        except Exception as e:
            st.error(f"❌ Error loading price data: {str(e)}")
            return
        
        # Load universe
        universe_path = 'universal_universe.csv'
        if not os.path.exists(universe_path):
            st.error("❌ Universe file not found")
            return
        
        universe = pd.read_csv(universe_path)
        universe = universe[universe['status'] == 'active']
        all_tickers = universe['ticker'].unique().tolist()
        
        # Analyze coverage
        tickers_in_prices = df['ticker'].unique().tolist()
        missing_tickers = sorted(list(set(all_tickers) - set(tickers_in_prices)))
        
        coverage_pct = (len(tickers_in_prices) / len(all_tickers) * 100) if all_tickers else 0
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tickers", len(all_tickers))
        
        with col2:
            st.metric("With Data", len(tickers_in_prices))
        
        with col3:
            status_emoji = "🟢" if coverage_pct >= 95 else "🟡" if coverage_pct >= 80 else "🔴"
            st.metric("Coverage", f"{status_emoji} {coverage_pct:.1f}%")
        
        with col4:
            days = (df['date'].max() - df['date'].min()).days if not df.empty else 0
            st.metric("Days of History", days)
        
        # Operational status
        st.markdown("---")
        if coverage_pct >= 95:
            st.success("🟢 **STATUS: FULLY OPERATIONAL** - All waves ready")
        elif coverage_pct >= 80:
            st.warning("🟡 **STATUS: MOSTLY OPERATIONAL** - Some data missing")
        elif coverage_pct >= 50:
            st.warning("🟠 **STATUS: PARTIALLY OPERATIONAL** - Significant gaps")
        else:
            st.error("🔴 **STATUS: LIMITED OPERATIONAL** - Major data gaps")
        
        # Wave-level analysis
        if 'index_membership' in universe.columns:
            st.markdown("### Wave Operational Status")
            
            # Extract unique wave names
            waves = set()
            for idx_mem in universe['index_membership'].dropna():
                for wave in str(idx_mem).split(','):
                    wave = wave.strip()
                    if wave.startswith('WAVE_'):
                        waves.add(wave)
            
            waves = sorted(list(waves))
            
            if waves:
                wave_readiness = []
                for wave in waves:
                    wave_tickers = universe[
                        universe['index_membership'].str.contains(wave, case=False, na=False)
                    ]['ticker'].unique().tolist()
                    
                    wave_tickers_with_data = [t for t in wave_tickers if t in tickers_in_prices]
                    wave_coverage = (len(wave_tickers_with_data) / len(wave_tickers) * 100) if wave_tickers else 0
                    
                    wave_readiness.append({
                        'wave': wave.replace('WAVE_', '').replace('_', ' ').title(),
                        'total': len(wave_tickers),
                        'with_data': len(wave_tickers_with_data),
                        'coverage': wave_coverage
                    })
                
                # Count operational waves
                operational_count = sum(1 for w in wave_readiness if w['coverage'] == 100)
                total_waves = len(wave_readiness)
                
                st.info(f"📊 **{operational_count}/{total_waves} waves fully operational**")
                
                # Show incomplete waves in expander
                incomplete = [w for w in wave_readiness if w['coverage'] < 100]
                if incomplete:
                    with st.expander(f"⚠️ Waves with incomplete data ({len(incomplete)})"):
                        for w in incomplete:
                            status = "🟢" if w['coverage'] >= 80 else "🟡" if w['coverage'] >= 50 else "🔴"
                            st.text(f"{status} {w['wave']}: {w['with_data']}/{w['total']} ({w['coverage']:.0f}%)")
                
                # Show operational waves
                complete = [w for w in wave_readiness if w['coverage'] == 100]
                if complete:
                    with st.expander(f"✅ Fully operational waves ({len(complete)})"):
                        for w in complete:
                            st.text(f"🟢 {w['wave']}: {w['total']}/{w['total']} (100%)")
        
        # Missing tickers section
        if missing_tickers:
            st.markdown("---")
            with st.expander(f"❌ Missing tickers ({len(missing_tickers)})"):
                st.text(", ".join(missing_tickers[:50]))
                if len(missing_tickers) > 50:
                    st.caption(f"... and {len(missing_tickers) - 50} more")
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data", key="refresh_data_btn"):
                st.info("Run: `python scripts/enable_full_data.py` to fetch latest data")
        
        with col2:
            if st.button("📊 Full Analysis", key="full_analysis_btn"):
                st.info("Run: `python scripts/analyze_data_readiness.py` for detailed report")
    
    except Exception as e:
        st.error(f"❌ Error displaying data readiness: {str(e)}")
