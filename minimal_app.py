"""
WAVES Intelligence™ — Minimal Console v1
=========================================

A minimal, focused Streamlit application for viewing WAVES portfolio metrics.

Features:
- Three-tab layout: Overview, Single Wave, Diagnostics
- Safe auto-refresh with circuit breaker (default OFF, 30s minimum)
- CSV data loading with caching (60s TTL)
- No infinite reruns or runaway behavior
- Concise UI focused on essential metrics

Requirements:
- Reads from live_snapshot.csv in runtime directory
- Displays Returns: 1D, 30D, 60D, 365D
- Displays Alpha vs benchmark: 1D, 30D, 60D, 365D
- Data freshness monitoring
- Error log excerpt display
"""

import streamlit as st
import pandas as pd
import os
import traceback
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ============================================================================
# CONFIGURATION
# ============================================================================

PAGE_TITLE = "WAVES Intelligence™ — Minimal Console v1"
DATA_FILE = "live_snapshot.csv"
CACHE_TTL = 60  # seconds
MIN_REFRESH_INTERVAL = 30000  # milliseconds (30 seconds)
MAX_CONSECUTIVE_ERRORS = 3

# ============================================================================
# DATA LOADING
# ============================================================================

def format_percentage(value):
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Numeric value (e.g., 0.05 for 5%)
        
    Returns:
        str: Formatted percentage string (e.g., "5.00%") or "N/A"
    """
    if pd.notna(value):
        return f"{value*100:.2f}%"
    return "N/A"


@st.cache_data(ttl=CACHE_TTL)
def load_snapshot_data():
    """
    Load live snapshot data from CSV file.
    Uses caching with 60-second TTL to avoid excessive file reads.
    
    Returns:
        pd.DataFrame: Snapshot data, or None if file not found
    """
    try:
        # Check both current directory and data subdirectory
        possible_paths = [
            DATA_FILE,
            os.path.join("data", DATA_FILE),
            os.path.join(os.getcwd(), DATA_FILE),
            os.path.join(os.getcwd(), "data", DATA_FILE)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
        
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def get_wave_metrics(df, wave_id=None):
    """
    Extract metrics for a specific wave or all waves.
    
    Args:
        df (pd.DataFrame): Snapshot data
        wave_id (str, optional): Specific wave ID to filter
        
    Returns:
        pd.DataFrame: Metrics data
    """
    if df is None:
        return None
    
    # Define columns of interest
    metric_columns = [
        'Wave_ID', 'Wave', 'Category', 'Mode',
        'Return_1D', 'Return_30D', 'Return_60D', 'Return_365D',
        'Alpha_1D', 'Alpha_30D', 'Alpha_60D', 'Alpha_365D',
        'NAV', 'Date', 'Coverage_Score', 'Flags'
    ]
    
    # Filter to available columns
    available_columns = [col for col in metric_columns if col in df.columns]
    
    if wave_id:
        wave_data = df[df['Wave_ID'] == wave_id]
        if not wave_data.empty:
            return wave_data[available_columns]
        return None
    
    return df[available_columns]


# ============================================================================
# CIRCUIT BREAKER FOR AUTO-REFRESH
# ============================================================================

def initialize_circuit_breaker():
    """Initialize circuit breaker state in session."""
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    if 'circuit_open' not in st.session_state:
        st.session_state.circuit_open = False
    if 'last_error_time' not in st.session_state:
        st.session_state.last_error_time = None


def record_error():
    """Record an error occurrence for circuit breaker."""
    st.session_state.error_count += 1
    st.session_state.last_error_time = datetime.now()
    
    if st.session_state.error_count >= MAX_CONSECUTIVE_ERRORS:
        st.session_state.circuit_open = True


def record_success():
    """Record successful operation, resetting error count."""
    st.session_state.error_count = 0


def is_circuit_open():
    """Check if circuit breaker is open (halting refresh)."""
    return st.session_state.get('circuit_open', False)


# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_overview_tab(df):
    """
    Render Overview tab showing all Waves with returns and alpha metrics.
    
    Args:
        df (pd.DataFrame): Snapshot data
    """
    st.header("📊 All Waves Overview")
    
    if df is None or df.empty:
        st.warning("No data available for overview.")
        return
    
    # Get metrics for all waves
    metrics = get_wave_metrics(df)
    
    if metrics is None or metrics.empty:
        st.warning("No metrics available.")
        return
    
    # Format numeric columns as percentages
    display_df = metrics.copy()
    
    percentage_columns = [
        'Return_1D', 'Return_30D', 'Return_60D', 'Return_365D',
        'Alpha_1D', 'Alpha_30D', 'Alpha_60D', 'Alpha_365D'
    ]
    
    for col in percentage_columns:
        if col in display_df.columns:
            # Vectorized percentage formatting
            display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
            display_df[col] = display_df[col].replace('nan%', 'N/A')
    
    # Display the table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Summary metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_waves = len(df)
        st.metric("Total Waves", total_waves)
    
    with col2:
        waves_with_data = len(df[df['NAV'].notna()])
        st.metric("Waves with Data", waves_with_data)
    
    with col3:
        coverage_avg = df['Coverage_Score'].mean() if 'Coverage_Score' in df.columns else 0
        st.metric("Avg Coverage Score", f"{coverage_avg:.0f}%")


def render_single_wave_tab(df):
    """
    Render Single Wave tab with dropdown selection and detailed metrics.
    
    Args:
        df (pd.DataFrame): Snapshot data
    """
    st.header("🔍 Single Wave Analysis")
    
    if df is None or df.empty:
        st.warning("No data available for wave analysis.")
        return
    
    # Wave selection dropdown
    wave_options = df[['Wave_ID', 'Wave']].dropna()
    wave_list = wave_options['Wave_ID'].tolist()
    
    if not wave_list:
        st.warning("No waves available for selection.")
        return
    
    selected_wave_id = st.selectbox(
        "Select a Wave:",
        options=wave_list,
        format_func=lambda x: df[df['Wave_ID'] == x]['Wave'].iloc[0] if not df[df['Wave_ID'] == x].empty else x
    )
    
    if not selected_wave_id:
        return
    
    # Get metrics for selected wave
    wave_data = get_wave_metrics(df, selected_wave_id)
    
    if wave_data is None or wave_data.empty:
        st.warning(f"No data available for {selected_wave_id}")
        return
    
    # Display wave information
    wave_info = wave_data.iloc[0]
    
    st.subheader(wave_info.get('Wave', selected_wave_id))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Category", wave_info.get('Category', 'N/A'))
    
    with col2:
        st.metric("Mode", wave_info.get('Mode', 'N/A'))
    
    with col3:
        nav = wave_info.get('NAV', None)
        st.metric("NAV", f"{nav:.2f}" if pd.notna(nav) else "N/A")
    
    with col4:
        coverage = wave_info.get('Coverage_Score', None)
        st.metric("Coverage", f"{coverage:.0f}%" if pd.notna(coverage) else "N/A")
    
    # Returns section
    st.markdown("---")
    st.subheader("📈 Returns")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ret_1d = wave_info.get('Return_1D', None)
        st.metric("1D", format_percentage(ret_1d))
    
    with col2:
        ret_30d = wave_info.get('Return_30D', None)
        st.metric("30D", format_percentage(ret_30d))
    
    with col3:
        ret_60d = wave_info.get('Return_60D', None)
        st.metric("60D", format_percentage(ret_60d))
    
    with col4:
        ret_365d = wave_info.get('Return_365D', None)
        st.metric("365D", format_percentage(ret_365d))
    
    # Alpha section
    st.markdown("---")
    st.subheader("🎯 Alpha vs Benchmark")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        alpha_1d = wave_info.get('Alpha_1D', None)
        st.metric("1D", format_percentage(alpha_1d))
    
    with col2:
        alpha_30d = wave_info.get('Alpha_30D', None)
        st.metric("30D", format_percentage(alpha_30d))
    
    with col3:
        alpha_60d = wave_info.get('Alpha_60D', None)
        st.metric("60D", format_percentage(alpha_60d))
    
    with col4:
        alpha_365d = wave_info.get('Alpha_365D', None)
        st.metric("365D", format_percentage(alpha_365d))
    
    # Additional information
    if 'Flags' in wave_info and pd.notna(wave_info['Flags']):
        st.markdown("---")
        st.info(f"**Flags:** {wave_info['Flags']}")


def render_diagnostics_tab(df):
    """
    Render Diagnostics tab with data freshness, missing tickers, and error logs.
    
    Args:
        df (pd.DataFrame): Snapshot data
    """
    st.header("🔧 Diagnostics")
    
    # Data freshness section
    st.subheader("📅 Data Freshness")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if df is not None and 'Date' in df.columns:
            latest_date = df['Date'].max()
            st.metric("Latest Data Date", latest_date)
        else:
            st.metric("Latest Data Date", "N/A")
    
    with col2:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.metric("Current Time", current_time)
    
    with col3:
        if df is not None:
            file_time = "Available"
        else:
            file_time = "Missing"
        st.metric("Snapshot File", file_time)
    
    # Missing tickers section
    st.markdown("---")
    st.subheader("⚠️ Missing Tickers")
    
    missing_tickers_file = "missing_tickers.csv"
    missing_paths = [
        missing_tickers_file,
        os.path.join(os.getcwd(), missing_tickers_file)
    ]
    
    missing_found = False
    for path in missing_paths:
        if os.path.exists(path):
            try:
                missing_df = pd.read_csv(path)
                if not missing_df.empty:
                    st.dataframe(missing_df.head(10), use_container_width=True)
                    st.caption(f"Showing first 10 of {len(missing_df)} missing tickers")
                else:
                    st.success("No missing tickers recorded.")
                missing_found = True
                break
            except Exception as e:
                st.warning(f"Error reading missing tickers: {str(e)}")
                break
    
    if not missing_found:
        st.info("Missing tickers file not found.")
    
    # Error log section
    st.markdown("---")
    st.subheader("📋 Error Log Excerpt")
    
    log_dir = "logs"
    error_log_path = os.path.join(log_dir, "error.log")
    
    if os.path.exists(error_log_path):
        try:
            with open(error_log_path, 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-20:] if len(lines) > 20 else lines
                
            if recent_lines:
                st.text_area(
                    "Recent Errors (last 20 lines)",
                    value=''.join(recent_lines),
                    height=200
                )
            else:
                st.success("No errors in log file.")
        except Exception as e:
            st.warning(f"Error reading log file: {str(e)}")
    else:
        st.info("Error log file not found.")
    
    # Circuit breaker status
    st.markdown("---")
    st.subheader("🔒 Circuit Breaker Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        error_count = st.session_state.get('error_count', 0)
        st.metric("Error Count", f"{error_count}/{MAX_CONSECUTIVE_ERRORS}")
    
    with col2:
        circuit_status = "🔴 OPEN (Halted)" if is_circuit_open() else "🟢 CLOSED (Active)"
        st.metric("Circuit Status", circuit_status)
    
    if is_circuit_open():
        st.error("⚠️ Circuit breaker is OPEN. Auto-refresh has been halted due to consecutive errors.")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize circuit breaker
    initialize_circuit_breaker()
    
    # Title
    st.title(PAGE_TITLE)
    
    # Sidebar - Auto-refresh control
    with st.sidebar:
        st.header("🔄 Auto-Refresh Control")
        
        # Auto-refresh toggle (default OFF)
        auto_refresh_enabled = st.toggle(
            "Enable Auto-Refresh",
            value=False,
            help="Enable automatic page refresh. Default is OFF to prevent runaway behavior."
        )
        
        if auto_refresh_enabled:
            st.warning("⚠️ Auto-refresh is ENABLED")
            
            # Show circuit breaker status
            if is_circuit_open():
                st.error("🔴 Circuit breaker OPEN - refresh halted")
                auto_refresh_enabled = False
            else:
                st.success("🟢 Circuit breaker CLOSED")
        else:
            st.info("🔴 Auto-refresh is DISABLED")
        
        st.markdown("---")
        
        # Show last update time
        st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    # SINGLE auto-refresh call - only active when enabled and circuit is closed
    # This is the ONLY st_autorefresh call in the entire file
    if auto_refresh_enabled and not is_circuit_open():
        st_autorefresh(interval=MIN_REFRESH_INTERVAL, key="minimal_console_refresh")
    
    # Load data
    try:
        df = load_snapshot_data()
        
        if df is None:
            st.error(f"""
            ⚠️ **Missing File: {DATA_FILE}**
            
            The required data file `{DATA_FILE}` was not found in the runtime directory.
            
            Expected locations:
            - Current directory: `{os.getcwd()}`
            - Data subdirectory: `{os.path.join(os.getcwd(), 'data')}`
            
            Please ensure the file exists before running this application.
            """)
            record_error()
            st.stop()  # Halt execution - no infinite loops
        
        # Data loaded successfully
        record_success()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Single Wave", "🔧 Diagnostics"])
        
        with tab1:
            render_overview_tab(df)
        
        with tab2:
            render_single_wave_tab(df)
        
        with tab3:
            render_diagnostics_tab(df)
    
    except Exception as e:
        st.error(f"⚠️ **Application Error:** {str(e)}")
        record_error()
        
        # Show error details in expander
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        
        # Don't continue if there's an error
        st.stop()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
