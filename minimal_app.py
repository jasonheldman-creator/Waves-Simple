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
WEIGHTS_FILE = "wave_weights.csv"
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
def load_expected_waves():
    """
    Load expected wave names from wave_weights.csv.
    Returns sorted list of unique wave names (should be 28 waves).
    
    Returns:
        list: Sorted list of expected wave names, or empty list if file not found
    """
    try:
        # Check both current directory and data subdirectory
        possible_paths = [
            WEIGHTS_FILE,
            os.path.join("data", WEIGHTS_FILE),
            os.path.join(os.getcwd(), WEIGHTS_FILE),
            os.path.join(os.getcwd(), "data", WEIGHTS_FILE)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'wave' in df.columns:
                    expected_waves = sorted(df['wave'].unique())
                    return expected_waves
        
        return []
    except Exception as e:
        st.warning(f"Error loading expected waves: {str(e)}")
        return []


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


def build_complete_snapshot(snapshot_df, expected_waves):
    """
    Build a complete snapshot ensuring all expected waves are present.
    For missing waves, populate with NaN returns and "NO DATA" status.
    
    Args:
        snapshot_df (pd.DataFrame): Loaded snapshot data (may be incomplete or None)
        expected_waves (list): List of expected wave names
        
    Returns:
        pd.DataFrame: Complete snapshot with all expected waves
    """
    # Create base dataframe with all expected waves
    complete_df = pd.DataFrame({
        'Wave': expected_waves,
        'Wave_ID': expected_waves,  # Use wave name as ID if not present
        'Status': ['NO DATA' for _ in expected_waves],
        'Missing_Tickers': ['' for _ in expected_waves],
        'Missing_Ticker_Count': [0 for _ in expected_waves]
    })
    
    # Add other expected columns with NaN
    for col in ['Return_1D', 'Return_30D', 'Return_60D', 'Return_365D',
                'Alpha_1D', 'Alpha_30D', 'Alpha_60D', 'Alpha_365D',
                'NAV', 'Coverage_Score', 'Beta_Real', 'Benchmark_Return_30D']:
        complete_df[col] = float('nan')
    
    complete_df['Date'] = None
    complete_df['Category'] = None
    complete_df['Mode'] = None
    complete_df['Flags'] = None
    
    # If no snapshot data, return complete_df with all NO DATA
    if snapshot_df is None or snapshot_df.empty:
        return complete_df
    
    # Determine the wave column name in snapshot
    wave_col = None
    if 'Wave' in snapshot_df.columns:
        wave_col = 'Wave'
    elif 'Wave_ID' in snapshot_df.columns:
        wave_col = 'Wave_ID'
    
    if not wave_col:
        # No wave column in snapshot, return complete_df with all NO DATA
        return complete_df
    
    # Update complete_df with data from snapshot where available
    for idx, row in complete_df.iterrows():
        wave_name = row['Wave']
        
        # Find matching row in snapshot
        snapshot_row = snapshot_df[snapshot_df[wave_col] == wave_name]
        
        if not snapshot_row.empty:
            # Wave exists in snapshot - copy all data
            snapshot_row = snapshot_row.iloc[0]
            
            # Copy all columns from snapshot
            for col in snapshot_df.columns:
                if col in complete_df.columns and col not in ['Wave', 'Wave_ID']:
                    complete_df.at[idx, col] = snapshot_row[col]
            
            # Set status to OK if we have NAV data
            if pd.notna(snapshot_row.get('NAV')):
                complete_df.at[idx, 'Status'] = 'OK'
    
    return complete_df


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
    
    # ========================================================================
    # A) EXECUTIVE OVERVIEW TABLE
    # ========================================================================
    st.subheader("Executive Overview")
    
    # Prepare the table with requested columns
    exec_columns = {
        'display_name': 'Wave',  # fallback to Wave_ID if Wave missing
        'status': 'Status',
        'return_1d': 'Return_1D',
        'return_30d': 'Return_30D',
        'return_60d': 'Return_60D',
        'return_365d': 'Return_365D',
        'alpha_1d': 'Alpha_1D',
        'alpha_30d': 'Alpha_30D',
        'alpha_60d': 'Alpha_60D',
        'alpha_365d': 'Alpha_365D',
        'beta': 'Beta_Real',
        'benchmark': 'Benchmark_Return_30D',  # using 30D benchmark as sample
        'coverage_pct': 'Coverage_Score',
        'missing_ticker_count': 'Missing_Ticker_Count',
        'stale_days_max': None  # not in current schema, will handle gracefully
    }
    
    exec_df = pd.DataFrame()
    
    # Build display_name column
    if 'Wave' in df.columns:
        exec_df['display_name'] = df['Wave']
    elif 'Wave_ID' in df.columns:
        exec_df['display_name'] = df['Wave_ID']
    else:
        exec_df['display_name'] = 'Unknown'
    
    # Add other columns if they exist
    for target_col, source_col in exec_columns.items():
        if target_col == 'display_name':
            continue  # already handled
        if source_col and source_col in df.columns:
            exec_df[target_col] = df[source_col]
        else:
            exec_df[target_col] = None  # gracefully handle missing columns
    
    # Sort: default by alpha_30d desc if present, else return_30d desc
    if 'alpha_30d' in exec_df.columns and exec_df['alpha_30d'].notna().any():
        exec_df = exec_df.sort_values('alpha_30d', ascending=False, na_position='last')
    elif 'return_30d' in exec_df.columns and exec_df['return_30d'].notna().any():
        exec_df = exec_df.sort_values('return_30d', ascending=False, na_position='last')
    
    # Format for display
    display_exec = exec_df.copy()
    
    # Format return/alpha columns as percentages with 2 decimals
    pct_cols_2dec = ['return_1d', 'return_30d', 'return_60d', 'return_365d',
                     'alpha_1d', 'alpha_30d', 'alpha_60d', 'alpha_365d', 'benchmark']
    for col in pct_cols_2dec:
        if col in display_exec.columns:
            display_exec[col] = display_exec[col].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
    
    # Format beta with 2 decimals
    if 'beta' in display_exec.columns:
        display_exec['beta'] = display_exec['beta'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
    
    # Format coverage_pct as 0-100 with 1 decimal
    if 'coverage_pct' in display_exec.columns:
        display_exec['coverage_pct'] = display_exec['coverage_pct'].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
    
    # Format stale_days_max as integer
    if 'stale_days_max' in display_exec.columns:
        display_exec['stale_days_max'] = display_exec['stale_days_max'].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
        )
    
    # Display the executive overview table
    st.dataframe(display_exec, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # B) LEADERBOARDS
    # ========================================================================
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Outperformers (30D Alpha)")
        if 'alpha_30d' in exec_df.columns and exec_df['alpha_30d'].notna().any():
            # Use unformatted exec_df for sorting
            top5 = exec_df.nlargest(5, 'alpha_30d')[['display_name', 'alpha_30d']].copy()
            # Format after sorting
            top5['alpha_30d'] = top5['alpha_30d'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
            st.dataframe(top5, use_container_width=True, hide_index=True)
        else:
            st.info("Alpha 30D data not available")
    
    with col2:
        st.subheader("⚠️ Needs Attention")
        if 'alpha_30d' in exec_df.columns and exec_df['alpha_30d'].notna().any():
            # Use unformatted exec_df for sorting
            # Show bottom 5 by alpha_30d
            bottom5 = exec_df.nsmallest(5, 'alpha_30d')[['display_name', 'alpha_30d']].copy()
            # Format after sorting
            bottom5['alpha_30d'] = bottom5['alpha_30d'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
            st.dataframe(bottom5, use_container_width=True, hide_index=True)
        else:
            # Show waves with missing data, low coverage, or high stale days
            needs_attention = []
            for idx, row in exec_df.iterrows():
                issues = []
                # Check coverage
                if 'coverage_pct' in exec_df.columns and pd.notna(row['coverage_pct']):
                    if row['coverage_pct'] < 100:
                        issues.append("Low Coverage")
                # Check stale days
                if 'stale_days_max' in exec_df.columns and pd.notna(row['stale_days_max']):
                    # Only compare if it's numeric
                    try:
                        if float(row['stale_days_max']) > 0:
                            issues.append("Stale Data")
                    except (ValueError, TypeError):
                        pass  # Skip if not numeric
                # Check for missing key columns
                if pd.isna(row.get('return_30d')) or pd.isna(row.get('return_60d')):
                    issues.append("Missing Returns")
                
                if issues:
                    needs_attention.append({
                        'display_name': row['display_name'],
                        'issue': ', '.join(issues)
                    })
            
            if needs_attention:
                attention_df = pd.DataFrame(needs_attention).head(5)
                st.dataframe(attention_df, use_container_width=True, hide_index=True)
            else:
                st.success("All waves look healthy!")
    
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
    
    # ========================================================================
    # NEW: Wave Data Diagnostics
    # ========================================================================
    st.subheader("📊 Wave Data Diagnostics")
    
    if df is not None and not df.empty:
        # Build diagnostics table
        diag_data = []
        for _, row in df.iterrows():
            wave_name = row.get('Wave', row.get('Wave_ID', 'Unknown'))
            status = row.get('Status', 'UNKNOWN')
            missing_tickers = row.get('Missing_Tickers', '')
            
            diag_data.append({
                'Wave': wave_name,
                'Status': status,
                'Missing Tickers': missing_tickers if missing_tickers else '-'
            })
        
        diag_df = pd.DataFrame(diag_data)
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
        
        # Summary metrics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        total_waves = len(df)
        waves_ok = len(df[df['Status'] == 'OK']) if 'Status' in df.columns else 0
        waves_no_data = len(df[df['Status'] == 'NO DATA']) if 'Status' in df.columns else 0
        
        with col1:
            st.metric("Total Expected Waves", total_waves)
        
        with col2:
            st.metric("Waves with OK Data", waves_ok)
        
        with col3:
            st.metric("Waves with NO DATA", waves_no_data)
    else:
        st.warning("No wave data available for diagnostics.")
    
    st.markdown("---")
    
    # Data freshness section
    st.subheader("📅 Data Freshness")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if df is not None and 'Date' in df.columns:
            try:
                # Filter out None/NaN values before finding max
                valid_dates = df['Date'].dropna()
                if not valid_dates.empty:
                    latest_date = valid_dates.max()
                    st.metric("Latest Data Date", latest_date)
                else:
                    st.metric("Latest Data Date", "N/A")
            except (TypeError, ValueError):
                st.metric("Latest Data Date", "N/A")
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
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Rebuild Live Snapshot button
        st.header("🔨 Data Rebuild")
        
        if st.button("🔨 Rebuild Live Snapshot Now", use_container_width=True):
            try:
                with st.spinner("Rebuilding live snapshot from market data..."):
                    # Import the snapshot generator
                    from analytics_truth import generate_live_snapshot_csv
                    
                    # Generate new snapshot
                    snapshot_df = generate_live_snapshot_csv()
                    
                    # Get metadata
                    total_waves = len(snapshot_df)
                    waves_with_data = (snapshot_df['status'] == 'OK').sum()
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Clear cache to force reload
                    st.cache_data.clear()
                    
                    # Show success message
                    st.success(f"""
                    ✅ **Snapshot Rebuilt Successfully!**
                    
                    - **Timestamp:** {timestamp}
                    - **Total Waves:** {total_waves}
                    - **Waves with Data:** {waves_with_data}
                    - **Waves with NO DATA:** {total_waves - waves_with_data}
                    """)
                    
                    # Rerun to show new data
                    st.rerun()
                    
            except Exception as e:
                st.error(f"⚠️ **Rebuild Failed:** {str(e)}")
        
        # Optional: Timed rebuild with cooldown (circuit breaker aware)
        if 'last_snapshot_build_ts' not in st.session_state:
            st.session_state.last_snapshot_build_ts = 0
        
        # Show time since last rebuild
        time_since_last = datetime.now().timestamp() - st.session_state.last_snapshot_build_ts
        if st.session_state.last_snapshot_build_ts > 0:
            minutes_ago = int(time_since_last / 60)
            st.caption(f"Last rebuild: {minutes_ago} min ago")
        
        # Auto-rebuild option (only if circuit breaker is closed)
        if not is_circuit_open():
            auto_rebuild = st.checkbox(
                "Enable Auto-Rebuild (every 5 min)",
                value=False,
                help="Automatically rebuild snapshot every 5 minutes. Only active when circuit breaker is closed."
            )
            
            # Trigger rebuild if enabled and cooldown period has passed
            if auto_rebuild and time_since_last >= 300:  # 300 seconds = 5 minutes
                try:
                    from analytics_truth import generate_live_snapshot_csv
                    
                    # Generate new snapshot silently
                    snapshot_df = generate_live_snapshot_csv()
                    st.session_state.last_snapshot_build_ts = datetime.now().timestamp()
                    
                    # Clear cache
                    st.cache_data.clear()
                    
                except Exception as e:
                    print(f"Auto-rebuild failed: {e}")
        
        st.markdown("---")
        
        # Show last update time
        st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    # SINGLE auto-refresh call - only active when enabled and circuit is closed
    # This is the ONLY st_autorefresh call in the entire file
    if auto_refresh_enabled and not is_circuit_open():
        st_autorefresh(interval=MIN_REFRESH_INTERVAL, key="minimal_console_refresh")
    
    # Load data
    try:
        # Load expected waves from weights CSV
        expected_waves = load_expected_waves()
        
        # Load snapshot data (may be incomplete or missing)
        snapshot_df = load_snapshot_data()
        
        # Build complete snapshot ensuring all expected waves are present
        df = build_complete_snapshot(snapshot_df, expected_waves)
        
        if df is None or df.empty:
            st.error(f"""
            ⚠️ **No Wave Data Available**
            
            Expected waves could not be loaded from `{WEIGHTS_FILE}`.
            Snapshot file `{DATA_FILE}` may also be missing.
            
            Expected locations:
            - Current directory: `{os.getcwd()}`
            - Data subdirectory: `{os.path.join(os.getcwd(), 'data')}`
            
            Please ensure both files exist before running this application.
            """)
            record_error()
            st.stop()  # Halt execution - no infinite loops
        
        # Data loaded successfully
        record_success()
        
        # ========================================================================
        # C) MARKET SNAPSHOT STRIP (above tabs)
        # ========================================================================
        # Find the file path that was used
        data_file_path = None
        possible_paths = [
            DATA_FILE,
            os.path.join("data", DATA_FILE),
            os.path.join(os.getcwd(), DATA_FILE),
            os.path.join(os.getcwd(), "data", DATA_FILE)
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_file_path = path
                break
        
        # Get data timestamp - try multiple column names
        data_timestamp = None
        timestamp_cols = ['asof', 'timestamp', 'last_updated', 'Date']
        for col in timestamp_cols:
            if col in df.columns and df[col].notna().any():
                try:
                    # Filter out None/NaN values before finding max
                    valid_dates = df[col].dropna()
                    if not valid_dates.empty:
                        data_timestamp = valid_dates.max()
                        break
                except (TypeError, ValueError):
                    # Skip if column has mixed types or non-comparable values
                    continue
        
        # Display market snapshot strip
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_waves = len(df)
            st.metric("📊 Waves Loaded", num_waves)
        
        with col2:
            if data_timestamp:
                st.metric("📅 Data As Of", data_timestamp)
            else:
                st.metric("📅 Data As Of", "N/A")
        
        with col3:
            if data_file_path:
                st.metric("📁 File Path", data_file_path)
            else:
                st.metric("📁 File Path", "Unknown")
        
        st.markdown("---")
        
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
