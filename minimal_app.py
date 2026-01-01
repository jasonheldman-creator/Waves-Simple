"""
WAVES Intelligence™ — Minimal Console v1
=========================================

A lightweight, minimal Streamlit application for WAVES Intelligence data visualization.

Features:
- Three tabs: Overview, Single Wave, Diagnostics
- Data source: live_snapshot.csv with caching (60-second TTL)
- Optional auto-refresh with circuit breaker protection
- Minimal dependencies and fast load times

Design Principles:
- Auto-refresh OFF by default
- Safe refresh intervals (30-second minimum)
- Circuit breaker stops after 3 consecutive errors
- Clear error messaging for missing data
- No infinite loops or runaway behavior
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Import streamlit-autorefresh conditionally
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = "live_snapshot.csv"
CACHE_TTL_SECONDS = 60
MIN_REFRESH_INTERVAL_MS = 30000  # 30 seconds minimum
DEFAULT_REFRESH_INTERVAL_MS = 60000  # 1 minute default
MAX_CONSECUTIVE_ERRORS = 3


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0
    if "circuit_breaker_active" not in st.session_state:
        st.session_state.circuit_breaker_active = False
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = False


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_snapshot_data():
    """
    Load live_snapshot.csv with caching.
    
    Returns:
        pd.DataFrame or None: Loaded data or None if file not found
    """
    # Check in current directory first
    if os.path.exists(DATA_FILE):
        file_path = DATA_FILE
    # Check in data subdirectory
    elif os.path.exists(f"data/{DATA_FILE}"):
        file_path = f"data/{DATA_FILE}"
    else:
        return None
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading {DATA_FILE}: {str(e)}")
        return None


# ============================================================================
# ERROR HANDLING
# ============================================================================

def handle_error(error_message):
    """
    Handle errors with circuit breaker logic.
    
    Args:
        error_message (str): Error message to log
    """
    st.session_state.error_count += 1
    st.session_state.last_error = error_message
    
    if st.session_state.error_count >= MAX_CONSECUTIVE_ERRORS:
        st.session_state.circuit_breaker_active = True
        st.session_state.auto_refresh_enabled = False


def reset_error_state():
    """Reset error counters on successful operation."""
    st.session_state.error_count = 0
    st.session_state.circuit_breaker_active = False
    st.session_state.last_error = None


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render sidebar with auto-refresh controls and status."""
    with st.sidebar:
        st.header("Controls")
        
        # Auto-refresh toggle
        if AUTOREFRESH_AVAILABLE:
            st.subheader("Auto-Refresh")
            
            # Circuit breaker warning
            if st.session_state.circuit_breaker_active:
                st.error("⚠️ Circuit Breaker Active")
                st.warning(f"Auto-refresh paused after {MAX_CONSECUTIVE_ERRORS} consecutive errors.")
                if st.session_state.last_error:
                    st.caption(f"Last error: {st.session_state.last_error}")
                
                if st.button("Reset Circuit Breaker"):
                    reset_error_state()
                    st.rerun()
            else:
                # Auto-refresh toggle
                auto_refresh = st.checkbox(
                    "Enable Auto-Refresh",
                    value=st.session_state.auto_refresh_enabled,
                    help="Automatically refresh data at regular intervals (OFF by default)"
                )
                
                if auto_refresh != st.session_state.auto_refresh_enabled:
                    st.session_state.auto_refresh_enabled = auto_refresh
                    st.rerun()
                
                if st.session_state.auto_refresh_enabled:
                    st.success("🟢 Auto-Refresh: ON")
                    st.caption(f"Refresh interval: {DEFAULT_REFRESH_INTERVAL_MS / 1000:.0f} seconds")
                else:
                    st.info("🔴 Auto-Refresh: OFF")
        else:
            st.info("Auto-refresh not available (streamlit-autorefresh not installed)")
        
        st.markdown("---")
        
        # Status information
        st.subheader("Status")
        
        # Error counter
        if st.session_state.error_count > 0:
            st.metric(
                "Error Count",
                st.session_state.error_count,
                delta=f"{MAX_CONSECUTIVE_ERRORS - st.session_state.error_count} remaining",
                delta_color="inverse"
            )
        else:
            st.metric("Error Count", 0, delta="✓ Healthy")
        
        # Last update timestamp
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def render_overview_tab(df):
    """
    Render the Overview tab with all waves and their metrics.
    
    Args:
        df (pd.DataFrame): Snapshot data
    """
    st.header("Overview")
    st.markdown("All waves with returns and alpha metrics")
    
    # Select relevant columns
    display_columns = [
        "Wave",
        "Return_1D", "Return_30D", "Return_60D", "Return_365D",
        "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D"
    ]
    
    # Filter to only existing columns
    available_columns = [col for col in display_columns if col in df.columns]
    
    if len(available_columns) < 2:
        st.warning("Insufficient data columns available in snapshot.")
        return
    
    # Prepare display dataframe
    display_df = df[available_columns].copy()
    
    # Format numeric columns as percentages (vectorized)
    for col in display_df.columns:
        if col != "Wave" and pd.api.types.is_numeric_dtype(display_df[col]):
            # Use vectorized string formatting for better performance
            display_df[col] = display_df[col].map(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
            )
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    # Summary metrics
    st.markdown("---")
    st.subheader("Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_waves = len(df)
        st.metric("Total Waves", total_waves)
    
    with col2:
        # Count waves with data (non-null Return_1D)
        if "Return_1D" in df.columns:
            waves_with_data = df["Return_1D"].notna().sum()
            st.metric("Waves with Data", waves_with_data)
        else:
            st.metric("Waves with Data", "N/A")
    
    with col3:
        # Coverage score average
        if "Coverage_Score" in df.columns:
            avg_coverage = df["Coverage_Score"].mean()
            st.metric("Avg Coverage Score", f"{avg_coverage:.1f}")
        else:
            st.metric("Avg Coverage Score", "N/A")


def render_single_wave_tab(df):
    """
    Render the Single Wave tab with dropdown and detailed metrics.
    
    Args:
        df (pd.DataFrame): Snapshot data
    """
    st.header("Single Wave")
    st.markdown("Detailed view of individual wave metrics")
    
    # Wave selector
    wave_names = sorted(df["Wave"].dropna().unique().tolist())
    
    if not wave_names:
        st.warning("No waves found in data.")
        return
    
    selected_wave = st.selectbox(
        "Select a wave:",
        options=wave_names,
        index=0
    )
    
    if not selected_wave:
        return
    
    # Get wave data
    filtered_df = df[df["Wave"] == selected_wave]
    if filtered_df.empty:
        st.error(f"Wave '{selected_wave}' not found in data.")
        return
    
    wave_data = filtered_df.iloc[0]
    
    st.subheader(selected_wave)
    
    # Display metrics in columns
    st.markdown("### Returns")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        val = wave_data.get("Return_1D", None)
        st.metric("1D Return", f"{val*100:.2f}%" if pd.notna(val) and isinstance(val, (int, float)) else "N/A")
    
    with col2:
        val = wave_data.get("Return_30D", None)
        st.metric("30D Return", f"{val*100:.2f}%" if pd.notna(val) and isinstance(val, (int, float)) else "N/A")
    
    with col3:
        val = wave_data.get("Return_60D", None)
        st.metric("60D Return", f"{val*100:.2f}%" if pd.notna(val) and isinstance(val, (int, float)) else "N/A")
    
    with col4:
        val = wave_data.get("Return_365D", None)
        st.metric("365D Return", f"{val*100:.2f}%" if pd.notna(val) and isinstance(val, (int, float)) else "N/A")
    
    st.markdown("### Alpha vs Benchmark")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        val = wave_data.get("Alpha_1D", None)
        st.metric("1D Alpha", f"{val*100:.2f}%" if pd.notna(val) and isinstance(val, (int, float)) else "N/A")
    
    with col2:
        val = wave_data.get("Alpha_30D", None)
        st.metric("30D Alpha", f"{val*100:.2f}%" if pd.notna(val) and isinstance(val, (int, float)) else "N/A")
    
    with col3:
        val = wave_data.get("Alpha_60D", None)
        st.metric("60D Alpha", f"{val*100:.2f}%" if pd.notna(val) and isinstance(val, (int, float)) else "N/A")
    
    with col4:
        val = wave_data.get("Alpha_365D", None)
        st.metric("365D Alpha", f"{val*100:.2f}%" if pd.notna(val) and isinstance(val, (int, float)) else "N/A")
    
    # Additional wave details
    st.markdown("---")
    st.markdown("### Additional Details")
    
    details_col1, details_col2 = st.columns(2)
    
    with details_col1:
        category = wave_data.get("Category", "N/A")
        st.text(f"Category: {category}")
        
        mode = wave_data.get("Mode", "N/A")
        st.text(f"Mode: {mode}")
        
        exposure = wave_data.get("Exposure", None)
        st.text(f"Exposure: {exposure:.2%}" if pd.notna(exposure) and isinstance(exposure, (int, float)) else "Exposure: N/A")
    
    with details_col2:
        coverage = wave_data.get("Coverage_Score", None)
        st.text(f"Coverage Score: {coverage}" if pd.notna(coverage) else "Coverage Score: N/A")
        
        regime = wave_data.get("Data_Regime_Tag", "N/A")
        st.text(f"Data Regime: {regime}")
        
        flags = wave_data.get("Flags", "N/A")
        st.text(f"Flags: {flags}")
    
    # Top holdings (if available in future data structure)
    # Placeholder for now since live_snapshot.csv doesn't have holdings
    st.markdown("---")
    st.markdown("### Top Holdings")
    st.info("Holdings data not available in current snapshot format.")


def render_diagnostics_tab(df):
    """
    Render the Diagnostics tab with metadata and system information.
    
    Args:
        df (pd.DataFrame): Snapshot data
    """
    st.header("Diagnostics")
    st.markdown("System metadata and data quality information")
    
    # Data freshness
    st.subheader("Data Freshness")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Last data date from snapshot
        if "Date" in df.columns:
            latest_date = df["Date"].max()
            st.metric("Latest Data Date", latest_date)
        else:
            st.metric("Latest Data Date", "N/A")
    
    with col2:
        # Current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.metric("Current Time", current_time)
    
    st.markdown("---")
    
    # Missing data analysis
    st.subheader("Missing Tickers / Data Issues")
    
    # Count waves with missing data
    if "Return_1D" in df.columns:
        missing_data_waves = df[df["Return_1D"].isna()]["Wave"].tolist()
        
        if missing_data_waves:
            st.warning(f"Found {len(missing_data_waves)} wave(s) with missing return data")
            with st.expander("View waves with missing data"):
                for wave in missing_data_waves:
                    st.text(f"• {wave}")
        else:
            st.success("✓ All waves have return data")
    
    # Check for flags
    if "Flags" in df.columns:
        waves_with_flags = df[df["Flags"].notna() & (df["Flags"] != "")]
        
        if len(waves_with_flags) > 0:
            st.warning(f"Found {len(waves_with_flags)} wave(s) with flags")
            with st.expander("View flagged waves"):
                st.dataframe(
                    waves_with_flags[["Wave", "Flags"]],
                    use_container_width=True
                )
        else:
            st.success("✓ No flagged waves")
    
    st.markdown("---")
    
    # Data quality metrics
    st.subheader("Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_rows = len(df)
        st.metric("Total Waves", total_rows)
    
    with col2:
        if "Coverage_Score" in df.columns:
            avg_coverage = df["Coverage_Score"].mean()
            st.metric("Avg Coverage", f"{avg_coverage:.1f}")
        else:
            st.metric("Avg Coverage", "N/A")
    
    with col3:
        total_columns = len(df.columns)
        st.metric("Data Columns", total_columns)
    
    st.markdown("---")
    
    # Error logs excerpt (from session state)
    st.subheader("Recent Error Logs")
    
    if st.session_state.last_error:
        st.error(f"Last error: {st.session_state.last_error}")
        st.caption(f"Error count: {st.session_state.error_count}/{MAX_CONSECUTIVE_ERRORS}")
    else:
        st.success("✓ No recent errors")
    
    # Circuit breaker status
    if st.session_state.circuit_breaker_active:
        st.warning("⚠️ Circuit breaker is active - auto-refresh paused")
    
    st.markdown("---")
    
    # File information
    st.subheader("Data Source Information")
    
    if os.path.exists(DATA_FILE):
        file_path = DATA_FILE
    elif os.path.exists(f"data/{DATA_FILE}"):
        file_path = f"data/{DATA_FILE}"
    else:
        file_path = None
    
    if file_path:
        file_stats = os.stat(file_path)
        file_size_kb = file_stats.st_size / 1024
        file_modified = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        st.text(f"File: {file_path}")
        st.text(f"Size: {file_size_kb:.2f} KB")
        st.text(f"Last Modified: {file_modified}")
    else:
        st.error(f"Data file '{DATA_FILE}' not found")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Set page configuration
    st.set_page_config(
        page_title="WAVES Intelligence™ — Minimal Console v1",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply auto-refresh if enabled
    # IMPORTANT: This is the ONLY st_autorefresh() call in the file
    if (AUTOREFRESH_AVAILABLE and 
        st.session_state.auto_refresh_enabled and 
        not st.session_state.circuit_breaker_active):
        st_autorefresh(interval=DEFAULT_REFRESH_INTERVAL_MS, key="main_autorefresh")
    
    # Page header
    st.title("📊 WAVES Intelligence™ — Minimal Console v1")
    st.markdown("Lightweight console for WAVES data visualization and diagnostics")
    
    # Render sidebar
    render_sidebar()
    
    # Load data
    df = load_snapshot_data()
    
    # Handle missing file
    if df is None:
        st.error(f"⚠️ Missing File: '{DATA_FILE}' not found")
        st.warning("The application cannot proceed without the data file.")
        st.info(f"Please ensure '{DATA_FILE}' exists in the current directory or 'data/' subdirectory.")
        
        # Increment error counter
        handle_error(f"Data file '{DATA_FILE}' not found")
        
        # Stop execution here - no further logic
        st.stop()
    
    # Reset error state on successful load
    reset_error_state()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Single Wave", "🔧 Diagnostics"])
    
    with tab1:
        try:
            render_overview_tab(df)
        except Exception as e:
            st.error(f"Error rendering Overview tab: {str(e)}")
            handle_error(str(e))
    
    with tab2:
        try:
            render_single_wave_tab(df)
        except Exception as e:
            st.error(f"Error rendering Single Wave tab: {str(e)}")
            handle_error(str(e))
    
    with tab3:
        try:
            render_diagnostics_tab(df)
        except Exception as e:
            st.error(f"Error rendering Diagnostics tab: {str(e)}")
            handle_error(str(e))
    
    # Footer
    st.markdown("---")
    st.caption(f"WAVES Intelligence™ Minimal Console v1 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
