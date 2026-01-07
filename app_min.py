"""
Minimal Console - Streamlit Entry File
A lightweight, minimal Streamlit console for quick testing and basic operations.

Features:
- Minimal dependencies
- Basic UI components
- Quick startup time
- Essential functionality only
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime


def main():
    """Main entry point for minimal console."""
    
    # Set page configuration
    st.set_page_config(
        page_title="Minimal Console",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ============================================================================
    # PROOF BANNER - Diagnostics and Visibility
    # ============================================================================
    # Initialize run counter in session state
    if "proof_run_counter" not in st.session_state:
        st.session_state.proof_run_counter = 0
    else:
        st.session_state.proof_run_counter += 1
    
    # Display Proof Banner
    st.markdown(
        f"""
        <div style="background-color: #2d2d2d; padding: 12px 20px; border: 2px solid #ff9800; margin-bottom: 16px; border-radius: 6px;">
            <div style="color: #ff9800; font-size: 14px; font-family: monospace; font-weight: bold; margin-bottom: 4px;">
                🔍 PROOF BANNER - DIAGNOSTICS MODE
            </div>
            <div style="color: #e0e0e0; font-size: 12px; font-family: monospace;">
                <strong>APP FILE:</strong> app_min.py<br>
                <strong>DIAGNOSTIC ID:</strong> DIAG_MIN_2026_01_05_A<br>
                <strong>TIMESTAMP:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>RUN COUNTER:</strong> {st.session_state.proof_run_counter}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Header
    st.title("📊 Minimal Console")
    st.markdown("Lightweight Streamlit interface for basic operations")
    
    # Sidebar
    with st.sidebar:
        st.header("Console Info")
        st.info(f"**Mode:** Minimal\n\n**Version:** 1.0\n\n**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        st.subheader("Quick Stats")
        st.metric("Status", "Active", delta="Running")
        
        # ========================================================================
        # DIAGNOSTICS DEBUG PANEL - Collapsible
        # ========================================================================
        st.markdown("---")
        with st.expander("🔍 Diagnostics Debug Panel", expanded=False):
            st.markdown("**Diagnostics & Visibility Panel**")
            
            try:
                # Display selected_wave_id if it exists
                selected_wave_id = st.session_state.get("selected_wave_id", "N/A - Minimal Console")
                st.text(f"selected_wave_id: {selected_wave_id}")
                
                # Display selectbox key info
                st.text("Selectbox key: N/A (Minimal Console)")
                
                # Check wave registry status
                st.text("Wave Registry: N/A (Minimal Console)")
                
                # Check portfolio snapshot status
                try:
                    snapshot_path = "data/live_snapshot.csv"
                    if os.path.exists(snapshot_path):
                        snapshot_df = pd.read_csv(snapshot_path)
                        row_count = len(snapshot_df)
                        st.text(f"Portfolio Snapshot: Loaded ({row_count} rows)")
                    else:
                        st.text("Portfolio Snapshot: File not found")
                except Exception as e:
                    st.text(f"Portfolio Snapshot: Error - {str(e)}")
                
                # Check price cache status
                try:
                    price_cache_path = "data/cache/prices_cache.parquet"
                    cache_exists = os.path.exists(price_cache_path)
                    st.text(f"Price Cache Path: {price_cache_path}")
                    st.text(f"Price Cache Exists: {cache_exists}")
                    
                    if cache_exists:
                        # Get file size
                        file_size = os.path.getsize(price_cache_path) / (1024 * 1024)  # Convert to MB
                        st.text(f"Price Cache Size: {file_size:.2f} MB")
                except Exception as e:
                    st.text(f"Price Cache: Error - {str(e)}")
                    
            except Exception as e:
                st.error(f"Debug panel error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Main content area
    render_main_content()
    
    # Footer
    st.markdown("---")
    st.caption(f"Minimal Console | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def render_main_content():
    """Render the main content area."""
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "📋 Data", "⚙️ Settings"])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_data_tab()
    
    with tab3:
        render_settings_tab()


def render_overview_tab():
    """Render the overview tab with basic metrics."""
    st.header("Overview")
    
    # Display basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Sessions", "1", delta=None)
    
    with col2:
        st.metric("Data Files", get_data_file_count(), delta=None)
    
    with col3:
        st.metric("Console Mode", "Minimal")
    
    with col4:
        st.metric("Uptime", "Active")
    
    st.markdown("---")
    
    # System information
    st.subheader("System Information")
    
    system_info = {
        "Working Directory": os.getcwd(),
        "Console Type": "Minimal Streamlit Console",
        "Python Version": "3.x",
        "Framework": "Streamlit"
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")
    
    # Optional: Display available data
    st.markdown("---")
    st.subheader("Available Data Files")
    
    data_files = get_available_data_files()
    if data_files:
        st.success(f"Found {len(data_files)} data file(s)")
        with st.expander("View Files"):
            for file in data_files[:10]:  # Show first 10
                st.text(f"• {file}")
    else:
        st.info("No data files found in current directory")


def render_data_tab():
    """Render the data tab for viewing basic data."""
    st.header("Data Viewer")
    
    st.info("This tab provides basic data viewing capabilities.")
    
    # Check for common data files
    data_files = get_available_data_files()
    
    if not data_files:
        st.warning("No CSV data files found in the current directory.")
        return
    
    # File selector
    selected_file = st.selectbox(
        "Select a data file to view:",
        options=data_files,
        index=0
    )
    
    if selected_file:
        try:
            # Load and display the selected file
            # Validate file is in the list to prevent path traversal
            if selected_file not in data_files:
                st.error("Invalid file selection")
                return
            
            file_path = os.path.join(os.getcwd(), selected_file)
            df = pd.read_csv(file_path)
            
            st.success(f"Loaded: {selected_file}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", len(df))
            
            with col2:
                st.metric("Columns", len(df.columns))
            
            with col3:
                st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Display column names
            st.subheader("Columns")
            st.write(", ".join(df.columns.tolist()))
            
            # Display data preview
            st.subheader("Data Preview (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Optional: Display data tail
            with st.expander("View Last 10 Rows"):
                st.dataframe(df.tail(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def render_settings_tab():
    """Render the settings tab."""
    st.header("Settings")
    
    st.info("Basic configuration options for the minimal console.")
    
    # Display mode
    st.subheader("Display Options")
    
    use_wide_layout = st.checkbox("Use wide layout", value=True)
    show_debug_info = st.checkbox("Show debug information", value=False)
    
    if show_debug_info:
        st.subheader("Debug Information")
        debug_info = {
            "Working Directory": os.getcwd(),
            "Session State Keys": list(st.session_state.keys()) if st.session_state else [],
            "Environment": "Debug" if show_debug_info else "Production"
        }
        st.json(debug_info)
    
    st.markdown("---")
    
    # About section
    st.subheader("About")
    st.markdown("""
    **Minimal Console** is a lightweight Streamlit application designed for:
    - Quick testing and prototyping
    - Basic data viewing
    - Minimal resource usage
    - Fast startup time
    
    This console provides essential functionality without the overhead of a full-featured application.
    """)


def get_data_file_count():
    """Count the number of CSV data files in the current directory."""
    try:
        return len(get_available_data_files())
    except Exception:
        return 0


def get_available_data_files():
    """Get list of available CSV data files."""
    try:
        current_dir = os.getcwd()
        files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
        return sorted(files)
    except Exception:
        return []


if __name__ == "__main__":
    main()
