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
            "Environment": "Production" if not show_debug_info else "Debug"
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
