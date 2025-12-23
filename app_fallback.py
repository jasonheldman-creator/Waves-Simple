"""
Safe Fallback UI - Minimal Stable Version
This provides a known-good backup state with basic functionality only.
Excludes optional or risky features that might fail.
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime


def run():
    """
    Main entry point for fallback UI.
    Provides minimal, stable functionality with basic wave display.
    """
    st.title("⚠️ Safe Mode - Minimal Console")
    
    st.info("""
    **Safe Mode Active**  
    You are viewing a simplified version of the Institutional Console with stable, core features only.
    Advanced analytics and optional features are disabled in this mode.
    """)
    
    # Simple header section
    render_simple_header()
    
    # Wave selector
    selected_wave = render_wave_selector()
    
    # Basic holdings display
    render_basic_holdings(selected_wave)
    
    # Basic performance table
    render_basic_performance(selected_wave)


def render_simple_header():
    """Render a simple header with system info."""
    st.markdown("---")
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mode", "Safe Mode")
        st.metric("Features", "Basic Only")
    
    with col2:
        st.metric("Status", "Operational")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.metric("Timestamp", current_time)
    
    st.markdown("---")


def render_wave_selector():
    """Render a simple wave selector."""
    st.subheader("Wave Selection")
    
    # Get available waves from wave history or use defaults
    available_waves = get_safe_wave_list()
    
    if not available_waves:
        st.warning("No waves available. Using default wave.")
        return "S&P 500 Wave"
    
    selected_wave = st.selectbox(
        "Select a wave to view:",
        options=available_waves,
        index=0 if "S&P 500 Wave" not in available_waves else available_waves.index("S&P 500 Wave"),
        help="Choose a wave to display basic information"
    )
    
    st.success(f"Selected: **{selected_wave}**")
    
    return selected_wave


def render_basic_holdings(wave_name: str):
    """Render basic holdings information for the selected wave."""
    st.markdown("---")
    st.subheader(f"Holdings: {wave_name}")
    
    # Try to load wave data safely
    df = safe_load_wave_data()
    
    if df is None or len(df) == 0:
        st.info("No holdings data available in safe mode.")
        return
    
    # Filter for the selected wave
    if 'wave' in df.columns:
        wave_df = df[df['wave'] == wave_name].copy()
    else:
        st.info("Wave data not available.")
        return
    
    if len(wave_df) == 0:
        st.info(f"No data available for {wave_name}")
        return
    
    # Display basic stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(wave_df))
    
    with col2:
        if 'date' in wave_df.columns:
            latest_date = wave_df['date'].max()
            st.metric("Latest Date", str(latest_date)[:10] if pd.notna(latest_date) else "N/A")
    
    with col3:
        if 'date' in wave_df.columns:
            earliest_date = wave_df['date'].min()
            st.metric("Earliest Date", str(earliest_date)[:10] if pd.notna(earliest_date) else "N/A")


def render_basic_performance(wave_name: str):
    """Render basic performance metrics for the selected wave."""
    st.markdown("---")
    st.subheader(f"Performance: {wave_name}")
    
    # Try to load wave data safely
    df = safe_load_wave_data()
    
    if df is None or len(df) == 0:
        st.info("No performance data available in safe mode.")
        return
    
    # Filter for the selected wave
    if 'wave' in df.columns:
        wave_df = df[df['wave'] == wave_name].copy()
    else:
        st.info("Wave data not available.")
        return
    
    if len(wave_df) == 0:
        st.info(f"No performance data available for {wave_name}")
        return
    
    # Display basic performance table (last 10 records)
    st.write("**Recent Performance (Last 10 Records)**")
    
    # Select relevant columns if they exist
    display_columns = []
    for col in ['date', 'portfolio_return', 'benchmark_return', 'alpha']:
        if col in wave_df.columns:
            display_columns.append(col)
    
    if display_columns:
        recent_data = wave_df[display_columns].tail(10).copy()
        
        # Format date column if present
        if 'date' in recent_data.columns:
            recent_data['date'] = recent_data['date'].astype(str).str[:10]
        
        # Format numeric columns
        for col in ['portfolio_return', 'benchmark_return', 'alpha']:
            if col in recent_data.columns:
                recent_data[col] = recent_data[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        
        st.dataframe(recent_data, use_container_width=True)
    else:
        st.info("No performance columns available to display.")
    
    # Calculate simple summary statistics if possible
    if 'portfolio_return' in wave_df.columns:
        st.write("**Summary Statistics**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mean_return = wave_df['portfolio_return'].mean()
            st.metric("Avg Return", f"{mean_return:.4f}" if pd.notna(mean_return) else "N/A")
        
        with col2:
            if 'benchmark_return' in wave_df.columns:
                mean_bench = wave_df['benchmark_return'].mean()
                st.metric("Avg Benchmark", f"{mean_bench:.4f}" if pd.notna(mean_bench) else "N/A")
        
        with col3:
            if 'alpha' in wave_df.columns:
                mean_alpha = wave_df['alpha'].mean()
                st.metric("Avg Alpha", f"{mean_alpha:.4f}" if pd.notna(mean_alpha) else "N/A")


def get_safe_wave_list():
    """
    Safely retrieve list of available waves.
    Returns a minimal default list if data cannot be loaded.
    """
    try:
        df = safe_load_wave_data()
        
        if df is not None and 'wave' in df.columns:
            waves = sorted(df['wave'].unique().tolist())
            return [w for w in waves if w and str(w).strip()]
        
        # Fallback to default waves
        return [
            "S&P 500 Wave",
            "Growth Wave",
            "Small Cap Growth Wave"
        ]
        
    except Exception:
        # Ultimate fallback
        return [
            "S&P 500 Wave",
            "Growth Wave"
        ]


def safe_load_wave_data():
    """
    Safely load wave history data with minimal error handling.
    Returns DataFrame or None if unavailable.
    """
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        
        if not os.path.exists(wave_history_path):
            return None
        
        df = pd.read_csv(wave_history_path)
        
        if df is None or len(df) == 0:
            return None
        
        # Convert date to datetime if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        
        # Ensure wave column exists
        if 'wave' not in df.columns:
            if 'display_name' in df.columns:
                df['wave'] = df['display_name']
            elif 'wave_id' in df.columns:
                df['wave'] = df['wave_id']
        
        # Basic cleanup
        if 'wave' in df.columns:
            df['wave'] = df['wave'].str.strip()
        
        return df
        
    except Exception:
        return None
