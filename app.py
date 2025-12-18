# app.py — WAVES Intelligence™ Production App
# Main Streamlit application for WAVES Vector Trading System

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Optional

# Import the waves engine
import waves_engine as we

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="WAVES Intelligence™",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clear Streamlit cache on deployment to avoid issues
if 'cache_cleared' not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.cache_cleared = True

# ============================================================
# Helper Functions - DEFINED BEFORE USE
# ============================================================

def benchmark_snapshot_id(wave_name: str, benchmark_mix: Dict[str, float]) -> str:
    """
    Generate a unique identifier for a benchmark snapshot.
    
    Args:
        wave_name: Name of the wave
        benchmark_mix: Dictionary mapping ticker symbols to weights
        
    Returns:
        A unique string identifier for this benchmark configuration
    """
    # Sort tickers for consistent ID generation
    sorted_tickers = sorted(benchmark_mix.items())
    ticker_str = "_".join([f"{t}:{w:.2f}" for t, w in sorted_tickers])
    
    # Use hash for long IDs to prevent length issues
    full_id = f"{wave_name}_{ticker_str}"
    if len(full_id) > 100:
        import hashlib
        ticker_hash = hashlib.md5(ticker_str.encode()).hexdigest()[:8]
        return f"{wave_name}_{ticker_hash}"
    
    return full_id


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as a percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a value as currency."""
    return f"${value:,.{decimals}f}"


def get_benchmark_weights(wave_name: str) -> Dict[str, float]:
    """
    Get benchmark weights for a given wave.
    
    Args:
        wave_name: Name of the wave
        
    Returns:
        Dictionary mapping ticker symbols to weights
    """
    # Default benchmarks based on wave type
    DEFAULT_BENCHMARKS = {
        "SMID": {"IWM": 0.6, "IJR": 0.4},
        "Large": {"SPY": 1.0},
        "Thematic": {"QQQ": 1.0},
        "Defensive": {"BIL": 1.0},
        "Crypto": {"BTC-USD": 1.0},
    }
    
    try:
        benchmark_table = we.get_benchmark_mix_table()
        wave_row = benchmark_table[benchmark_table['wave'] == wave_name]
        
        if wave_row.empty:
            # Try to find appropriate default based on wave name
            for wave_type, benchmark in DEFAULT_BENCHMARKS.items():
                if wave_type.lower() in wave_name.lower():
                    return benchmark
            return {"SPY": 1.0}
        
        # Extract benchmark mix from the row
        # The benchmark_mix_table returns a dataframe with benchmark ticker columns
        benchmark_weights = {}
        for col in wave_row.columns:
            if col not in ['wave'] and wave_row[col].iloc[0] > 0:
                benchmark_weights[col] = float(wave_row[col].iloc[0])
        
        return benchmark_weights if benchmark_weights else {"SPY": 1.0}
    except Exception as e:
        st.error(f"Error getting benchmark weights: {e}")
        return {"SPY": 1.0}


def create_performance_chart(nav_df: pd.DataFrame, wave_name: str) -> go.Figure:
    """Create a performance chart for the wave."""
    fig = go.Figure()
    
    # Add wave NAV line
    fig.add_trace(go.Scatter(
        x=nav_df.index,
        y=nav_df['nav'],
        mode='lines',
        name=wave_name,
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add benchmark line if available
    if 'benchmark_nav' in nav_df.columns:
        fig.add_trace(go.Scatter(
            x=nav_df.index,
            y=nav_df['benchmark_nav'],
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f"{wave_name} Performance",
        xaxis_title="Date",
        yaxis_title="NAV",
        hovermode='x unified',
        height=400
    )
    
    return fig


# ============================================================
# Main Application
# ============================================================

def main():
    """Main application function."""
    
    # Header
    st.title("🌊 WAVES Intelligence™")
    st.caption("Vector Trading System — Dynamic Strategy with VIX + SmartSafe")
    
    # Sidebar
    with st.sidebar:
        st.header("Wave Configuration")
        
        # Get available waves
        available_waves = we.get_all_waves()
        selected_wave = st.selectbox(
            "Select Wave",
            options=available_waves,
            index=0 if available_waves else None
        )
        
        # Get available modes
        available_modes = we.get_modes()
        selected_mode = st.selectbox(
            "Select Mode",
            options=available_modes,
            index=0 if available_modes else None
        )
        
        # History period
        history_days = st.slider(
            "History (days)",
            min_value=30,
            max_value=730,
            value=365,
            step=30
        )
        
        # Get benchmark mix for selected wave
        bm_mix = get_benchmark_weights(selected_wave)
        
        # Generate benchmark snapshot ID - USING THE FUNCTION DEFINED ABOVE
        bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
        
        st.markdown("---")
        st.caption(f"Benchmark ID: {bm_id[:30]}...")
    
    # Main content area
    if not selected_wave:
        st.warning("No waves available. Please check the configuration.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance", 
        "📈 Holdings", 
        "🎯 Diagnostics",
        "⚙️ Parameters"
    ])
    
    # Tab 1: Performance
    with tab1:
        st.subheader(f"Performance Analysis — {selected_wave}")
        
        try:
            with st.spinner("Loading performance data..."):
                # Get historical NAV
                nav_df = we.compute_history_nav(
                    wave_name=selected_wave,
                    mode=selected_mode,
                    days=history_days
                )
                
                if nav_df.empty:
                    st.warning("No performance data available for this wave.")
                else:
                    # Display chart
                    chart = create_performance_chart(nav_df, selected_wave)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Show latest NAV values
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        latest_nav = nav_df['nav'].iloc[-1]
                        st.metric("Current NAV", f"{latest_nav:.4f}")
                    
                    with col2:
                        if len(nav_df) > 1:
                            initial_nav = nav_df['nav'].iloc[0]
                            if initial_nav != 0:
                                total_return = (latest_nav / initial_nav - 1) * 100
                                st.metric("Total Return", f"{total_return:.2f}%")
                            else:
                                st.metric("Total Return", "N/A")
                    
                    with col3:
                        if 'benchmark_nav' in nav_df.columns and len(nav_df) > 1:
                            initial_bm = nav_df['benchmark_nav'].iloc[0]
                            if initial_bm != 0 and initial_nav != 0:
                                total_return = (latest_nav / initial_nav - 1) * 100
                                bm_return = (nav_df['benchmark_nav'].iloc[-1] / initial_bm - 1) * 100
                                alpha = total_return - bm_return
                                st.metric("Alpha vs Benchmark", f"{alpha:.2f}%")
                            else:
                                st.metric("Alpha vs Benchmark", "N/A")
                    
                    # Show data table in expander
                    with st.expander("View Raw Data"):
                        st.dataframe(nav_df, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error loading performance data: {e}")
            st.exception(e)
    
    # Tab 2: Holdings
    with tab2:
        st.subheader(f"Current Holdings — {selected_wave}")
        
        try:
            with st.spinner("Loading holdings..."):
                holdings_df = we.get_wave_holdings(selected_wave)
                
                if holdings_df.empty:
                    st.info("No holdings data available for this wave.")
                else:
                    # Display holdings table
                    st.dataframe(holdings_df, use_container_width=True)
                    
                    # Show summary stats
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Number of Holdings", len(holdings_df))
                    
                    with col2:
                        if 'weight' in holdings_df.columns:
                            total_weight = holdings_df['weight'].sum()
                            st.metric("Total Weight", f"{total_weight:.2%}")
                    
                    with col3:
                        if 'weight' in holdings_df.columns:
                            max_weight = holdings_df['weight'].max()
                            st.metric("Max Position", f"{max_weight:.2%}")
                        
        except Exception as e:
            st.error(f"Error loading holdings: {e}")
            st.exception(e)
    
    # Tab 3: Diagnostics
    with tab3:
        st.subheader(f"Diagnostics — {selected_wave}")
        
        try:
            with st.spinner("Loading diagnostics..."):
                diagnostics = we.get_latest_diagnostics(
                    wave_name=selected_wave,
                    mode=selected_mode,
                    days=history_days
                )
                
                if not diagnostics:
                    st.info("No diagnostics available for this wave.")
                else:
                    # Display diagnostics in a structured way
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Risk Metrics**")
                        for key in ['volatility', 'max_drawdown', 'sharpe_ratio']:
                            if key in diagnostics:
                                st.metric(
                                    key.replace('_', ' ').title(),
                                    f"{diagnostics[key]:.4f}"
                                )
                    
                    with col2:
                        st.markdown("**Performance Metrics**")
                        for key in ['total_return', 'annualized_return', 'tracking_error']:
                            if key in diagnostics:
                                st.metric(
                                    key.replace('_', ' ').title(),
                                    f"{diagnostics[key]:.4f}"
                                )
                    
                    # Show all diagnostics in expander
                    with st.expander("View All Diagnostics"):
                        st.json(diagnostics)
                        
        except Exception as e:
            st.error(f"Error loading diagnostics: {e}")
            st.exception(e)
    
    # Tab 4: Parameters
    with tab4:
        st.subheader(f"Parameters — {selected_wave}")
        
        try:
            with st.spinner("Loading parameters..."):
                params = we.get_parameter_defaults(
                    wave_name=selected_wave,
                    mode=selected_mode
                )
                
                if not params:
                    st.info("No parameters available for this wave.")
                else:
                    st.markdown("**Default Parameters for this Wave/Mode combination:**")
                    
                    # Display parameters in a nice table
                    params_df = pd.DataFrame([
                        {"Parameter": k, "Value": v}
                        for k, v in params.items()
                    ])
                    st.dataframe(params_df, use_container_width=True)
                    
                    st.info(
                        "💡 These are the default parameters. "
                        "Advanced users can override these in shadow simulations."
                    )
                        
        except Exception as e:
            st.error(f"Error loading parameters: {e}")
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.caption(f"WAVES Intelligence™ v17.1 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()