"""
Institutional Console v2 - Executive Layer v2
Full implementation with advanced analytics and visualization

Modular structure:
1. Configuration and styling
2. Utility functions  
3. Safe data-loading helpers
4. Data processing and calculation functions
5. Visualization components
6. Reusable UI components
7. Render functions for tabs and analytics
8. Main entry point
"""

import streamlit as st
import subprocess
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# SECTION 1: CONFIGURATION AND STYLING
# ============================================================================

st.set_page_config(page_title="Institutional Console - Executive Layer v2", layout="wide")


# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

def get_git_commit_hash():
    """Get the current git commit hash, return 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_branch_name():
    """Get the current git branch name, return 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_deploy_timestamp():
    """Get the current timestamp as deploy timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


def calculate_wavescore(wave_data):
    """
    Calculate WaveScore for a wave based on cumulative alpha over 30 days.
    Returns a score between 0 and 100.
    
    WaveScore formula:
    - Base score: 50
    - Add points for cumulative alpha (1000x multiplier)
    - Adjust for consistency (reduce score for high volatility)
    - Clamp to 0-100 range
    """
    try:
        if wave_data is None or len(wave_data) == 0:
            return 0
        
        if 'alpha' not in wave_data.columns:
            return 0
        
        # Calculate cumulative alpha
        cumulative_alpha = wave_data['alpha'].sum()
        
        # Base score calculation
        base_score = (cumulative_alpha * 1000) + 50
        
        # Consistency adjustment: penalize high volatility
        alpha_std = wave_data['alpha'].std()
        if alpha_std > 0:
            # Reduce score if volatility is high relative to returns
            consistency_penalty = min(10, alpha_std * 200)
            base_score -= consistency_penalty
        
        # Normalize to 0-100 range
        wavescore = min(100, max(0, base_score))
        
        return wavescore
    except Exception:
        return 0


def calculate_wave_correlation(wave1_data, wave2_data):
    """
    Calculate correlation between two waves based on their returns.
    Returns correlation coefficient or None if unavailable.
    """
    try:
        if wave1_data is None or wave2_data is None:
            return None
        
        if len(wave1_data) == 0 or len(wave2_data) == 0:
            return None
        
        if 'date' not in wave1_data.columns or 'date' not in wave2_data.columns:
            return None
        
        if 'portfolio_return' not in wave1_data.columns or 'portfolio_return' not in wave2_data.columns:
            return None
        
        # Merge on date to get overlapping periods
        wave1_returns = wave1_data[['date', 'portfolio_return']].rename(columns={'portfolio_return': 'return1'})
        wave2_returns = wave2_data[['date', 'portfolio_return']].rename(columns={'portfolio_return': 'return2'})
        
        merged = pd.merge(wave1_returns, wave2_returns, on='date', how='inner')
        
        if len(merged) < 2:
            return None
        
        correlation = merged['return1'].corr(merged['return2'])
        
        return correlation
        
    except Exception:
        return None


def calculate_wave_metrics(wave_data):
    """
    Calculate comprehensive metrics for a wave.
    Returns a dictionary with all calculated metrics.
    """
    metrics = {
        'cumulative_return': 'N/A',
        'cumulative_alpha': 'N/A',
        'volatility': 'N/A',
        'max_drawdown': 'N/A',
        'wavescore': 'N/A',
        'sharpe_ratio': 'N/A',
        'win_rate': 'N/A'
    }
    
    try:
        if wave_data is None or len(wave_data) == 0:
            return metrics
        
        # Calculate alpha
        if 'portfolio_return' in wave_data.columns and 'benchmark_return' in wave_data.columns:
            wave_data = wave_data.copy()
            wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
            
            # Cumulative return
            cumulative_return = wave_data['portfolio_return'].sum()
            metrics['cumulative_return'] = cumulative_return
            
            # Cumulative alpha
            cumulative_alpha = wave_data['alpha'].sum()
            metrics['cumulative_alpha'] = cumulative_alpha
            
            # Volatility
            volatility = wave_data['portfolio_return'].std()
            metrics['volatility'] = volatility
            
            # WaveScore
            wavescore = calculate_wavescore(wave_data)
            metrics['wavescore'] = wavescore
            
            # Sharpe ratio
            avg_return = wave_data['portfolio_return'].mean()
            if volatility > 0:
                sharpe = (avg_return / volatility) * np.sqrt(252)
                metrics['sharpe_ratio'] = sharpe
            
            # Win rate
            positive_days = len(wave_data[wave_data['alpha'] > 0])
            total_days = len(wave_data)
            if total_days > 0:
                metrics['win_rate'] = positive_days / total_days
            
            # Max drawdown
            cumulative_returns = (1 + wave_data['portfolio_return']).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
    
    except Exception:
        pass
    
    return metrics


def determine_winner(wave1_metrics, wave2_metrics):
    """
    Determine which wave is the winner based on WaveScore.
    Returns a tuple: (winner_name, notes)
    """
    try:
        wave1_score = wave1_metrics.get('wavescore', 0)
        wave2_score = wave2_metrics.get('wavescore', 0)
        
        if wave1_score == 'N/A' or wave2_score == 'N/A':
            return None, "Insufficient data to determine winner"
        
        if wave1_score > wave2_score:
            winner = wave1_metrics['name']
            margin = wave1_score - wave2_score
            notes = f"{winner} leads by {margin:.1f} WaveScore points"
        elif wave2_score > wave1_score:
            winner = wave2_metrics['name']
            margin = wave2_score - wave1_score
            notes = f"{winner} leads by {margin:.1f} WaveScore points"
        else:
            winner = "TIE"
            notes = "Both waves have identical WaveScore"
        
        return winner, notes
        
    except Exception:
        return None, "Error determining winner"


# ============================================================================
# SECTION 3: SAFE DATA-LOADING HELPERS
# ============================================================================

def safe_load_wave_history():
    """
    Safely load wave history data with comprehensive error handling.
    Returns DataFrame or None if unavailable.
    """
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        
        if not os.path.exists(wave_history_path):
            return None
        
        df = pd.read_csv(wave_history_path)
        
        if df is None or len(df) == 0:
            return None
        
        # Validate required columns
        if 'date' not in df.columns:
            return None
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        if len(df) == 0:
            return None
        
        return df
        
    except Exception:
        return None


def get_latest_data_timestamp():
    """Get the latest available 'as of' data timestamp from wave_history.csv."""
    try:
        df = safe_load_wave_history()
        if df is not None and 'date' in df.columns and len(df) > 0:
            latest_date = df['date'].max()
            return latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "unknown"
    except Exception:
        pass
    return "unknown"


def get_available_waves():
    """
    Get list of available waves from wave history data.
    Returns sorted list of wave names or empty list if unavailable.
    """
    try:
        df = safe_load_wave_history()
        
        if df is None:
            return []
        
        if 'wave' not in df.columns:
            return []
        
        waves = df['wave'].dropna().unique()
        return sorted(waves.tolist())
        
    except Exception:
        return []


def get_wave_data_filtered(wave_name=None, days=30):
    """
    Get wave data filtered by wave name and/or date range.
    Returns DataFrame or None if unavailable.
    
    Args:
        wave_name: Name of wave to filter (None for all waves)
        days: Number of days to include (from latest date)
    """
    try:
        df = safe_load_wave_history()
        
        if df is None:
            return None
        
        # Filter by wave if specified
        if wave_name is not None:
            if 'wave' not in df.columns:
                return None
            df = df[df['wave'] == wave_name].copy()
        
        if len(df) == 0:
            return None
        
        # Filter by date range if days specified
        if days is not None and days > 0:
            latest_date = df['date'].max()
            cutoff_date = latest_date - timedelta(days=days)
            df = df[df['date'] >= cutoff_date].copy()
        
        if len(df) == 0:
            return None
        
        return df
        
    except Exception:
        return None


# ============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# ============================================================================

def create_wavescore_bar_chart(leaderboard_df):
    """
    Create a horizontal bar chart for WaveScore leaderboard.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if leaderboard_df is None or len(leaderboard_df) == 0:
            return None
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars with color gradient based on score
        fig.add_trace(go.Bar(
            y=leaderboard_df['Wave'],
            x=leaderboard_df['WaveScore'],
            orientation='h',
            marker=dict(
                color=leaderboard_df['WaveScore'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="WaveScore")
            ),
            text=leaderboard_df['WaveScore'].apply(lambda x: f"{x:.1f}"),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>WaveScore: %{x:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top Performers by WaveScore",
            xaxis_title="WaveScore",
            yaxis_title="Wave",
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception:
        return None


def create_movers_chart(movers_df):
    """
    Create a waterfall chart showing biggest WaveScore movers.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if movers_df is None or len(movers_df) == 0:
            return None
        
        # Create bar chart with color coding for positive/negative changes
        fig = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in movers_df['Change']]
        
        fig.add_trace(go.Bar(
            x=movers_df['Wave'],
            y=movers_df['Change'],
            marker_color=colors,
            text=movers_df['Change'].apply(lambda x: f"{x:+.1f}"),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Change: %{y:+.1f}<br>Previous: %{customdata[0]:.1f}<br>Current: %{customdata[1]:.1f}<extra></extra>',
            customdata=movers_df[['Previous', 'Current']].values
        ))
        
        fig.update_layout(
            title="Biggest WaveScore Movers (Month-over-Month)",
            xaxis_title="Wave",
            yaxis_title="WaveScore Change",
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=-45)
        
        return fig
        
    except Exception:
        return None


def create_wave_performance_chart(wave_data, wave_name):
    """
    Create a multi-panel chart showing wave performance over time.
    Includes cumulative returns, alpha, and drawdown.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if wave_data is None or len(wave_data) == 0:
            return None
        
        if 'date' not in wave_data.columns:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Cumulative Returns',
                'Daily Alpha',
                'Drawdown'
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Calculate metrics
        if 'portfolio_return' in wave_data.columns:
            cumulative_returns = (1 + wave_data['portfolio_return']).cumprod() - 1
            
            # Panel 1: Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=wave_data['date'],
                    y=cumulative_returns * 100,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue', width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            if 'benchmark_return' in wave_data.columns:
                cumulative_benchmark = (1 + wave_data['benchmark_return']).cumprod() - 1
                fig.add_trace(
                    go.Scatter(
                        x=wave_data['date'],
                        y=cumulative_benchmark * 100,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='gray', width=2, dash='dash'),
                        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Panel 2: Daily alpha
        if 'alpha' in wave_data.columns or ('portfolio_return' in wave_data.columns and 'benchmark_return' in wave_data.columns):
            if 'alpha' not in wave_data.columns:
                wave_data = wave_data.copy()
                wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
            
            colors = ['green' if x > 0 else 'red' for x in wave_data['alpha']]
            
            fig.add_trace(
                go.Bar(
                    x=wave_data['date'],
                    y=wave_data['alpha'] * 100,
                    name='Alpha',
                    marker_color=colors,
                    hovertemplate='%{x|%Y-%m-%d}<br>Alpha: %{y:.3f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Panel 3: Drawdown
        if 'portfolio_return' in wave_data.columns:
            cumulative_returns = (1 + wave_data['portfolio_return']).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            
            fig.add_trace(
                go.Scatter(
                    x=wave_data['date'],
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Alpha (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.update_layout(
            title=f"Performance Analysis: {wave_name}",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception:
        return None


def create_comparison_radar_chart(wave1_metrics, wave2_metrics):
    """
    Create a radar chart comparing two waves across multiple dimensions.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if wave1_metrics is None or wave2_metrics is None:
            return None
        
        # Define metrics for comparison (normalized to 0-100 scale)
        categories = []
        wave1_values = []
        wave2_values = []
        
        # WaveScore (already 0-100)
        if wave1_metrics['wavescore'] != 'N/A' and wave2_metrics['wavescore'] != 'N/A':
            categories.append('WaveScore')
            wave1_values.append(wave1_metrics['wavescore'])
            wave2_values.append(wave2_metrics['wavescore'])
        
        # Sharpe Ratio (normalize: assume range -2 to 4, map to 0-100)
        if wave1_metrics['sharpe_ratio'] != 'N/A' and wave2_metrics['sharpe_ratio'] != 'N/A':
            categories.append('Sharpe Ratio')
            wave1_values.append(min(100, max(0, (wave1_metrics['sharpe_ratio'] + 2) * 100 / 6)))
            wave2_values.append(min(100, max(0, (wave2_metrics['sharpe_ratio'] + 2) * 100 / 6)))
        
        # Win Rate (already percentage, scale to 0-100)
        if wave1_metrics['win_rate'] != 'N/A' and wave2_metrics['win_rate'] != 'N/A':
            categories.append('Win Rate')
            wave1_values.append(wave1_metrics['win_rate'] * 100)
            wave2_values.append(wave2_metrics['win_rate'] * 100)
        
        # Returns (normalize to 0-100, assuming -20% to +20% range)
        if wave1_metrics['cumulative_return'] != 'N/A' and wave2_metrics['cumulative_return'] != 'N/A':
            categories.append('Returns')
            wave1_values.append(min(100, max(0, (wave1_metrics['cumulative_return'] + 0.2) * 100 / 0.4)))
            wave2_values.append(min(100, max(0, (wave2_metrics['cumulative_return'] + 0.2) * 100 / 0.4)))
        
        # Risk Control (inverse of max drawdown, normalize)
        if wave1_metrics['max_drawdown'] != 'N/A' and wave2_metrics['max_drawdown'] != 'N/A':
            categories.append('Risk Control')
            # Less negative drawdown is better, convert to 0-100 scale
            wave1_values.append(min(100, max(0, (1 + wave1_metrics['max_drawdown']) * 100)))
            wave2_values.append(min(100, max(0, (1 + wave2_metrics['max_drawdown']) * 100)))
        
        if len(categories) == 0:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=wave1_values,
            theta=categories,
            fill='toself',
            name=wave1_metrics['name'],
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=wave2_values,
            theta=categories,
            fill='toself',
            name=wave2_metrics['name'],
            line=dict(color='red')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Multi-Dimensional Comparison",
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception:
        return None


def create_correlation_heatmap(wave1_data, wave2_data, wave1_name, wave2_name):
    """
    Create a correlation matrix heatmap for two waves.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if wave1_data is None or wave2_data is None:
            return None
        
        if 'date' not in wave1_data.columns or 'date' not in wave2_data.columns:
            return None
        
        if 'portfolio_return' not in wave1_data.columns or 'portfolio_return' not in wave2_data.columns:
            return None
        
        # Merge data on date
        wave1_returns = wave1_data[['date', 'portfolio_return']].rename(columns={'portfolio_return': wave1_name})
        wave2_returns = wave2_data[['date', 'portfolio_return']].rename(columns={'portfolio_return': wave2_name})
        
        merged = pd.merge(wave1_returns, wave2_returns, on='date', how='inner')
        
        if len(merged) < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = merged[[wave1_name, wave2_name]].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 16},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Return Correlation Matrix",
            height=400,
            width=500
        )
        
        return fig
        
    except Exception:
        return None


# ============================================================================
# SECTION 5: DATA PROCESSING FUNCTIONS
# ============================================================================

def get_mission_control_data():
    """
    Retrieve Mission Control metrics from available data.
    Returns dict with all metrics, using 'unknown' for unavailable data.
    Enhanced with additional system health indicators.
    """
    mc_data = {
        'market_regime': 'unknown',
        'vix_gate_status': 'unknown',
        'alpha_today': 'unknown',
        'alpha_30day': 'unknown',
        'wavescore_leader': 'unknown',
        'wavescore_leader_score': 'unknown',
        'data_freshness': 'unknown',
        'data_age_days': None,
        'total_waves': 0,
        'active_waves': 0,
        'system_status': 'unknown'
    }
    
    try:
        df = safe_load_wave_history()
        
        if df is None:
            mc_data['system_status'] = 'Data Unavailable'
            return mc_data
        
        # Get latest date and data freshness
        latest_date = df['date'].max()
        mc_data['data_freshness'] = latest_date.strftime('%Y-%m-%d')
        
        # Calculate data age in days
        age_days = (datetime.now() - latest_date).days
        mc_data['data_age_days'] = age_days
        
        # Count waves
        if 'wave' in df.columns:
            mc_data['total_waves'] = df['wave'].nunique()
            
            # Count active waves (with recent data)
            recent_data = df[df['date'] >= (latest_date - timedelta(days=7))]
            mc_data['active_waves'] = recent_data['wave'].nunique()
        
        # System status based on data age
        if age_days <= 1:
            mc_data['system_status'] = 'Excellent'
        elif age_days <= 3:
            mc_data['system_status'] = 'Good'
        elif age_days <= 7:
            mc_data['system_status'] = 'Fair'
        else:
            mc_data['system_status'] = 'Stale'
        
        # Calculate Market Regime based on recent returns
        recent_days = 5
        recent_data = df[df['date'] >= (latest_date - timedelta(days=recent_days))]
        
        if 'portfolio_return' in df.columns and len(recent_data) > 0:
            avg_return = recent_data['portfolio_return'].mean()
            volatility = recent_data['portfolio_return'].std()
            
            # Enhanced regime detection
            if avg_return > 0.005:
                if volatility < 0.015:
                    mc_data['market_regime'] = 'Risk-On (Stable)'
                else:
                    mc_data['market_regime'] = 'Risk-On (Volatile)'
            elif avg_return < -0.005:
                if volatility > 0.02:
                    mc_data['market_regime'] = 'Risk-Off (Volatile)'
                else:
                    mc_data['market_regime'] = 'Risk-Off (Stable)'
            else:
                mc_data['market_regime'] = 'Neutral'
        
        # VIX Gate Status estimation (based on volatility)
        if 'portfolio_return' in df.columns and len(recent_data) > 0:
            recent_vol = recent_data['portfolio_return'].std() * np.sqrt(252) * 100
            if recent_vol < 15:
                mc_data['vix_gate_status'] = 'GREEN (Low Vol)'
            elif recent_vol < 25:
                mc_data['vix_gate_status'] = 'YELLOW (Med Vol)'
            else:
                mc_data['vix_gate_status'] = 'RED (High Vol)'
        
        # Calculate Alpha metrics
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
            
            # Today's alpha
            today_data = df[df['date'] == latest_date]
            if len(today_data) > 0:
                alpha_today = today_data['alpha'].mean()
                mc_data['alpha_today'] = f"{alpha_today*100:.2f}%"
            
            # 30-day cumulative alpha
            days_30_ago = latest_date - timedelta(days=30)
            last_30_days = df[df['date'] >= days_30_ago]
            if len(last_30_days) > 0:
                alpha_30day = last_30_days['alpha'].sum()
                mc_data['alpha_30day'] = f"{alpha_30day*100:.2f}%"
        
        # WaveScore Leader
        if 'wave' in df.columns and 'alpha' in df.columns:
            days_30_ago = latest_date - timedelta(days=30)
            last_30_days = df[df['date'] >= days_30_ago]
            
            wave_performance = last_30_days.groupby('wave')['alpha'].sum().sort_values(ascending=False)
            
            if len(wave_performance) > 0:
                top_wave = wave_performance.index[0]
                top_alpha = wave_performance.iloc[0]
                
                wavescore = min(100, max(0, (top_alpha * 1000) + 50))
                
                mc_data['wavescore_leader'] = top_wave
                mc_data['wavescore_leader_score'] = f"{wavescore:.1f}"
        
    except Exception:
        pass
    
    return mc_data


def get_wavescore_leaderboard():
    """
    Get top 10 waves by WaveScore (30-day cumulative alpha).
    Returns a DataFrame with wave names and scores, or None if unavailable.
    """
    try:
        df = get_wave_data_filtered(wave_name=None, days=30)
        
        if df is None:
            return None
        
        if 'wave' not in df.columns:
            return None
        
        # Calculate alpha
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
        else:
            return None
        
        # Calculate WaveScore for each wave
        wave_scores = []
        for wave in df['wave'].unique():
            wave_data = df[df['wave'] == wave]
            score = calculate_wavescore(wave_data)
            wave_scores.append({'Wave': wave, 'WaveScore': score})
        
        if len(wave_scores) == 0:
            return None
        
        # Create DataFrame and sort by score
        leaderboard_df = pd.DataFrame(wave_scores)
        leaderboard_df = leaderboard_df.sort_values('WaveScore', ascending=False).head(10)
        leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)
        leaderboard_df = leaderboard_df[['Rank', 'Wave', 'WaveScore']]
        
        return leaderboard_df
        
    except Exception:
        return None


def get_biggest_movers():
    """
    Get biggest month-over-month WaveScore changes.
    Returns a DataFrame with wave names and score changes, or None if unavailable.
    """
    try:
        df = safe_load_wave_history()
        
        if df is None:
            return None
        
        if 'wave' not in df.columns or 'date' not in df.columns:
            return None
        
        # Calculate alpha
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
        else:
            return None
        
        latest_date = df['date'].max()
        
        # Get last 30 days (current period)
        days_30_ago = latest_date - timedelta(days=30)
        current_period = df[df['date'] >= days_30_ago]
        
        # Get previous 30 days (30-60 days ago)
        days_60_ago = latest_date - timedelta(days=60)
        previous_period = df[(df['date'] >= days_60_ago) & (df['date'] < days_30_ago)]
        
        if len(current_period) == 0 or len(previous_period) == 0:
            return None
        
        # Calculate WaveScores for both periods
        movers = []
        waves = set(current_period['wave'].unique()) & set(previous_period['wave'].unique())
        
        for wave in waves:
            current_data = current_period[current_period['wave'] == wave]
            previous_data = previous_period[previous_period['wave'] == wave]
            
            current_score = calculate_wavescore(current_data)
            previous_score = calculate_wavescore(previous_data)
            
            change = current_score - previous_score
            
            movers.append({
                'Wave': wave,
                'Previous': previous_score,
                'Current': current_score,
                'Change': change
            })
        
        if len(movers) == 0:
            return None
        
        # Create DataFrame and sort by absolute change
        movers_df = pd.DataFrame(movers)
        movers_df = movers_df.sort_values('Change', ascending=False, key=abs).head(10)
        
        return movers_df
        
    except Exception:
        return None


def get_system_alerts():
    """
    Generate system alerts based on data quality and risk signals.
    Returns a list of alert dictionaries with severity and message.
    """
    alerts = []
    
    try:
        df = safe_load_wave_history()
        
        if df is None:
            alerts.append({
                'severity': 'error',
                'message': 'Wave history data file not found or invalid'
            })
            return alerts
        
        if len(df) == 0:
            alerts.append({
                'severity': 'error',
                'message': 'Wave history data is empty'
            })
            return alerts
        
        latest_date = df['date'].max()
        
        # Data freshness alert
        age_days = (datetime.now() - latest_date).days
        if age_days > 7:
            alerts.append({
                'severity': 'warning',
                'message': f'Data is {age_days} days old - consider updating'
            })
        elif age_days > 2:
            alerts.append({
                'severity': 'info',
                'message': f'Data is {age_days} days old'
            })
        
        # Calculate alpha for remaining checks
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
            
            # Check for data gaps
            date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
            actual_dates = df['date'].unique()
            missing_dates = len(date_range) - len(actual_dates)
            
            if missing_dates > 10:
                alerts.append({
                    'severity': 'warning',
                    'message': f'{missing_dates} days of data missing in date range'
                })
            
            # Check for high volatility
            last_30_days = df[df['date'] >= (latest_date - timedelta(days=30))]
            if len(last_30_days) > 0:
                volatility = last_30_days['alpha'].std()
                if volatility > 0.05:  # 5% daily volatility threshold
                    alerts.append({
                        'severity': 'warning',
                        'message': f'High volatility detected: {volatility*100:.2f}% (potential drawdown risk)'
                    })
            
            # Check for negative cumulative alpha
            if len(last_30_days) > 0:
                last_30_days_cumulative_alpha = last_30_days['alpha'].sum()
                if last_30_days_cumulative_alpha < -0.05:  # -5% cumulative underperformance
                    alerts.append({
                        'severity': 'warning',
                        'message': f'Significant underperformance: {last_30_days_cumulative_alpha*100:.2f}% cumulative alpha'
                    })
        
        # If no alerts, add an all-clear message
        if len(alerts) == 0:
            alerts.append({
                'severity': 'success',
                'message': 'All systems operational'
            })
        
    except Exception as e:
        alerts.append({
            'severity': 'error',
            'message': f'Error checking system status: {str(e)}'
        })
    
    return alerts


def generate_wave_narrative(wave_name, wave_data):
    """
    Generate an institutional narrative for a Wave.
    
    Args:
        wave_name: Name of the wave
        wave_data: DataFrame containing wave performance data
        
    Returns:
        String containing the narrative
    """
    narrative_parts = []
    
    # Header
    narrative_parts.append(f"# Institutional Narrative: {wave_name}")
    narrative_parts.append("")
    
    try:
        if wave_data is None or len(wave_data) == 0:
            narrative_parts.append("**Data Status:** Insufficient data available for analysis.")
            return "\n".join(narrative_parts)
        
        # Calculate key metrics
        wave_data = wave_data.copy()
        
        if 'portfolio_return' not in wave_data.columns or 'benchmark_return' not in wave_data.columns:
            narrative_parts.append("**Data Status:** Required return data unavailable.")
            return "\n".join(narrative_parts)
        
        wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
        
        # What Happened section
        narrative_parts.append("## What Happened")
        
        start_date = wave_data['date'].min().strftime("%Y-%m-%d")
        end_date = wave_data['date'].max().strftime("%Y-%m-%d")
        num_days = len(wave_data)
        
        narrative_parts.append(f"Over the period from {start_date} to {end_date} ({num_days} trading days), {wave_name} generated the following performance:")
        
        cumulative_return = wave_data['portfolio_return'].sum()
        cumulative_benchmark = wave_data['benchmark_return'].sum()
        cumulative_alpha = wave_data['alpha'].sum()
        
        narrative_parts.append(f"- Portfolio Return: {cumulative_return*100:.2f}%")
        narrative_parts.append(f"- Benchmark Return: {cumulative_benchmark*100:.2f}%")
        narrative_parts.append(f"- Alpha Generated: {cumulative_alpha*100:.2f}%")
        narrative_parts.append("")
        
        # Drivers of Alpha section
        narrative_parts.append("## Drivers of Alpha")
        
        avg_daily_alpha = wave_data['alpha'].mean()
        positive_days = len(wave_data[wave_data['alpha'] > 0])
        total_days = len(wave_data)
        win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
        
        narrative_parts.append(f"The wave demonstrated an average daily alpha of {avg_daily_alpha*100:.4f}%, with positive alpha on {positive_days} of {total_days} trading days ({win_rate:.1f}% win rate).")
        
        # Identify best and worst days
        best_day = wave_data.loc[wave_data['alpha'].idxmax()]
        worst_day = wave_data.loc[wave_data['alpha'].idxmin()]
        
        narrative_parts.append(f"- Best performance: {best_day['alpha']*100:.2f}% alpha on {best_day['date'].strftime('%Y-%m-%d')}")
        narrative_parts.append(f"- Worst performance: {worst_day['alpha']*100:.2f}% alpha on {worst_day['date'].strftime('%Y-%m-%d')}")
        narrative_parts.append("")
        
        # Risk Posture section
        narrative_parts.append("## Overall Risk Posture")
        
        volatility = wave_data['portfolio_return'].std()
        sharpe_ratio = (avg_daily_alpha / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        narrative_parts.append(f"Daily return volatility: {volatility*100:.2f}%")
        narrative_parts.append(f"Annualized Sharpe Ratio (estimated): {sharpe_ratio:.2f}")
        
        # Calculate drawdown
        cumulative_returns = (1 + wave_data['portfolio_return']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        narrative_parts.append(f"Maximum drawdown: {max_drawdown*100:.2f}%")
        
        # Risk assessment
        if volatility < 0.01:
            risk_level = "LOW"
        elif volatility < 0.02:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        narrative_parts.append(f"Risk Level: {risk_level}")
        narrative_parts.append("")
        
        # Recommended Action section
        narrative_parts.append("## Recommended Action Language")
        
        if cumulative_alpha > 0.05:  # >5% cumulative alpha
            if risk_level == "LOW":
                action = "STRONG BUY - Wave is generating significant alpha with controlled risk. Consider increasing allocation."
            elif risk_level == "MODERATE":
                action = "BUY - Wave is generating strong alpha but monitor volatility. Maintain or slightly increase position."
            else:
                action = "HOLD - Wave is generating alpha but with elevated volatility. Monitor closely before increasing exposure."
        elif cumulative_alpha > 0.01:  # >1% cumulative alpha
            action = "HOLD - Wave is generating positive but modest alpha. Maintain current allocation and monitor."
        elif cumulative_alpha > -0.01:  # Between -1% and 1%
            action = "NEUTRAL - Wave is performing in line with benchmark. Review strategy and consider alternatives."
        else:  # <-1% cumulative alpha
            action = "REDUCE - Wave is underperforming benchmark. Consider reducing allocation or investigating root causes."
        
        narrative_parts.append(action)
        narrative_parts.append("")
        
        # Data Quality Note
        narrative_parts.append("---")
        narrative_parts.append(f"*Analysis based on {num_days} days of data from {start_date} to {end_date}.*")
        
    except Exception as e:
        narrative_parts.append(f"**Error generating narrative:** {str(e)}")
        narrative_parts.append("**Data Status:** Some required fields may be unavailable.")
    
    return "\n".join(narrative_parts)


def get_wave_comparison_data(wave1_name, wave2_name):
    """
    Retrieve comparison data for two waves.
    Returns a dictionary with comparison metrics or None if unavailable.
    """
    try:
        wave1_data = get_wave_data_filtered(wave_name=wave1_name, days=30)
        wave2_data = get_wave_data_filtered(wave_name=wave2_name, days=30)
        
        if wave1_data is None or wave2_data is None:
            return None
        
        # Calculate metrics for Wave 1
        wave1_metrics = calculate_wave_metrics(wave1_data)
        wave1_metrics['name'] = wave1_name
        
        # Calculate metrics for Wave 2
        wave2_metrics = calculate_wave_metrics(wave2_data)
        wave2_metrics['name'] = wave2_name
        
        # Calculate correlation
        correlation = calculate_wave_correlation(wave1_data, wave2_data)
        
        return {
            'wave1': wave1_metrics,
            'wave2': wave2_metrics,
            'correlation': correlation
        }
        
    except Exception:
        return None


# ============================================================================
# SECTION 6: REUSABLE UI COMPONENTS
# ============================================================================

def render_mission_control():
    """
    Render the Mission Control summary strip at the top of the page.
    Enhanced with additional system metrics and visual indicators.
    """
    st.markdown("### 🎯 Mission Control - Executive Layer v2")
    
    mc_data = get_mission_control_data()
    
    # Top row: Primary metrics (5 columns)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        regime_value = mc_data['market_regime']
        # Add emoji indicators
        if 'Risk-On' in regime_value:
            regime_display = f"📈 {regime_value}"
        elif 'Risk-Off' in regime_value:
            regime_display = f"📉 {regime_value}"
        else:
            regime_display = f"➖ {regime_value}"
        
        st.metric(
            label="Market Regime",
            value=regime_display,
            help="Current market regime based on recent portfolio performance and volatility"
        )
    
    with col2:
        vix_value = mc_data['vix_gate_status']
        # Add color indicators
        if 'GREEN' in vix_value:
            vix_display = f"🟢 {vix_value}"
        elif 'YELLOW' in vix_value:
            vix_display = f"🟡 {vix_value}"
        elif 'RED' in vix_value:
            vix_display = f"🔴 {vix_value}"
        else:
            vix_display = vix_value
        
        st.metric(
            label="VIX Gate Status",
            value=vix_display,
            help="Volatility-based risk gate (Green=Low, Yellow=Medium, Red=High)"
        )
    
    with col3:
        st.markdown("**Alpha Captured**")
        alpha_today_str = mc_data['alpha_today']
        alpha_30day_str = mc_data['alpha_30day']
        
        # Add color coding if possible
        try:
            if alpha_today_str != 'unknown' and '%' in alpha_today_str:
                alpha_today_val = float(alpha_today_str.replace('%', ''))
                if alpha_today_val > 0:
                    alpha_today_str = f"🟢 {alpha_today_str}"
                elif alpha_today_val < 0:
                    alpha_today_str = f"🔴 {alpha_today_str}"
        except:
            pass
        
        st.write(f"Latest: {alpha_today_str}")
        st.write(f"30-Day: {alpha_30day_str}")
    
    with col4:
        st.markdown("**WaveScore Leader**")
        if mc_data['wavescore_leader'] != 'unknown':
            # Truncate long wave names
            wave_name_display = mc_data['wavescore_leader']
            if len(wave_name_display) > 20:
                wave_name_display = wave_name_display[:20] + "..."
            st.write(f"🏆 {wave_name_display}")
            st.write(f"Score: {mc_data['wavescore_leader_score']}")
        else:
            st.write("No data")
    
    with col5:
        system_status = mc_data['system_status']
        freshness_value = mc_data['data_freshness']
        
        # Add status indicator
        if system_status == 'Excellent':
            status_display = f"✅ {system_status}"
        elif system_status == 'Good':
            status_display = f"🟢 {system_status}"
        elif system_status == 'Fair':
            status_display = f"🟡 {system_status}"
        else:
            status_display = f"🔴 {system_status}"
        
        st.metric(
            label="System Health",
            value=status_display,
            help=f"Data freshness: {freshness_value}"
        )
        st.caption(f"Data: {freshness_value}")
    
    # Bottom row: Secondary metrics (3 columns)
    st.markdown("---")
    
    sec_col1, sec_col2, sec_col3 = st.columns(3)
    
    with sec_col1:
        st.metric(
            label="Total Waves",
            value=mc_data.get('total_waves', 0),
            help="Total number of waves in the system"
        )
    
    with sec_col2:
        st.metric(
            label="Active Waves",
            value=mc_data.get('active_waves', 0),
            help="Waves with data in the last 7 days"
        )
    
    with sec_col3:
        data_age = mc_data.get('data_age_days')
        if data_age is not None:
            age_display = f"{data_age} day{'s' if data_age != 1 else ''}"
            if data_age == 0:
                age_display = "Today"
        else:
            age_display = "Unknown"
        
        st.metric(
            label="Data Age",
            value=age_display,
            help="Time since last data update"
        )
    
    st.divider()


def render_sidebar_info():
    """Render sidebar information including build info and menu."""
    st.sidebar.title("Risk Lab")
    st.sidebar.write("Advanced risk analytics and monitoring tools for institutional portfolio management.")
    
    st.sidebar.title("Correlation Matrix")
    st.sidebar.write("Cross-asset correlation analysis for portfolio diversification insights.")
    
    st.sidebar.title("Rolling Alpha / Volatility")
    st.sidebar.write("Time-series analysis of alpha generation and volatility patterns.")
    
    st.sidebar.title("Drawdown Monitor")
    st.sidebar.write("Real-time tracking of portfolio drawdowns and recovery metrics.")
    
    # Build Information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Build Information")
    
    version_label = "Console v1.0"
    commit_hash = get_git_commit_hash()
    branch_name = get_git_branch_name()
    deploy_time = get_deploy_timestamp()
    data_timestamp = get_latest_data_timestamp()
    
    st.sidebar.text(f"Version: {version_label}")
    st.sidebar.text(f"Commit: {commit_hash}")
    st.sidebar.text(f"Branch: {branch_name}")
    st.sidebar.text(f"Deployed: {deploy_time}")
    st.sidebar.text(f"Data as of: {data_timestamp}")


# ============================================================================
# SECTION 7: TAB RENDER FUNCTIONS
# ============================================================================

def render_executive_tab():
    """
    Render the Executive tab with enhanced visualizations.
    Includes: Leaderboard, Movers, Alerts, and Performance Charts.
    """
    st.header("📊 Executive Dashboard - Command Center")
    
    # WaveScore Command Center Section
    st.markdown("### 🎯 WaveScore Command Center")
    
    # Create two columns for Leaderboard and Movers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top Performers")
        
        leaderboard = get_wavescore_leaderboard()
        if leaderboard is not None and len(leaderboard) > 0:
            # Show interactive chart
            chart = create_wavescore_bar_chart(leaderboard)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            
            # Also show data table
            with st.expander("View Data Table"):
                leaderboard_display = leaderboard.copy()
                leaderboard_display['WaveScore'] = leaderboard_display['WaveScore'].apply(lambda x: f"{x:.1f}")
                st.dataframe(leaderboard_display, use_container_width=True, hide_index=True)
        else:
            st.info("Data unavailable")
    
    with col2:
        st.markdown("#### 📈 Biggest Movers")
        
        movers = get_biggest_movers()
        if movers is not None and len(movers) > 0:
            # Show interactive chart
            chart = create_movers_chart(movers)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            
            # Also show data table
            with st.expander("View Data Table"):
                movers_display = movers.copy()
                movers_display['Previous'] = movers_display['Previous'].apply(lambda x: f"{x:.1f}")
                movers_display['Current'] = movers_display['Current'].apply(lambda x: f"{x:.1f}")
                movers_display['Change'] = movers_display['Change'].apply(
                    lambda x: f"{'↑' if x > 0 else '↓'} {abs(x):.1f}"
                )
                st.dataframe(movers_display, use_container_width=True, hide_index=True)
        else:
            st.info("Data unavailable")
    
    st.divider()
    
    # Performance Deep Dive Section
    st.markdown("### 📊 Performance Deep Dive")
    st.write("Select a wave to view detailed performance charts")
    
    try:
        waves = get_available_waves()
        
        if len(waves) > 0:
            selected_wave = st.selectbox(
                "Select Wave for Analysis",
                options=waves,
                help="Choose a wave to view detailed performance metrics and charts"
            )
            
            if selected_wave:
                wave_data = get_wave_data_filtered(wave_name=selected_wave, days=30)
                
                if wave_data is not None:
                    # Calculate and display key metrics in columns
                    metrics = calculate_wave_metrics(wave_data)
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        cum_ret = metrics.get('cumulative_return', 'N/A')
                        if cum_ret != 'N/A':
                            st.metric("30-Day Return", f"{cum_ret*100:.2f}%")
                        else:
                            st.metric("30-Day Return", "N/A")
                    
                    with metric_col2:
                        cum_alpha = metrics.get('cumulative_alpha', 'N/A')
                        if cum_alpha != 'N/A':
                            st.metric("Cumulative Alpha", f"{cum_alpha*100:.2f}%")
                        else:
                            st.metric("Cumulative Alpha", "N/A")
                    
                    with metric_col3:
                        wavescore = metrics.get('wavescore', 'N/A')
                        if wavescore != 'N/A':
                            st.metric("WaveScore", f"{wavescore:.1f}")
                        else:
                            st.metric("WaveScore", "N/A")
                    
                    with metric_col4:
                        sharpe = metrics.get('sharpe_ratio', 'N/A')
                        if sharpe != 'N/A':
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        else:
                            st.metric("Sharpe Ratio", "N/A")
                    
                    # Display performance chart
                    chart = create_wave_performance_chart(wave_data, selected_wave)
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.info("Unable to generate performance chart")
                else:
                    st.warning(f"No data available for {selected_wave}")
        else:
            st.warning("No waves available for analysis")
    
    except Exception as e:
        st.error(f"Error in performance deep dive: {str(e)}")
    
    st.divider()
    
    # System Alerts Section
    st.markdown("### 🚨 System Alerts & Risk Signals")
    
    alerts = get_system_alerts()
    
    if alerts:
        # Group alerts by severity
        errors = [a for a in alerts if a.get('severity') == 'error']
        warnings = [a for a in alerts if a.get('severity') == 'warning']
        successes = [a for a in alerts if a.get('severity') == 'success']
        infos = [a for a in alerts if a.get('severity') == 'info']
        
        # Display in order of importance
        for alert in errors:
            st.error(f"❌ {alert.get('message', '')}")
        
        for alert in warnings:
            st.warning(f"⚠️ {alert.get('message', '')}")
        
        for alert in successes:
            st.success(f"✅ {alert.get('message', '')}")
        
        for alert in infos:
            st.info(f"ℹ️ {alert.get('message', '')}")
    else:
        st.info("No alerts at this time")


def render_vector_explain_panel():
    """
    Render the Vector Explain panel for generating Wave narratives.
    Enhanced with performance visualization.
    """
    st.subheader("📝 Vector Explain - Narrative Generator")
    st.write("Generate comprehensive institutional narratives with performance visualizations")
    
    try:
        waves = get_available_waves()
        
        if len(waves) == 0:
            st.warning("No wave data available")
            return
        
        # Wave selector
        selected_wave = st.selectbox(
            "Select Wave",
            options=waves,
            key="narrative_wave_selector",
            help="Choose a wave to generate an institutional narrative"
        )
        
        # Time period selector
        time_period = st.selectbox(
            "Analysis Period",
            options=[30, 60, 90],
            format_func=lambda x: f"{x} days",
            help="Select the time period for analysis"
        )
        
        if st.button("Generate Narrative", type="primary"):
            with st.spinner("Generating comprehensive narrative and analysis..."):
                wave_data = get_wave_data_filtered(wave_name=selected_wave, days=time_period)
                
                if wave_data is not None:
                    # Generate narrative
                    narrative = generate_wave_narrative(selected_wave, wave_data)
                    
                    # Store in session state
                    st.session_state['current_narrative'] = narrative
                    st.session_state['current_narrative_wave'] = selected_wave
                    st.session_state['current_narrative_data'] = wave_data
                else:
                    st.error(f"Unable to load data for {selected_wave}")
                    return
        
        # Display narrative and visualization if available
        if 'current_narrative' in st.session_state:
            st.divider()
            
            # Display performance chart first
            if 'current_narrative_data' in st.session_state and 'current_narrative_wave' in st.session_state:
                chart = create_wave_performance_chart(
                    st.session_state['current_narrative_data'],
                    st.session_state['current_narrative_wave']
                )
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
            
            st.divider()
            
            # Display narrative
            st.markdown(st.session_state['current_narrative'])
            
            # Export options
            st.divider()
            st.markdown("### 📤 Export Options")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 View as Text", use_container_width=True):
                    st.code(st.session_state['current_narrative'], language=None)
                    st.success("Narrative displayed above - use your browser to copy")
            
            with col2:
                # Download button
                st.download_button(
                    label="💾 Download Narrative",
                    data=st.session_state['current_narrative'],
                    file_name=f"wave_narrative_{st.session_state.get('current_narrative_wave', 'wave')}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"Error rendering Vector Explain panel: {str(e)}")


def render_compare_waves_panel():
    """
    Render the Compare Waves panel for head-to-head wave comparison.
    Enhanced with radar chart and correlation visualization.
    """
    st.subheader("⚖️ Compare Waves - Head-to-Head Analysis")
    st.write("Select two waves for comprehensive side-by-side performance comparison")
    
    try:
        waves = get_available_waves()
        
        if len(waves) < 2:
            st.warning("At least two waves required for comparison")
            return
        
        # Wave selectors
        col1, col2 = st.columns(2)
        
        with col1:
            wave1 = st.selectbox(
                "Wave 1",
                options=waves,
                key="compare_wave1",
                help="Select first wave for comparison"
            )
        
        with col2:
            # Filter out wave1 from wave2 options
            wave2_options = [w for w in waves if w != wave1]
            wave2 = st.selectbox(
                "Wave 2",
                options=wave2_options,
                key="compare_wave2",
                help="Select second wave for comparison"
            )
        
        if st.button("🔍 Compare Waves", type="primary"):
            with st.spinner("Generating comprehensive comparison analysis..."):
                comparison_data = get_wave_comparison_data(wave1, wave2)
                
                if comparison_data is None:
                    st.error("Unable to generate comparison - data unavailable")
                    return
                
                # Store comparison data and raw wave data
                st.session_state['comparison_data'] = comparison_data
                st.session_state['comparison_wave1_data'] = get_wave_data_filtered(wave_name=wave1, days=30)
                st.session_state['comparison_wave2_data'] = get_wave_data_filtered(wave_name=wave2, days=30)
                st.session_state['comparison_wave1_name'] = wave1
                st.session_state['comparison_wave2_name'] = wave2
        
        # Display comparison if available
        if 'comparison_data' in st.session_state:
            comp = st.session_state['comparison_data']
            wave1_metrics = comp['wave1']
            wave2_metrics = comp['wave2']
            correlation = comp.get('correlation')
            
            st.divider()
            
            # Visual Comparison Section
            st.markdown("### 📊 Visual Comparison")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Radar chart
                radar_chart = create_comparison_radar_chart(wave1_metrics, wave2_metrics)
                if radar_chart is not None:
                    st.plotly_chart(radar_chart, use_container_width=True)
                else:
                    st.info("Radar chart unavailable")
            
            with viz_col2:
                # Correlation heatmap
                if 'comparison_wave1_data' in st.session_state and 'comparison_wave2_data' in st.session_state:
                    heatmap = create_correlation_heatmap(
                        st.session_state['comparison_wave1_data'],
                        st.session_state['comparison_wave2_data'],
                        st.session_state.get('comparison_wave1_name', 'Wave 1'),
                        st.session_state.get('comparison_wave2_name', 'Wave 2')
                    )
                    if heatmap is not None:
                        st.plotly_chart(heatmap, use_container_width=True)
                    else:
                        st.info("Correlation matrix unavailable")
            
            st.divider()
            
            # Metrics Table
            st.markdown("### 📋 Detailed Metrics Comparison (30-Day)")
            
            # Helper function to format metric
            def format_metric(value, metric_type='percent'):
                if value == 'N/A':
                    return 'N/A'
                if metric_type == 'percent':
                    return f"{value*100:.2f}%"
                elif metric_type == 'score':
                    return f"{value:.1f}"
                elif metric_type == 'ratio':
                    return f"{value:.2f}"
                else:
                    return f"{value:.4f}"
            
            # Build comparison table
            comparison_rows = [
                {
                    'Metric': 'Cumulative Return',
                    wave1_metrics['name']: format_metric(wave1_metrics['cumulative_return']),
                    wave2_metrics['name']: format_metric(wave2_metrics['cumulative_return']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('cumulative_return', 0) > wave2_metrics.get('cumulative_return', 0) else wave2_metrics['name']
                },
                {
                    'Metric': 'Cumulative Alpha',
                    wave1_metrics['name']: format_metric(wave1_metrics['cumulative_alpha']),
                    wave2_metrics['name']: format_metric(wave2_metrics['cumulative_alpha']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('cumulative_alpha', 0) > wave2_metrics.get('cumulative_alpha', 0) else wave2_metrics['name']
                },
                {
                    'Metric': 'Volatility',
                    wave1_metrics['name']: format_metric(wave1_metrics['volatility']),
                    wave2_metrics['name']: format_metric(wave2_metrics['volatility']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('volatility', float('inf')) < wave2_metrics.get('volatility', float('inf')) else wave2_metrics['name']
                },
                {
                    'Metric': 'Max Drawdown',
                    wave1_metrics['name']: format_metric(wave1_metrics['max_drawdown']),
                    wave2_metrics['name']: format_metric(wave2_metrics['max_drawdown']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('max_drawdown', float('-inf')) > wave2_metrics.get('max_drawdown', float('-inf')) else wave2_metrics['name']
                },
                {
                    'Metric': 'WaveScore',
                    wave1_metrics['name']: format_metric(wave1_metrics['wavescore'], 'score'),
                    wave2_metrics['name']: format_metric(wave2_metrics['wavescore'], 'score'),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('wavescore', 0) > wave2_metrics.get('wavescore', 0) else wave2_metrics['name']
                },
                {
                    'Metric': 'Sharpe Ratio',
                    wave1_metrics['name']: format_metric(wave1_metrics['sharpe_ratio'], 'ratio'),
                    wave2_metrics['name']: format_metric(wave2_metrics['sharpe_ratio'], 'ratio'),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('sharpe_ratio', float('-inf')) > wave2_metrics.get('sharpe_ratio', float('-inf')) else wave2_metrics['name']
                },
                {
                    'Metric': 'Win Rate',
                    wave1_metrics['name']: format_metric(wave1_metrics['win_rate']),
                    wave2_metrics['name']: format_metric(wave2_metrics['win_rate']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('win_rate', 0) > wave2_metrics.get('win_rate', 0) else wave2_metrics['name']
                }
            ]
            
            # Display table
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Summary Analysis
            st.markdown("### 🏆 Winner Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Overall Winner")
                winner, notes = determine_winner(wave1_metrics, wave2_metrics)
                
                if winner:
                    if winner == "TIE":
                        st.success(f"**{winner}**")
                    else:
                        st.success(f"🏆 **{winner}**")
                    st.write(notes)
                else:
                    st.info(notes)
            
            with col2:
                st.markdown("#### Correlation Analysis")
                if correlation is not None and correlation != 'N/A':
                    st.metric(
                        label="Return Correlation",
                        value=f"{correlation:.3f}",
                        help="Correlation between daily returns (-1 to 1)"
                    )
                    
                    # Add interpretation
                    if abs(correlation) > 0.7:
                        corr_note = "Strong correlation"
                    elif abs(correlation) > 0.3:
                        corr_note = "Moderate correlation"
                    else:
                        corr_note = "Weak correlation"
                    
                    if correlation > 0:
                        corr_note += " (positive)"
                        corr_interpretation = "Waves tend to move together"
                    elif correlation < 0:
                        corr_note += " (negative)"
                        corr_interpretation = "Waves tend to move in opposite directions"
                    else:
                        corr_interpretation = "No clear relationship"
                    
                    st.info(corr_note)
                    st.write(corr_interpretation)
                else:
                    st.info("Correlation data unavailable")
            
            # Additional insights
            st.divider()
            st.markdown("### 📝 Key Insights")
            
            insights = []
            
            # Performance insight
            if wave1_metrics.get('cumulative_return', 0) != 'N/A' and wave2_metrics.get('cumulative_return', 0) != 'N/A':
                ret_diff = abs(wave1_metrics['cumulative_return'] - wave2_metrics['cumulative_return'])
                if ret_diff > 0.10:  # 10% difference
                    insights.append(f"⚠️ **Significant return divergence**: {ret_diff*100:.1f}% difference in cumulative returns")
            
            # Risk insight
            if wave1_metrics.get('max_drawdown', 0) != 'N/A' and wave2_metrics.get('max_drawdown', 0) != 'N/A':
                dd_diff = abs(wave1_metrics['max_drawdown'] - wave2_metrics['max_drawdown'])
                if dd_diff > 0.05:  # 5% difference in drawdown
                    insights.append(f"⚠️ **Risk profile differs**: {dd_diff*100:.1f}% difference in max drawdown")
            
            # Diversification insight
            if correlation is not None and abs(correlation) < 0.3:
                insights.append(f"✅ **Good diversification potential**: Low correlation ({correlation:.2f}) suggests complementary performance")
            elif correlation is not None and abs(correlation) > 0.8:
                insights.append(f"ℹ️ **High correlation**: Waves move similarly ({correlation:.2f}), limited diversification benefit")
            
            if insights:
                for insight in insights:
                    st.markdown(insight)
            else:
                st.write("No significant insights detected")
            
            # Export comparison
            st.divider()
            st.markdown("### 📤 Export Comparison")
            
            # Create export text
            export_text = f"""# Wave Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Waves Compared
- Wave 1: {wave1_metrics['name']}
- Wave 2: {wave2_metrics['name']}

## Performance Metrics (30-Day)

| Metric | {wave1_metrics['name']} | {wave2_metrics['name']} |
|--------|----------|----------|
| Cumulative Return | {format_metric(wave1_metrics['cumulative_return'])} | {format_metric(wave2_metrics['cumulative_return'])} |
| Cumulative Alpha | {format_metric(wave1_metrics['cumulative_alpha'])} | {format_metric(wave2_metrics['cumulative_alpha'])} |
| WaveScore | {format_metric(wave1_metrics['wavescore'], 'score')} | {format_metric(wave2_metrics['wavescore'], 'score')} |
| Sharpe Ratio | {format_metric(wave1_metrics['sharpe_ratio'], 'ratio')} | {format_metric(wave2_metrics['sharpe_ratio'], 'ratio')} |
| Volatility | {format_metric(wave1_metrics['volatility'])} | {format_metric(wave2_metrics['volatility'])} |
| Max Drawdown | {format_metric(wave1_metrics['max_drawdown'])} | {format_metric(wave2_metrics['max_drawdown'])} |
| Win Rate | {format_metric(wave1_metrics['win_rate'])} | {format_metric(wave2_metrics['win_rate'])} |

## Winner
{winner}: {notes}

## Correlation
Return Correlation: {correlation if correlation is not None else 'N/A'}
"""
            
            st.download_button(
                label="💾 Download Comparison Report",
                data=export_text,
                file_name=f"wave_comparison_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"Error rendering Compare Waves panel: {str(e)}")
    
    try:
        waves = get_available_waves()
        
        if len(waves) < 2:
            st.warning("At least two waves required for comparison")
            return
        
        # Wave selectors
        col1, col2 = st.columns(2)
        
        with col1:
            wave1 = st.selectbox(
                "Wave 1",
                options=waves,
                key="compare_wave1",
                help="Select first wave for comparison"
            )
        
        with col2:
            # Filter out wave1 from wave2 options
            wave2_options = [w for w in waves if w != wave1]
            wave2 = st.selectbox(
                "Wave 2",
                options=wave2_options,
                key="compare_wave2",
                help="Select second wave for comparison"
            )
        
        if st.button("Compare Waves", type="primary"):
            with st.spinner("Generating comparison..."):
                comparison_data = get_wave_comparison_data(wave1, wave2)
                
                if comparison_data is None:
                    st.error("Unable to generate comparison - data unavailable")
                    return
                
                # Store in session state
                st.session_state['comparison_data'] = comparison_data
        
        # Display comparison if available
        if 'comparison_data' in st.session_state:
            comp = st.session_state['comparison_data']
            wave1_metrics = comp['wave1']
            wave2_metrics = comp['wave2']
            correlation = comp.get('correlation')
            
            st.divider()
            st.markdown("### 📊 Head-to-Head Comparison (30-Day)")
            
            # Helper function to format metric
            def format_metric(value, metric_type='percent'):
                if value == 'N/A':
                    return 'N/A'
                if metric_type == 'percent':
                    return f"{value*100:.2f}%"
                elif metric_type == 'score':
                    return f"{value:.1f}"
                elif metric_type == 'ratio':
                    return f"{value:.2f}"
                else:
                    return f"{value:.4f}"
            
            # Build comparison table
            comparison_rows = [
                {
                    'Metric': 'Cumulative Return',
                    wave1_metrics['name']: format_metric(wave1_metrics['cumulative_return']),
                    wave2_metrics['name']: format_metric(wave2_metrics['cumulative_return'])
                },
                {
                    'Metric': 'Cumulative Alpha',
                    wave1_metrics['name']: format_metric(wave1_metrics['cumulative_alpha']),
                    wave2_metrics['name']: format_metric(wave2_metrics['cumulative_alpha'])
                },
                {
                    'Metric': 'Volatility',
                    wave1_metrics['name']: format_metric(wave1_metrics['volatility']),
                    wave2_metrics['name']: format_metric(wave2_metrics['volatility'])
                },
                {
                    'Metric': 'Max Drawdown',
                    wave1_metrics['name']: format_metric(wave1_metrics['max_drawdown']),
                    wave2_metrics['name']: format_metric(wave2_metrics['max_drawdown'])
                },
                {
                    'Metric': 'WaveScore',
                    wave1_metrics['name']: format_metric(wave1_metrics['wavescore'], 'score'),
                    wave2_metrics['name']: format_metric(wave2_metrics['wavescore'], 'score')
                },
                {
                    'Metric': 'Sharpe Ratio',
                    wave1_metrics['name']: format_metric(wave1_metrics['sharpe_ratio'], 'ratio'),
                    wave2_metrics['name']: format_metric(wave2_metrics['sharpe_ratio'], 'ratio')
                },
                {
                    'Metric': 'Win Rate',
                    wave1_metrics['name']: format_metric(wave1_metrics['win_rate']),
                    wave2_metrics['name']: format_metric(wave2_metrics['win_rate'])
                }
            ]
            
            # Display table
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Correlation and Winner
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔗 Correlation")
                if correlation is not None and correlation != 'N/A':
                    st.metric(
                        label="Return Correlation",
                        value=f"{correlation:.3f}",
                        help="Correlation between daily returns (-1 to 1)"
                    )
                    
                    # Add interpretation
                    if abs(correlation) > 0.7:
                        corr_note = "Strong correlation"
                    elif abs(correlation) > 0.3:
                        corr_note = "Moderate correlation"
                    else:
                        corr_note = "Weak correlation"
                    
                    if correlation > 0:
                        corr_note += " (positive)"
                    elif correlation < 0:
                        corr_note += " (negative)"
                    
                    st.info(corr_note)
                else:
                    st.info("Correlation data unavailable")
            
            with col2:
                st.markdown("### 🏆 Winner")
                winner, notes = determine_winner(wave1_metrics, wave2_metrics)
                
                if winner:
                    if winner == "TIE":
                        st.success(f"**{winner}**")
                    else:
                        st.success(f"**{winner}**")
                    st.write(notes)
                else:
                    st.info(notes)
            
            # Additional notes
            st.divider()
            st.markdown("### 📝 Notes")
            st.write("- All metrics calculated over the most recent 30-day period")
            st.write("- WaveScore is the primary performance indicator (0-100 scale)")
            st.write("- Correlation measures return co-movement (diversification benefit when low)")
            
    except Exception as e:
        st.error(f"Error rendering Compare Waves panel: {str(e)}")


def render_overview_tab():
    """Render the Overview tab with Vector Explain and Compare Waves."""
    st.header("Overview")
    
    # Create sub-tabs for Vector Explain and Compare Waves
    overview_subtabs = st.tabs(["Vector Explain", "Compare Waves"])
    
    with overview_subtabs[0]:
        render_vector_explain_panel()
    
    with overview_subtabs[1]:
        render_compare_waves_panel()


def render_details_tab():
    """Render the Details tab."""
    st.header("Details")
    st.write("Detailed analytics and metrics for individual waves.")
    st.info("Data unavailable")


def render_reports_tab():
    """Render the Reports tab."""
    st.header("Reports")
    st.write("Comprehensive reporting and analysis tools.")
    st.info("Data unavailable")


def render_overlays_tab():
    """Render the Overlays tab."""
    st.header("Analytics Overlays")
    
    # Capital-Weighted Alpha Section
    st.subheader("Capital-Weighted Alpha")
    st.write("Alpha attribution weighted by capital allocation across portfolio.")
    st.info("Data unavailable")
    
    st.divider()
    
    # Exposure-Adjusted Alpha Section
    st.subheader("Exposure-Adjusted Alpha")
    st.write("Alpha metrics adjusted for market exposure and beta.")
    st.info("Data unavailable")
    
    st.divider()
    
    # Risk-On vs Risk-Off Attribution Section
    st.subheader("Risk-On vs Risk-Off Attribution")
    st.write("Performance attribution segmented by market regime.")
    st.info("Data unavailable")


# ============================================================================
# SECTION 8: MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main application entry point - Executive Layer v2.
    Orchestrates the entire Institutional Console UI with enhanced analytics.
    """
    # Render Mission Control at the top
    render_mission_control()
    
    # Render sidebar
    render_sidebar_info()
    
    # Main analytics tabs
    st.title("Institutional Console - Executive Layer v2")
    
    analytics_tabs = st.tabs(["Executive", "Overview", "Details", "Reports", "Overlays"])
    
    with analytics_tabs[0]:
        render_executive_tab()
    
    with analytics_tabs[1]:
        render_overview_tab()
    
    with analytics_tabs[2]:
        render_details_tab()
    
    with analytics_tabs[3]:
        render_reports_tab()
    
    with analytics_tabs[4]:
        render_overlays_tab()


# Run the application
if __name__ == "__main__":
    main()
