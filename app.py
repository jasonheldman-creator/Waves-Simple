import streamlit as st
import subprocess
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

st.set_page_config(page_title="Institutional Console", layout="wide")

# ============================================================================
# MISSION CONTROL SUMMARY STRIP
# ============================================================================

def get_mission_control_data():
    """
    Retrieve Mission Control metrics from available data.
    Returns dict with all metrics, using 'unknown' for unavailable data.
    """
    mc_data = {
        'market_regime': 'unknown',
        'vix_gate_status': 'unknown',
        'alpha_today': 'unknown',
        'alpha_30day': 'unknown',
        'wavescore_leader': 'unknown',
        'wavescore_leader_score': 'unknown',
        'data_freshness': 'unknown',
        'data_age_days': None
    }
    
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            return mc_data
            
        df = pd.read_csv(wave_history_path)
        
        if 'date' not in df.columns or len(df) == 0:
            return mc_data
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Get latest date and data freshness
        latest_date = df['date'].max()
        mc_data['data_freshness'] = latest_date.strftime('%Y-%m-%d')
        
        # Calculate data age in days
        age_days = (datetime.now() - latest_date).days
        mc_data['data_age_days'] = age_days
        
        # Calculate Market Regime based on recent returns
        # Simple heuristic: average portfolio returns over last 5 days
        recent_days = 5
        recent_data = df[df['date'] >= (latest_date - timedelta(days=recent_days))]
        if 'portfolio_return' in df.columns and len(recent_data) > 0:
            avg_return = recent_data['portfolio_return'].mean()
            if avg_return > 0.005:  # > 0.5% average
                mc_data['market_regime'] = 'Risk-On'
            elif avg_return < -0.005:  # < -0.5% average
                mc_data['market_regime'] = 'Risk-Off'
            else:
                mc_data['market_regime'] = 'Neutral'
        
        # VIX Gate Status - gracefully unavailable for now
        mc_data['vix_gate_status'] = 'Unknown'
        
        # Calculate Alpha metrics (portfolio_return - benchmark_return)
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
            
            # Today's alpha (most recent date, averaged across all waves)
            today_data = df[df['date'] == latest_date]
            if len(today_data) > 0:
                alpha_today = today_data['alpha'].mean()
                mc_data['alpha_today'] = f"{alpha_today*100:.2f}%"
            
            # 30-day average alpha
            days_30_ago = latest_date - timedelta(days=30)
            last_30_days = df[df['date'] >= days_30_ago]
            if len(last_30_days) > 0:
                alpha_30day = last_30_days['alpha'].mean()
                mc_data['alpha_30day'] = f"{alpha_30day*100:.2f}%"
        
        # WaveScore Leader - identify wave with best recent performance
        # Using 30-day cumulative alpha as proxy for WaveScore
        if 'wave' in df.columns and 'alpha' in df.columns:
            days_30_ago = latest_date - timedelta(days=30)
            last_30_days = df[df['date'] >= days_30_ago]
            
            # Calculate cumulative alpha by wave
            wave_performance = last_30_days.groupby('wave')['alpha'].sum().sort_values(ascending=False)
            
            if len(wave_performance) > 0:
                top_wave = wave_performance.index[0]
                top_alpha = wave_performance.iloc[0]
                
                # Convert to a 0-100 score (normalize cumulative alpha)
                # Use a simple heuristic: alpha * 1000 + 50, clamped to 0-100
                wavescore = min(100, max(0, (top_alpha * 1000) + 50))
                
                mc_data['wavescore_leader'] = top_wave
                mc_data['wavescore_leader_score'] = f"{wavescore:.1f}"
        
    except Exception as e:
        # Silently handle errors, return 'unknown' values
        pass
    
    return mc_data


def render_mission_control():
    """Render the Mission Control summary strip at the top of the page."""
    st.markdown("### 🎯 Mission Control")
    
    mc_data = get_mission_control_data()
    
    # Create 5 columns for the 5 tiles
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Market Regime",
            value=mc_data['market_regime'],
            help="Current market regime based on recent portfolio performance"
        )
    
    with col2:
        st.metric(
            label="VIX Gate Status",
            value=mc_data['vix_gate_status'],
            help="VIX-based gating status for risk management"
        )
    
    with col3:
        st.markdown("**Alpha Captured**")
        st.write(f"Today: {mc_data['alpha_today']}")
        st.write(f"30-Day Avg: {mc_data['alpha_30day']}")
    
    with col4:
        st.markdown("**WaveScore Leader**")
        if mc_data['wavescore_leader'] != 'unknown':
            st.write(f"{mc_data['wavescore_leader'][:25]}...")
            st.write(f"Score: {mc_data['wavescore_leader_score']}")
        else:
            st.write("Unknown")
    
    with col5:
        freshness_label = "System Health"
        freshness_value = mc_data['data_freshness']
        
        # Add indicator based on data age
        if mc_data['data_age_days'] is not None:
            if mc_data['data_age_days'] <= 2:
                freshness_value = f"✅ {freshness_value}"
            elif mc_data['data_age_days'] <= 7:
                freshness_value = f"⚠️ {freshness_value}"
            else:
                freshness_value = f"❌ {freshness_value}"
        
        st.metric(
            label=freshness_label,
            value=freshness_value,
            help="Last data update timestamp"
        )
    
    st.divider()


# ============================================================================
# EXECUTIVE SECTION - LEADERBOARD, MOVERS, ALERTS
# ============================================================================

def calculate_wavescore(wave_data):
    """
    Calculate WaveScore for a wave based on cumulative alpha over 30 days.
    Returns a score between 0 and 100.
    """
    try:
        if len(wave_data) == 0:
            return 0
        
        # Calculate cumulative alpha
        cumulative_alpha = wave_data['alpha'].sum()
        
        # Normalize to 0-100 range
        # Using heuristic: alpha * 1000 + 50, clamped to 0-100
        wavescore = min(100, max(0, (cumulative_alpha * 1000) + 50))
        
        return wavescore
    except Exception:
        return 0


def get_wavescore_leaderboard():
    """
    Get top 10 waves by WaveScore (30-day cumulative alpha).
    Returns a DataFrame with wave names and scores, or None if unavailable.
    """
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            return None
        
        df = pd.read_csv(wave_history_path)
        
        if 'date' not in df.columns or 'wave' not in df.columns or len(df) == 0:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate alpha if needed
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
        else:
            return None
        
        # Get last 30 days of data
        latest_date = df['date'].max()
        days_30_ago = latest_date - timedelta(days=30)
        last_30_days = df[df['date'] >= days_30_ago]
        
        if len(last_30_days) == 0:
            return None
        
        # Calculate WaveScore for each wave
        wave_scores = []
        for wave in last_30_days['wave'].unique():
            wave_data = last_30_days[last_30_days['wave'] == wave]
            score = calculate_wavescore(wave_data)
            wave_scores.append({'Wave': wave, 'WaveScore': score})
        
        # Create DataFrame and sort by score
        leaderboard_df = pd.DataFrame(wave_scores)
        leaderboard_df = leaderboard_df.sort_values('WaveScore', ascending=False).head(10)
        leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)
        leaderboard_df = leaderboard_df[['Rank', 'Wave', 'WaveScore']]
        
        return leaderboard_df
        
    except Exception as e:
        return None


def get_biggest_movers():
    """
    Get biggest month-over-month WaveScore changes.
    Returns a DataFrame with wave names and score changes, or None if unavailable.
    """
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            return None
        
        df = pd.read_csv(wave_history_path)
        
        if 'date' not in df.columns or 'wave' not in df.columns or len(df) == 0:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate alpha if needed
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
        
        # Create DataFrame and sort by absolute change
        movers_df = pd.DataFrame(movers)
        movers_df = movers_df.sort_values('Change', ascending=False, key=abs).head(10)
        
        return movers_df
        
    except Exception as e:
        return None


def get_system_alerts():
    """
    Generate system alerts based on data quality and risk signals.
    Returns a list of alert dictionaries with severity and message.
    """
    alerts = []
    
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            alerts.append({
                'severity': 'error',
                'message': 'Wave history data file not found'
            })
            return alerts
        
        df = pd.read_csv(wave_history_path)
        
        if 'date' not in df.columns or len(df) == 0:
            alerts.append({
                'severity': 'error',
                'message': 'Wave history data is empty or missing date column'
            })
            return alerts
        
        df['date'] = pd.to_datetime(df['date'])
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
            
            # Check for data gaps (missing dates)
            date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
            actual_dates = df['date'].unique()
            missing_dates = len(date_range) - len(actual_dates)
            
            if missing_dates > 10:
                alerts.append({
                    'severity': 'warning',
                    'message': f'{missing_dates} days of data missing in date range'
                })
            
            # Check for high volatility (potential drawdown widening)
            last_30_days = df[df['date'] >= (latest_date - timedelta(days=30))]
            if len(last_30_days) > 0:
                volatility = last_30_days['alpha'].std()
                if volatility > 0.05:  # 5% daily volatility threshold
                    alerts.append({
                        'severity': 'warning',
                        'message': f'High volatility detected: {volatility*100:.2f}% (potential drawdown risk)'
                    })
            
            # Check for negative cumulative alpha (underperformance)
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
        wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
        
        # What Happened section
        narrative_parts.append("## What Happened")
        
        # Get date range
        start_date = wave_data['date'].min()
        end_date = wave_data['date'].max()
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
        
        narrative_parts.append(f"- Best performance: {best_day['alpha']*100:.2f}% alpha on {best_day['date']}")
        narrative_parts.append(f"- Worst performance: {worst_day['alpha']*100:.2f}% alpha on {worst_day['date']}")
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


def get_wave_data_for_narrative(wave_name):
    """
    Retrieve wave data for narrative generation.
    Returns DataFrame for the specified wave over the last 30 days.
    """
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            return None
        
        df = pd.read_csv(wave_history_path)
        
        if 'date' not in df.columns or 'wave' not in df.columns or len(df) == 0:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for specific wave
        wave_df = df[df['wave'] == wave_name].copy()
        
        if len(wave_df) == 0:
            return None
        
        # Get last 30 days
        latest_date = wave_df['date'].max()
        days_30_ago = latest_date - timedelta(days=30)
        wave_df_30d = wave_df[wave_df['date'] >= days_30_ago].copy()
        
        return wave_df_30d
        
    except Exception as e:
        return None


def get_wave_comparison_data(wave1_name, wave2_name):
    """
    Retrieve comparison data for two waves.
    Returns a dictionary with comparison metrics or None if unavailable.
    """
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            return None
        
        df = pd.read_csv(wave_history_path)
        
        if 'date' not in df.columns or 'wave' not in df.columns or len(df) == 0:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for both waves
        wave1_df = df[df['wave'] == wave1_name].copy()
        wave2_df = df[df['wave'] == wave2_name].copy()
        
        if len(wave1_df) == 0 or len(wave2_df) == 0:
            return None
        
        # Get last 30 days for both waves
        latest_date = df['date'].max()
        days_30_ago = latest_date - timedelta(days=30)
        
        wave1_30d = wave1_df[wave1_df['date'] >= days_30_ago].copy()
        wave2_30d = wave2_df[wave2_df['date'] >= days_30_ago].copy()
        
        if len(wave1_30d) == 0 or len(wave2_30d) == 0:
            return None
        
        # Calculate metrics for Wave 1
        wave1_metrics = calculate_wave_metrics(wave1_30d)
        wave1_metrics['name'] = wave1_name
        
        # Calculate metrics for Wave 2
        wave2_metrics = calculate_wave_metrics(wave2_30d)
        wave2_metrics['name'] = wave2_name
        
        # Calculate correlation if we have overlapping dates
        correlation = calculate_wave_correlation(wave1_30d, wave2_30d)
        
        return {
            'wave1': wave1_metrics,
            'wave2': wave2_metrics,
            'correlation': correlation
        }
        
    except Exception as e:
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
        if len(wave_data) == 0:
            return metrics
        
        # Calculate alpha
        if 'portfolio_return' in wave_data.columns and 'benchmark_return' in wave_data.columns:
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


def calculate_wave_correlation(wave1_data, wave2_data):
    """
    Calculate correlation between two waves based on their returns.
    Returns correlation coefficient or None if unavailable.
    """
    try:
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


def render_compare_waves_panel():
    """Render the Compare Waves panel for head-to-head wave comparison."""
    st.subheader("⚖️ Compare Waves")
    st.write("Select two waves for head-to-head performance comparison")
    
    try:
        # Get list of available waves
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            st.warning("Wave history data not available")
            return
        
        df = pd.read_csv(wave_history_path)
        
        if 'wave' not in df.columns or len(df) == 0:
            st.warning("No wave data available")
            return
        
        waves = sorted(df['wave'].unique())
        
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
            
            # Create comparison table
            comparison_rows = []
            
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
            comparison_rows.append({
                'Metric': 'Cumulative Return',
                wave1_metrics['name']: format_metric(wave1_metrics['cumulative_return']),
                wave2_metrics['name']: format_metric(wave2_metrics['cumulative_return'])
            })
            
            comparison_rows.append({
                'Metric': 'Cumulative Alpha',
                wave1_metrics['name']: format_metric(wave1_metrics['cumulative_alpha']),
                wave2_metrics['name']: format_metric(wave2_metrics['cumulative_alpha'])
            })
            
            comparison_rows.append({
                'Metric': 'Volatility',
                wave1_metrics['name']: format_metric(wave1_metrics['volatility']),
                wave2_metrics['name']: format_metric(wave2_metrics['volatility'])
            })
            
            comparison_rows.append({
                'Metric': 'Max Drawdown',
                wave1_metrics['name']: format_metric(wave1_metrics['max_drawdown']),
                wave2_metrics['name']: format_metric(wave2_metrics['max_drawdown'])
            })
            
            comparison_rows.append({
                'Metric': 'WaveScore',
                wave1_metrics['name']: format_metric(wave1_metrics['wavescore'], 'score'),
                wave2_metrics['name']: format_metric(wave2_metrics['wavescore'], 'score')
            })
            
            comparison_rows.append({
                'Metric': 'Sharpe Ratio',
                wave1_metrics['name']: format_metric(wave1_metrics['sharpe_ratio'], 'ratio'),
                wave2_metrics['name']: format_metric(wave2_metrics['sharpe_ratio'], 'ratio')
            })
            
            comparison_rows.append({
                'Metric': 'Win Rate',
                wave1_metrics['name']: format_metric(wave1_metrics['win_rate']),
                wave2_metrics['name']: format_metric(wave2_metrics['win_rate'])
            })
            
            # Display table
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Correlation
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


def render_vector_explain_panel():
    """Render the Vector Explain panel for generating Wave narratives."""
    st.subheader("📝 Vector Explain")
    st.write("Generate an institutional narrative for a selected Wave")
    
    try:
        # Get list of available waves
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if not os.path.exists(wave_history_path):
            st.warning("Wave history data not available")
            return
        
        df = pd.read_csv(wave_history_path)
        
        if 'wave' not in df.columns or len(df) == 0:
            st.warning("No wave data available")
            return
        
        waves = sorted(df['wave'].unique())
        
        # Wave selector
        selected_wave = st.selectbox(
            "Select Wave",
            options=waves,
            help="Choose a wave to generate an institutional narrative"
        )
        
        if st.button("Generate Narrative", type="primary"):
            with st.spinner("Generating narrative..."):
                wave_data = get_wave_data_for_narrative(selected_wave)
                narrative = generate_wave_narrative(selected_wave, wave_data)
                
                # Store in session state
                st.session_state['current_narrative'] = narrative
        
        # Display narrative if available
        if 'current_narrative' in st.session_state:
            st.divider()
            st.markdown(st.session_state['current_narrative'])
            
            # Copy Summary button
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📋 Copy Summary to Clipboard", use_container_width=True):
                    st.code(st.session_state['current_narrative'], language=None)
                    st.success("Summary displayed above - use your browser to copy the text")
    
    except Exception as e:
        st.error(f"Error rendering Vector Explain panel: {str(e)}")


def render_executive_section():
    """Render the Executive section with Leaderboard, Movers, and Alerts."""
    
    st.header("📊 Executive Dashboard")
    
    # Create two columns for Leaderboard and Movers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 WaveScore Leaderboard")
        st.write("Top 10 Waves by 30-day performance")
        
        leaderboard = get_wavescore_leaderboard()
        if leaderboard is not None and len(leaderboard) > 0:
            # Format the WaveScore column
            leaderboard_display = leaderboard.copy()
            leaderboard_display['WaveScore'] = leaderboard_display['WaveScore'].apply(lambda x: f"{x:.1f}")
            st.dataframe(leaderboard_display, use_container_width=True, hide_index=True)
        else:
            st.info("Data unavailable")
    
    with col2:
        st.subheader("📈 Biggest Movers")
        st.write("Month-over-Month WaveScore changes")
        
        movers = get_biggest_movers()
        if movers is not None and len(movers) > 0:
            # Format the display
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
    
    # Alerts section
    st.subheader("🚨 System Alerts")
    
    alerts = get_system_alerts()
    
    if alerts:
        for alert in alerts:
            severity = alert.get('severity', 'info')
            message = alert.get('message', '')
            
            if severity == 'error':
                st.error(f"❌ {message}")
            elif severity == 'warning':
                st.warning(f"⚠️ {message}")
            elif severity == 'success':
                st.success(f"✅ {message}")
            else:
                st.info(f"ℹ️ {message}")
    else:
        st.info("No alerts at this time")


# Render Mission Control at the top
render_mission_control()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Pseudocode for restoring the Institutional Console layout and analytics
# Including Risk Lab, Correlation Matrix, Rolling Alpha/Vol, Drawdown Monitor, and full analytics

# Risk Lab
st.sidebar.title("Risk Lab")
st.sidebar.write("Description of Risk Lab...")

# Correlation Matrix
st.sidebar.title("Correlation Matrix")
st.sidebar.write("Description of Correlation Matrix...")

# Rolling Alpha/Vol
st.sidebar.title("Rolling Alpha / Volatility")
st.sidebar.write("Description of Rolling Alpha and Volatility...")

# Drawdown Monitor
st.sidebar.title("Drawdown Monitor")
st.sidebar.write("Description of Drawdown Monitor...")

# Full Analytics
st.title("Institutional Console Analytics")

# Placeholder for analytics tabs and data flows
analytics_tabs = st.tabs(["Executive", "Overview", "Details", "Reports", "Overlays"])
with analytics_tabs[0]:
    # Executive tab - new addition
    render_executive_section()
with analytics_tabs[1]:
    st.header("Overview")
    
    # Create sub-tabs for Vector Explain and Compare Waves
    overview_subtabs = st.tabs(["Vector Explain", "Compare Waves"])
    
    with overview_subtabs[0]:
        render_vector_explain_panel()
    
    with overview_subtabs[1]:
        render_compare_waves_panel()
with analytics_tabs[2]:
    st.write("Details Content...")
with analytics_tabs[3]:
    st.write("Reports Content...")
with analytics_tabs[4]:
    st.header("Analytics Overlays")
    
    # Capital-Weighted Alpha Section
    st.subheader("Capital-Weighted Alpha")
    st.write("Data unavailable")
    
    st.divider()
    
    # Exposure-Adjusted Alpha Section
    st.subheader("Exposure-Adjusted Alpha")
    st.write("Data unavailable")
    
    st.divider()
    
    # Risk-On vs Risk-Off Attribution Section
    st.subheader("Risk-On vs Risk-Off Attribution")
    st.write("Data unavailable")

# Build ID Footer Functions
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

def get_latest_data_timestamp():
    """Get the latest available 'as of' data timestamp from wave_history.csv."""
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        if os.path.exists(wave_history_path):
            df = pd.read_csv(wave_history_path)
            if 'date' in df.columns and len(df) > 0:
                latest_date = df['date'].max()
                return latest_date
    except Exception:
        pass
    return "unknown"

def get_deploy_timestamp():
    """Get the current timestamp as deploy timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

# Build ID Footer Display
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
