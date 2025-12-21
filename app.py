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
    st.write("Overview Content...")
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
