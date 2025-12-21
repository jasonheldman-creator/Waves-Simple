import streamlit as st
import subprocess
import os
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Institutional Console", layout="wide")

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
analytics_tabs = st.tabs(["Overview", "Details", "Reports", "Overlays"])
with analytics_tabs[0]:
    st.write("Overview Content...")
with analytics_tabs[1]:
    st.write("Details Content...")
with analytics_tabs[2]:
    st.write("Reports Content...")
with analytics_tabs[3]:
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
