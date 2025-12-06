import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------
# Folder Setup
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
POSITIONS_DIR = LOGS_DIR / "positions"
PERFORMANCE_DIR = LOGS_DIR / "performance"

st.set_page_config(page_title="WAVES Live Engine", layout="wide")

st.title("🌊 WAVES Engine Dashboard")
st.write("App loaded OK — checking for performance logs...")

# ---------------------------------------------------------
# Discover available waves
# ---------------------------------------------------------

perf_files = list(PERFORMANCE_DIR.glob("*_performance_daily.csv"))
waves = sorted([f.stem.replace("_performance_daily", "") for f in perf_files])

if not waves:
    st.warning("No performance logs found yet. Run:  `python3 waves_engine.py` first.")
    st.stop()

wave_label_map = {w: w.replace("_", " ") for w in waves}

selected_wave = st.sidebar.selectbox("Select Wave", waves, format_func=lambda w: wave_label_map[w])
wave_name = selected_wave  # used for naming files

# ---------------------------------------------------------
# Load performance data
# ---------------------------------------------------------

perf_path = PERFORMANCE_DIR / f"{wave_name}_performance_daily.csv"
perf_df = pd.read_csv(perf_path)
perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])

st.subheader(f"📈 Performance — {wave_label_map[wave_name]}")

col1, col2 = st.columns(2)

with col1:
    st.metric("Latest NAV", f"{perf_df['nav'].iloc[-1]:.4f}")
    st.metric("Latest Alpha", f"{perf_df['alpha'].iloc[-1]:.2%}")

with col2:
    st.line_chart(perf_df.set_index("timestamp")[["nav", "benchmark_price"]])

st.caption("NAV vs Benchmark")

# ---------------------------------------------------------
# Latest positions
# ---------------------------------------------------------

st.subheader(f"📊 Latest Positions — {wave_label_map[wave_name]}")

pos_files = sorted(POSITIONS_DIR.glob(f"{wave_name}_positions_*.csv"))

if not pos_files:
    st.warning("No positions files found.")
else:
    latest_pos_path = pos_files[-1]
    pos_df = pd.read_csv(latest_pos_path)
    st.dataframe(pos_df)

    if "DollarWeight" in pos_df.columns:
        chart_df = pos_df.set_index("Ticker")["DollarWeight"]
        st.bar_chart(chart_df)
