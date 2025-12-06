import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------
# Paths / folders
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
POSITIONS_DIR = LOGS_DIR / "positions"
PERFORMANCE_DIR = LOGS_DIR / "performance"

st.set_page_config(page_title="WAVES Live Engine", layout="wide")

st.title("🌊 WAVES Engine Dashboard")
st.write("App loaded — checking logs for live Waves data...")

# ---------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------

if not LOGS_DIR.exists():
    st.error("`logs/` folder not found. Make sure you're running this in the same folder as `waves_engine.py`.")
    st.stop()

if not PERFORMANCE_DIR.exists():
    st.warning("`logs/performance/` folder not found yet. Run `python3 waves_engine.py` once to create logs.")
    st.stop()

# ---------------------------------------------------------
# Discover Waves from performance logs
# ---------------------------------------------------------

perf_files = list(PERFORMANCE_DIR.glob("*_performance_daily.csv"))

if not perf_files:
    st.warning("No performance logs found. Run `python3 waves_engine.py` to generate data.")
    st.stop()

waves = sorted([f.stem.replace("_performance_daily", "") for f in perf_files])
wave_label_map = {w: w.replace("_", " ") for w in waves}

selected_wave = st.sidebar.selectbox(
    "Select Wave",
    options=waves,
    format_func=lambda w: wave_label_map[w],
)

wave_name = selected_wave  # used for filenames

# ---------------------------------------------------------
# Load performance data for selected wave
# ---------------------------------------------------------

perf_path = PERFORMANCE_DIR / f"{wave_name}_performance_daily.csv"

try:
    perf_df = pd.read_csv(perf_path)
except FileNotFoundError:
    st.error(f"Performance file not found: {perf_path.name}")
    st.stop()

if perf_df.empty:
    st.warning("Performance file is empty. Run `python3 waves_engine.py` to append data.")
    st.stop()

# ensure timestamp is datetime
if "timestamp" in perf_df.columns:
    perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])

st.subheader(f"📈 Performance — {wave_label_map[wave_name]}")

col1, col2 = st.columns(2)

with col1:
    latest_nav = perf_df["nav"].iloc[-1] if "nav" in perf_df.columns else float("nan")
    latest_alpha = perf_df["alpha"].iloc[-1] if "alpha" in perf_df.columns else 0.0

    st.metric("Latest NAV", f"{latest_nav:.4f}")
    st.metric("Latest Alpha", f"{latest_alpha:.2%}")

with col2:
    chart_cols = []
    if "nav" in perf_df.columns:
        chart_cols.append("nav")
    if "benchmark_price" in perf_df.columns:
        chart_cols.append("benchmark_price")

    if chart_cols:
        st.line_chart(
            perf_df.set_index("timestamp")[chart_cols]
        )
    else:
        st.info("No NAV / benchmark columns available for charting.")

st.caption("NAV vs Benchmark (raw values; not scaled)")

# ---------------------------------------------------------
# Latest Positions
# ---------------------------------------------------------

st.subheader(f"📊 Latest Positions — {wave_label_map[wave_name]}")

if not POSITIONS_DIR.exists():
    st.warning("`logs/positions/` folder not found yet. Run `python3 waves_engine.py` to create positions logs.")
else:
    pos_files = sorted(POSITIONS_DIR.glob(f"{wave_name}_positions_*.csv"))

    if not pos_files:
        st.warning("No positions files found for this wave yet.")
    else:
        latest_pos_path = pos_files[-1]
        try:
            pos_df = pd.read_csv(latest_pos_path)
        except Exception as e:
            st.error(f"Error reading positions file: {latest_pos_path.name}\n\n{e}")
        else:
            st.write(f"Latest positions file: `{latest_pos_path.name}`")
            st.dataframe(pos_df)

            # Bar chart of dollar weights if present
            if "DollarWeight" in pos_df.columns and "Ticker" in pos_df.columns:
                chart_df = pos_df.set_index("Ticker")["DollarWeight"]
                st.bar_chart(chart_df)
            else:
                st.info("No `DollarWeight` column available for bar chart.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")
st.caption(
    "WAVES Intelligence™ Live Engine — driven by `waves_engine.py` logs "
    "from `logs/performance` and `logs/positions`."
)
