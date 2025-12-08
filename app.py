import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from urllib.parse import quote_plus

# Optional: try to import the engine so we can "turn it on" if available
try:
    import waves_engine  # type: ignore
except ImportError:
    waves_engine = None


# ---------- Helpers for data discovery ----------

LOGS_POSITIONS_DIR = os.path.join("logs", "positions")
LOGS_PERFORMANCE_DIR = os.path.join("logs", "performance")


def get_available_waves() -> List[str]:
    """
    Discover wave names from performance logs.
    Assumes filenames like: <Wave>_performance_daily.csv
    """
    waves = set()

    if os.path.isdir(LOGS_PERFORMANCE_DIR):
        for fname in os.listdir(LOGS_PERFORMANCE_DIR):
            if not fname.endswith(".csv"):
                continue
            if "_performance_" not in fname:
                continue
            wave_name = fname.split("_performance_")[0]
            if wave_name:
                waves.add(wave_name)

    # Fallback to positions dir if needed
    if not waves and os.path.isdir(LOGS_POSITIONS_DIR):
        for fname in os.listdir(LOGS_POSITIONS_DIR):
            if not fname.endswith(".csv"):
                continue
            if "_positions_" not in fname:
                continue
            wave_name = fname.split("_positions_")[0]
            if wave_name:
                waves.add(wave_name)

    return sorted(waves)


def parse_date_from_filename(fname: str) -> Optional[datetime]:
    """
    Extract YYYYMMDD from filenames like <Wave>_positions_YYYYMMDD.csv
    or <Wave>_performance_daily_YYYYMMDD.csv if present.
    Falls back to None if not parseable.
    """
    base = os.path.basename(fname)
    # Find the last 8-digit group
    for part in base.replace(".csv", "").split("_")[::-1]:
        if len(part) == 8 and part.isdigit():
            try:
                return datetime.strptime(part, "%Y%m%d")
            except ValueError:
                return None
    return None


def load_latest_positions(wave: str) -> Optional[pd.DataFrame]:
    if not os.path.isdir(LOGS_POSITIONS_DIR):
        return None

    candidates = []
    for fname in os.listdir(LOGS_POSITIONS_DIR):
        if not fname.endswith(".csv"):
            continue
        if not fname.startswith(f"{wave}_positions_"):
            continue
        dt = parse_date_from_filename(fname)
        if dt is not None:
            candidates.append((dt, fname))

    if not candidates:
        return None

    latest_fname = max(candidates, key=lambda x: x[0])[1]
    full_path = os.path.join(LOGS_POSITIONS_DIR, latest_fname)
    try:
        return pd.read_csv(full_path)
    except Exception:
        return None


def load_performance_history(wave: str) -> Optional[pd.DataFrame]:
    if not os.path.isdir(LOGS_PERFORMANCE_DIR):
        return None

    # Prefer per-wave daily file if exists
    daily_file = f"{wave}_performance_daily.csv"
    full_path = os.path.join(LOGS_PERFORMANCE_DIR, daily_file)
    if os.path.exists(full_path):
        try:
            df = pd.read_csv(full_path)
        except Exception:
            return None
    else:
        # fallback: pick any file that matches pattern
        candidates = []
        for fname in os.listdir(LOGS_PERFORMANCE_DIR):
            if not fname.endswith(".csv"):
                continue
            if not fname.startswith(f"{wave}_performance_"):
                continue
            dt = parse_date_from_filename(fname)
            if dt is not None:
                candidates.append((dt, fname))
        if not candidates:
            return None
        latest_fname = max(candidates, key=lambda x: x[0])[1]
        full_path = os.path.join(LOGS_PERFORMANCE_DIR, latest_fname)
        try:
            df = pd.read_csv(full_path)
        except Exception:
            return None

    # Normalize date column
    for col in ["date", "Date", "DATE", "as_of"]:
        if col in df.columns:
            try:
                df["date"] = pd.to_datetime(df[col])
                df = df.sort_values("date")
            except Exception:
                pass
            break

    return df


# ---------- Google Finance link helpers ----------

def google_quote_url(ticker: str) -> str:
    """
    Build a robust Google Finance quote URL for a ticker.
    We don't assume an exchange; Google will resolve it.
    """
    if not isinstance(ticker, str):
        return "https://www.google.com/finance"
    clean = ticker.strip().upper()
    if not clean:
        return "https://www.google.com/finance"
    # Use q= param so we don't need exchange codes
    return f"https://www.google.com/finance?q={quote_plus(clean)}"


def render_top10_holdings(holdings_df: Optional[pd.DataFrame], wave_name: str):
    """
    holdings_df MUST have at least: ['Ticker', 'Name', 'Weight'] (or similar).
    Displays a Top 10 table with working Google Finance links.
    """
    if holdings_df is None or holdings_df.empty:
        st.info(f"No holdings found for {wave_name}.")
        return

    # Normalize column names just in case
    cols = {c.lower(): c for c in holdings_df.columns}
    ticker_col = cols.get("ticker")
    name_col = cols.get("name") or cols.get("company") or cols.get("security")
    weight_col = cols.get("weight") or cols.get("portfolio_weight") or cols.get("weight_%")

    if not ticker_col or not weight_col:
        st.warning(f"Top 10 holdings for {wave_name} could not be displayed (missing Ticker/Weight).")
        return

    top10 = (
        holdings_df
        .sort_values(weight_col, ascending=False)
        .head(10)
        .copy()
    )

    # Build Google quote links
    top10["Google Quote"] = top10[ticker_col].apply(
        lambda t: f"[{str(t).upper()}]({google_quote_url(str(t))})"
    )

    # Nice, clean table for display
    display_cols = ["Google Quote"]
    if name_col:
        display_cols.append(name_col)
    display_cols.append(weight_col)

    display_df = top10[display_cols].rename(columns={
        name_col: "Name" if name_col else name_col,
        weight_col: "Weight"
    })

    st.subheader(f"Top 10 Holdings — {wave_name}")
    st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)


# ---------- Metrics helpers ----------

def compute_summary_metrics(perf_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    """
    Compute intraday, 30d, 60d total return and alpha metrics
    from the performance history DataFrame.

    We are *intentionally* excluding 1-year and since-inception figures
    per your instructions.
    """
    metrics = {
        "intraday_return": None,
        "return_30d": None,
        "return_60d": None,
        "alpha_30d": None,   # 30-day internal alpha only
    }

    if perf_df is None or perf_df.empty:
        return metrics

    last_row = perf_df.iloc[-1]

    # Helper to pull the "best guess" column by candidate names
    def pull(col_candidates: List[str]) -> Optional[float]:
        for c in col_candidates:
            if c in perf_df.columns:
                try:
                    return float(last_row[c])
                except Exception:
                    continue
        return None

    # These will adapt to whatever column names your engine is using
    metrics["intraday_return"] = pull(["intraday_return", "daily_return", "return_1d", "r_1d"])
    metrics["return_30d"] = pull(["return_30d", "r_30d", "total_return_30d"])
    metrics["return_60d"] = pull(["return_60d", "r_60d", "total_return_60d"])

    # Only up to 30-day alpha per your latest direction
    metrics["alpha_30d"] = pull(["alpha_30d", "alpha30", "alpha_1m", "alpha_30"])

    return metrics


def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{x * 100:.2f}%"
    except Exception:
        return "N/A"


# ---------- Alpha Capture Matrix ----------

def load_alpha_capture_matrix() -> Optional[pd.DataFrame]:
    """
    Try to load an alpha capture matrix if present.
    Path/filename can be adjusted to match your engine output.
    """
    # Common possible locations / names
    candidates = [
        os.path.join("logs", "performance", "alpha_capture_matrix.csv"),
        os.path.join("logs", "alpha_capture_matrix.csv"),
        "alpha_capture_matrix.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return None


def render_alpha_capture_tab():
    st.subheader("Alpha Capture Matrix")
    df = load_alpha_capture_matrix()
    if df is None or df.empty:
        st.info("Alpha Capture Matrix not available yet.")
        return

    st.dataframe(df, use_container_width=True)


# ---------- Streamlit App ----------

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live OS Console • Adaptive Portfolio Waves™ • Alpha-Minus-Beta • Private Logic™")

# Optional: light "engine kick" if available
if waves_engine is not None:
    with st.expander("Engine Status / Controls", expanded=False):
        st.write("`waves_engine` module detected. You can kick off or monitor the engine here.")
        if st.button("Run Engine Once (Manual Kick)"):
            try:
                waves_engine.run_all_waves()  # type: ignore[attr-defined]
                st.success("Engine run triggered successfully.")
            except Exception as e:
                st.error(f"Error running engine: {e}")

# Discover Waves
waves = get_available_waves()
if not waves:
    st.error("No Waves discovered in logs. Make sure the engine has written logs to /logs/performance or /logs/positions.")
    st.stop()

# Sidebar controls
st.sidebar.header("Wave & Mode")
selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.write("Mode affects how the engine manages risk, beta, and turnover.")
st.sidebar.write("- **Standard**: Balanced growth vs risk")
st.sidebar.write("- **Alpha-Minus-Beta**: Defensive, beta-targeted overlays")
st.sidebar.write("- **Private Logic™**: Aggressive, proprietary alpha-seeking logic")

# Main layout
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"{selected_wave} — Overview")

    perf_df = load_performance_history(selected_wave)
    metrics = compute_summary_metrics(perf_df)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Intraday Return", format_pct(metrics["intraday_return"]))
    with m2:
        st.metric("30-Day Total Return", format_pct(metrics["return_30d"]))
    with m3:
        st.metric("60-Day Total Return", format_pct(metrics["return_60d"]))
    with m4:
        st.metric("30-Day Alpha", format_pct(metrics["alpha_30d"]))

    # Performance chart (30/60d)
    if perf_df is not None and not perf_df.empty and "date" in perf_df.columns:
        chart_cols = []
        for c in ["return_30d", "return_60d", "alpha_30d"]:
            if c in perf_df.columns:
                chart_cols.append(c)

        if chart_cols:
            st.markdown("### 30–60 Day Performance & Alpha (Engine Output)")
            chart_data = perf_df.set_index("date")[chart_cols]
            st.line_chart(chart_data)

with col_right:
    # Load positions and show Top 10 with Google links
    positions_df = load_latest_positions(selected_wave)
    render_top10_holdings(positions_df, selected_wave)

st.markdown("---")

# Tabs for more detail
tab1, tab2 = st.tabs(["Wave Details", "Alpha Capture Matrix"])

with tab1:
    st.subheader(f"{selected_wave} — Detailed Positions")
    if positions_df is not None and not positions_df.empty:
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No detailed positions available for this Wave yet.")

with tab2:
    render_alpha_capture_tab()