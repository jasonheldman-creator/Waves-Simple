import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from urllib.parse import quote_plus

import waves_engine  # Our engine OS layer

# ---------- Paths ----------

LOGS_POSITIONS_DIR = os.path.join("logs", "positions")
LOGS_PERFORMANCE_DIR = os.path.join("logs", "performance")

# ---------- Streamlit Config ----------

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Adaptive Portfolio Waves™ • Alpha-Minus-Beta • Private Logic™ • SmartSafe™")


# ---------- Wave Discovery ----------

def get_available_waves() -> List[str]:
    """
    Discover waves from performance logs; fallback to weights if needed.
    """
    waves = set()

    if os.path.isdir(LOGS_PERFORMANCE_DIR):
        for fname in os.listdir(LOGS_PERFORMANCE_DIR):
            if not fname.endswith(".csv"):
                continue
            if "_performance_" not in fname and "_performance" not in fname:
                continue
            base = fname.replace(".csv", "")
            # e.g., S&P_500_Wave_performance_daily
            parts = base.split("_performance")
            wave_name = parts[0].replace("_", " ")
            if wave_name:
                waves.add(wave_name)

    if not waves:
        # Fallback: discover via weights file
        try:
            weights_df = waves_engine.load_wave_weights()  # type: ignore
            if weights_df is not None:
                waves.update(waves_engine.discover_waves_from_weights(weights_df))  # type: ignore
        except Exception:
            pass

    return sorted(waves)


# ---------- File Helpers ----------

def _perf_path_for_wave(wave: str) -> str:
    safe = wave.replace(" ", "_")
    return os.path.join(LOGS_PERFORMANCE_DIR, f"{safe}_performance_daily.csv")


def _latest_positions_path_for_wave(wave: str) -> Optional[str]:
    if not os.path.isdir(LOGS_POSITIONS_DIR):
        return None
    safe = wave.replace(" ", "_")
    candidates = []
    for fname in os.listdir(LOGS_POSITIONS_DIR):
        if not fname.endswith(".csv"):
            continue
        if not fname.startswith(safe + "_positions_"):
            continue
        # extract YYYYMMDD
        stem = fname.replace(".csv", "")
        parts = stem.split("_")
        if parts and len(parts[-1]) == 8 and parts[-1].isdigit():
            try:
                dt = datetime.strptime(parts[-1], "%Y%m%d")
                candidates.append((dt, fname))
            except Exception:
                continue
    if not candidates:
        return None
    latest_fname = max(candidates, key=lambda x: x[0])[1]
    return os.path.join(LOGS_POSITIONS_DIR, latest_fname)


def load_performance_history(wave: str) -> Optional[pd.DataFrame]:
    path = _perf_path_for_wave(wave)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


def load_latest_positions(wave: str) -> Optional[pd.DataFrame]:
    path = _latest_positions_path_for_wave(wave)
    if path is None or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------- Google Finance Link Helpers ----------

def google_quote_url(ticker: str) -> str:
    """
    Build a robust Google Finance quote URL for a ticker.
    """
    if not isinstance(ticker, str):
        return "https://www.google.com/finance"
    clean = ticker.strip().upper()
    if not clean:
        return "https://www.google.com/finance"
    return f"https://www.google.com/finance?q={quote_plus(clean)}"


def render_top10_holdings(holdings_df: Optional[pd.DataFrame], wave_name: str) -> None:
    """
    Display Top 10 holdings with clickable Google Finance links.
    """
    if holdings_df is None or holdings_df.empty:
        st.info(f"No holdings available for {wave_name}.")
        return

    cols = {c.lower(): c for c in holdings_df.columns}
    ticker_col = cols.get("ticker")
    name_col = cols.get("name") or cols.get("security") or cols.get("company")
    weight_col = cols.get("weight") or cols.get("Weight") or cols.get("portfolio_weight")

    if ticker_col is None or weight_col is None:
        st.warning(f"Cannot render Top 10 for {wave_name} (missing Ticker/Weight columns).")
        return

    df = holdings_df.copy()
    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()

    df = df.sort_values(weight_col, ascending=False).head(10)

    df["Google Quote"] = df[ticker_col].apply(
        lambda t: f"[{t}]({google_quote_url(t)})"
    )

    display_cols = ["Google Quote"]
    if name_col:
        display_cols.append(name_col)
    display_cols.append(weight_col)

    display_df = df[display_cols].rename(columns={
        name_col: "Name" if name_col else name_col,
        weight_col: "Weight"
    })

    st.subheader(f"Top 10 Holdings — {wave_name}")
    st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)


# ---------- Metrics Helpers ----------

def compute_summary_metrics(perf_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    """
    Compute intraday (proxy), 30d, 60d total returns & 30d alpha.
    We *intentionally* do not show 1-year or since-inception here.
    """
    metrics: Dict[str, Optional[float]] = {
        "intraday_return": None,
        "return_30d": None,
        "return_60d": None,
        "alpha_30d": None,
    }

    if perf_df is None or perf_df.empty:
        return metrics

    last = perf_df.iloc[-1]

    def pull(col_candidates: List[str]) -> Optional[float]:
        for c in col_candidates:
            if c in perf_df.columns:
                try:
                    return float(last[c])
                except Exception:
                    continue
        return None

    # Intraday proxy: latest 1d return
    metrics["intraday_return"] = pull(["return_1d", "intraday_return", "daily_return"])
    metrics["return_30d"] = pull(["return_30d"])
    metrics["return_60d"] = pull(["return_60d"])
    metrics["alpha_30d"] = pull(["alpha_30d"])

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
    candidates = [
        os.path.join(LOGS_PERFORMANCE_DIR, "alpha_capture_matrix.csv"),
        "alpha_capture_matrix.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return None


def render_alpha_capture_tab() -> None:
    st.subheader("Alpha Capture Matrix")
    df = load_alpha_capture_matrix()
    if df is None or df.empty:
        st.info("Alpha Capture Matrix not available.")
        return
    st.dataframe(df, use_container_width=True)


# ---------- Sidebar Controls ----------

waves = get_available_waves()
if not waves:
    st.error(
        "No Waves discovered yet. Run the engine first so logs/performance files are generated."
    )
    st.stop()

st.sidebar.header("Wave & Mode")

selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

st.sidebar.markdown("---")
if st.sidebar.button("Run Engine for ALL Waves (Manual Kick)"):
    try:
        waves_engine.run_all_waves(mode=mode, debug=True)  # type: ignore
        st.sidebar.success("Engine run triggered successfully.")
    except Exception as e:
        st.sidebar.error(f"Error running engine: {e}")

st.sidebar.markdown("**Mode Definitions**")
st.sidebar.write("- **Standard**: Balanced risk vs. return.")
st.sidebar.write("- **Alpha-Minus-Beta**: Defensive, beta-targeted overlays.")
st.sidebar.write("- **Private Logic™**: Aggressive, proprietary alpha logic.")


# ---------- Main Layout ----------

col_left, col_right = st.columns([2, 1])

perf_df = load_performance_history(selected_wave)
metrics = compute_summary_metrics(perf_df)
positions_df = load_latest_positions(selected_wave)

with col_left:
    st.subheader(f"{selected_wave} — Overview")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Intraday Return", format_pct(metrics["intraday_return"]))
    with m2:
        st.metric("30-Day Total Return", format_pct(metrics["return_30d"]))
    with m3:
        st.metric("60-Day Total Return", format_pct(metrics["return_60d"]))
    with m4:
        st.metric("30-Day Alpha", format_pct(metrics["alpha_30d"]))

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
    render_top10_holdings(positions_df, selected_wave)

st.markdown("---")

tab1, tab2 = st.tabs(["Wave Details", "Alpha Capture Matrix"])

with tab1:
    st.subheader(f"{selected_wave} — Detailed Positions")
    if positions_df is not None and not positions_df.empty:
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No detailed positions available yet for this Wave.")

with tab2:
    render_alpha_capture_tab()
    