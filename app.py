import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from urllib.parse import quote_plus

try:
    import streamlit as st
except Exception:  # for environments without Streamlit during linting
    class _Dummy:
        def __getattr__(self, name):
            def f(*a, **k): 
                pass
            return f
    st = _Dummy()

# Try to import your engine (optional)
try:
    import waves_engine  # type: ignore
    HAS_ENGINE = True
except Exception:
    waves_engine = None
    HAS_ENGINE = False

APP_TITLE = "WAVES Intelligence™ Institutional Console — Vector1"
APP_SUBTITLE = "Adaptive Portfolio Waves™ • Alpha-Minus-Beta • Private Logic™ • SmartSafe™"

LOGS_POSITIONS_DIR = os.path.join("logs", "positions")
LOGS_PERFORMANCE_DIR = os.path.join("logs", "performance")
HUMAN_OVERRIDE_DIR = os.path.join("logs", "human_overrides")

ALPHA_CAPTURE_CANDIDATES = [
    os.path.join(LOGS_PERFORMANCE_DIR, "alpha_capture_matrix.csv"),
    "alpha_capture_matrix.csv",
]

WAVESCORE_CANDIDATES = [
    os.path.join(LOGS_PERFORMANCE_DIR, "wavescore_summary.csv"),
    "wavescore_summary.csv",
]

# Simple metadata (kept small so it pastes cleanly on mobile)
WAVE_METADATA: Dict[str, Dict[str, str]] = {
    "S&P 500 Wave": {
        "category": "Core Equity",
        "benchmark": "SPY",
        "tagline": "Core S&P 500 exposure with adaptive overlays.",
    },
    "Growth Wave": {
        "category": "Growth Equity",
        "benchmark": "QQQ",
        "tagline": "High-growth exposure tuned for volatility and drawdowns.",
    },
    "Quantum Computing Wave": {
        "category": "Thematic Equity",
        "benchmark": "QQQ",
        "tagline": "Quantum, AI, and deep-tech acceleration.",
    },
    "Clean Transit & Infrastructure Wave": {
        "category": "Thematic Equity",
        "benchmark": "IDEV",
        "tagline": "Clean transit, infrastructure and mobility.",
    },
    "Income Wave": {
        "category": "Income / Fixed Income",
        "benchmark": "AGG",
        "tagline": "Income-oriented engine with downside awareness.",
    },
    "SmartSafe Wave": {
        "category": "SmartSafe™ Stable",
        "benchmark": "BIL",
        "tagline": "SmartSafe™ cash-equivalent Wave with 3-mode structure.",
    },
}


# ---------- Path helpers ----------

def ensure_dirs() -> None:
    os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(HUMAN_OVERRIDE_DIR, exist_ok=True)


def safe_wave_to_file_prefix(wave_name: str) -> str:
    return wave_name.replace(" ", "_")


def performance_path_for_wave(wave: str) -> Optional[str]:
    safe = safe_wave_to_file_prefix(wave)
    direct = os.path.join(LOGS_PERFORMANCE_DIR, f"{safe}_performance_daily.csv")
    if os.path.exists(direct):
        return direct
    if not os.path.isdir(LOGS_PERFORMANCE_DIR):
        return None
    for fname in os.listdir(LOGS_PERFORMANCE_DIR):
        if fname.endswith(".csv") and fname.startswith(safe + "_performance"):
            return os.path.join(LOGS_PERFORMANCE_DIR, fname)
    return None


def parse_date_from_positions_filename(fname: str) -> Optional[datetime]:
    base = os.path.basename(fname)
    stem = base.replace(".csv", "")
    parts = stem.split("_")
    if not parts:
        return None
    candidate = parts[-1]
    if len(candidate) == 8 and candidate.isdigit():
        try:
            return datetime.strptime(candidate, "%Y%m%d")
        except Exception:
            return None
    return None


def latest_positions_path_for_wave(wave: str) -> Optional[str]:
    if not os.path.isdir(LOGS_POSITIONS_DIR):
        return None
    safe = safe_wave_to_file_prefix(wave)
    candidates: List[Tuple[datetime, str]] = []
    for fname in os.listdir(LOGS_POSITIONS_DIR):
        if fname.endswith(".csv") and fname.startswith(safe + "_positions_"):
            dt = parse_date_from_positions_filename(fname)
            if dt is not None:
                candidates.append((dt, fname))
    if not candidates:
        return None
    latest = max(candidates, key=lambda x: x[0])[1]
    return os.path.join(LOGS_POSITIONS_DIR, latest)


# ---------- Data loaders ----------

def get_available_waves() -> List[str]:
    waves: set[str] = set()

    # From performance logs
    if os.path.isdir(LOGS_PERFORMANCE_DIR):
        for fname in os.listdir(LOGS_PERFORMANCE_DIR):
            if fname.endswith(".csv") and "_performance" in fname:
                base = fname.replace(".csv", "")
                wave_part = base.split("_performance")[0]
                waves.add(wave_part.replace("_", " ").strip())

    # Fallback to weights via engine
    if not waves and HAS_ENGINE and hasattr(waves_engine, "load_wave_weights"):
        try:
            weights_df = waves_engine.load_wave_weights()  # type: ignore
            if weights_df is not None and not weights_df.empty:
                cols = {c.lower(): c for c in weights_df.columns}
                wave_col = cols.get("wave")
                if wave_col:
                    waves.update(weights_df[wave_col].dropna().astype(str).unique().tolist())
        except Exception:
            pass

    # Last resort: metadata keys
    if not waves:
        waves.update(WAVE_METADATA.keys())

    return sorted(waves)


def load_performance_history(wave: str) -> Optional[pd.DataFrame]:
    path = performance_path_for_wave(wave)
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
        except Exception:
            pass
    return df


def load_latest_positions(wave: str) -> Optional[pd.DataFrame]:
    path = latest_positions_path_for_wave(wave)
    if path is None or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_alpha_capture_matrix() -> Optional[pd.DataFrame]:
    for path in ALPHA_CAPTURE_CANDIDATES:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except Exception:
                continue
    return None


def load_wavescore_summary() -> Optional[pd.DataFrame]:
    for path in WAVESCORE_CANDIDATES:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except Exception:
                continue
    return None


# ---------- Google Finance links & metrics ----------

def google_quote_url(ticker: str) -> str:
    if not isinstance(ticker, str):
        return "https://www.google.com/finance"
    clean = ticker.strip().upper()
    if not clean:
        return "https://www.google.com/finance"
    return f"https://www.google.com/finance?q={quote_plus(clean)}"


def render_top10_holdings(holdings_df: Optional[pd.DataFrame], wave_name: str) -> None:
    if holdings_df is None or holdings_df.empty:
        st.info(f"No holdings available for {wave_name}.")
        return

    df = holdings_df.copy()
    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker")
    name_col = cols.get("name") or cols.get("security") or cols.get("company")
    weight_col = cols.get("weight") or cols.get("portfolio_weight") or cols.get("weight_%")

    if ticker_col is None or weight_col is None:
        st.warning(f"Cannot render Top 10 for {wave_name} (missing Ticker/Weight columns).")
        return

    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
    try:
        df[weight_col] = df[weight_col].astype(float)
    except Exception:
        pass

    df = df.sort_values(weight_col, ascending=False).head(10)
    df["Google Quote"] = df[ticker_col].apply(lambda t: f"[{t}]({google_quote_url(t)})")

    display_cols: List[str] = ["Google Quote"]
    if name_col:
        display_cols.append(name_col)
    display_cols.append(weight_col)

    display_df = df[display_cols].rename(columns={
        name_col: "Name" if name_col else name_col,
        weight_col: "Weight",
    })

    st.subheader(f"Top 10 Holdings — {wave_name}")
    st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)


def compute_summary_metrics(perf_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "intraday_return": None,
        "return_30d": None,
        "return_60d": None,
        "alpha_30d": None,
    }

    if perf_df is None or perf_df.empty:
        return metrics

    last = perf_df.iloc[-1]

    def pull(cands: List[str]) -> Optional[float]:
        for c in cands:
            if c in perf_df.columns:
                try:
                    return float(last[c])
                except Exception:
                    continue
        return None

    metrics["intraday_return"] = pull(["return_1d", "intraday_return", "daily_return", "r_1d"])
    metrics["return_30d"] = pull(["return_30d", "r_30d", "total_return_30d"])
    metrics["return_60d"] = pull(["return_60d", "r_60d", "total_return_60d"])
    metrics["alpha_30d"] = pull(["alpha_30d", "alpha30", "alpha_1m", "alpha_30"])

    return metrics


def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{x * 100:.2f}%"
    except Exception:
        return "N/A"


def compute_multi_wave_snapshot(waves: List[str]) -> pd.DataFrame:
    records = []
    for wave in waves:
        perf_df = load_performance_history(wave)
        m = compute_summary_metrics(perf_df)
        meta = WAVE_METADATA.get(wave, {})
        records.append({
            "Wave": wave,
            "Category": meta.get("category", ""),
            "Benchmark": meta.get("benchmark", ""),
            "Intraday": m["intraday_return"],
            "30d Return": m["return_30d"],
            "60d Return": m["return_60d"],
            "30d Alpha": m["alpha_30d"],
        })
    return pd.DataFrame(records)


# ---------- Tabs ----------

def render_wavescore_tab() -> None:
    st.subheader("WAVESCORE™ v1.0 — Wave Quality Dashboard")
    df = load_wavescore_summary()
    if df is None or df.empty:
        st.info("WaveScore summary file not found yet.")
        return
    st.dataframe(df, use_container_width=True)


def render_alpha_capture_tab() -> None:
    st.subheader("Alpha Capture Matrix")
    df = load_alpha_capture_matrix()
    if df is None or df.empty:
        st.info("Alpha Capture Matrix not yet generated.")
        return
    st.dataframe(df, use_container_width=True)


def get_last_update_time(path: Optional[str]) -> Optional[datetime]:
    if path is None or not os.path.exists(path):
        return None
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts)
    except Exception:
        return None


def render_system_status_tab(waves: List[str]) -> None:
    st.subheader("System Status — Engine & Data Health")

    engine_col, logs_col = st.columns(2)

    with engine_col:
        st.markdown("#### Engine")
        if HAS_ENGINE:
            st.success("waves_engine module detected.")
        else:
            st.error("waves_engine module NOT found.")

        st.markdown("#### Directories")
        st.write(f"Logs - Positions: `{LOGS_POSITIONS_DIR}`")
        st.write(f"Logs - Performance: `{LOGS_PERFORMANCE_DIR}`")

    with logs_col:
        st.markdown("#### Latest Updates per Wave")
        rows = []
        for wave in waves:
            perf_path = performance_path_for_wave(wave)
            pos_path = latest_positions_path_for_wave(wave)
            rows.append({
                "Wave": wave,
                "Last Perf Update": get_last_update_time(perf_path),
                "Last Positions Update": get_last_update_time(pos_path),
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No updates found yet.")

    st.markdown("---")
    st.markdown("#### Files Present in /logs")
    present_files = []
    for root, _, files in os.walk("logs"):
        for f in files:
            if f.endswith(".csv"):
                full_path = os.path.join(root, f)
                present_files.append({
                    "File": full_path,
                    "Last Modified": get_last_update_time(full_path),
                })
    if present_files:
        df_files = pd.DataFrame(present_files)
        st.dataframe(df_files, use_container_width=True)
    else:
        st.info("No log CSV files detected in /logs.")


# ---------- Main app ----------

def main() -> None:
    ensure_dirs()

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    waves = get_available_waves()
    if not waves:
        st.error("No Waves discovered yet.")
        return

    st.sidebar.header("Wave & Mode")
    selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)

    mode = st.sidebar.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    st.sidebar.markdown("---")

    if HAS_ENGINE and hasattr(waves_engine, "run_all_waves"):
        if st.sidebar.button("Run Engine for ALL Waves (Manual Kick)"):
            try:
                waves_engine.run_all_waves(mode=mode, debug=True)  # type: ignore
                st.sidebar.success("Engine run triggered successfully.")
            except Exception as e:
                st.sidebar.error(f"Error running engine: {e}")
    else:
        st.sidebar.info("Engine module not loaded or run_all_waves() not available here.")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Active Wave: **{selected_wave}**")
    if selected_wave in WAVE_METADATA and WAVE_METADATA[selected_wave].get("tagline"):
        st.sidebar.caption(WAVE_METADATA[selected_wave]["tagline"])

    col_left, col_right = st.columns([2, 1])

    perf_df = load_performance_history(selected_wave)
    metrics = compute_summary_metrics(perf_df)
    positions_df = load_latest_positions(selected_wave)

    with col_left:
        st.subheader(f"{selected_wave} — Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Intraday Return", format_pct(metrics["intraday_return"]))
        m2.metric("30-Day Return", format_pct(metrics["return_30d"]))
        m3.metric("60-Day Return", format_pct(metrics["return_60d"]))
        m4.metric("30-Day Alpha", format_pct(metrics["alpha_30d"]))

        if perf_df is not None and not perf_df.empty and "date" in perf_df.columns:
            chart_cols = [c for c in ["return_30d", "return_60d", "alpha_30d"] if c in perf_df.columns]
            if chart_cols:
                chart_data = perf_df.set_index("date")[chart_cols]
                st.line_chart(chart_data)

    with col_right:
        render_top10_holdings(positions_df, selected_wave)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Wave Details", "Alpha Capture", "WaveScore", "System Status"]
    )

    with tab1:
        st.subheader(f"{selected_wave} — Detailed Positions")
        if positions_df is not None and not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No detailed positions available.")
        st.markdown("### Raw Performance History")
        if perf_df is not None and not perf_df.empty:
            st.dataframe(perf_df, use_container_width=True)
        else:
            st.info("No performance history yet.")

    with tab2:
        render_alpha_capture_tab()

    with tab3:
        render_wavescore_tab()

    with tab4:
        render_system_status_tab(waves)


if __name__ == "__main__":
    main()