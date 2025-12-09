import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
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

# Try to import your engine
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

# Debug tracker: which files each wave is using
MATCH_DEBUG: Dict[str, Dict[str, Optional[str]]] = {
    "performance": {},
    "positions": {},
}

# ---- Wave metadata (display names) ----
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
    "Clean Transit-Infrastructure Wave": {
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

# ---------- Basic helpers ----------

def ensure_dirs() -> None:
    os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(HUMAN_OVERRIDE_DIR, exist_ok=True)


def normalize_for_match(s: str) -> str:
    """Normalize wave names / filenames for fuzzy matching."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    for ch in ["&", "-", "_"]:
        s = s.replace(ch, " ")
    for token in ["wave", "portfolio", "positions", "performance", "daily"]:
        s = s.replace(token, " ")
    s = " ".join(s.split())
    return s


def parse_date_from_positions_filename(fname: str) -> Optional[datetime]:
    """Try to parse YYYYMMDD from ..._YYYYMMDD.csv"""
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


def find_best_performance_path(wave_name: str) -> Optional[str]:
    """
    Fuzzy match performance file for a wave.
    Stores result in MATCH_DEBUG['performance'][wave_name].
    """
    MATCH_DEBUG["performance"][wave_name] = None

    if not os.path.isdir(LOGS_PERFORMANCE_DIR):
        return None
    target = normalize_for_match(wave_name)
    if not target:
        return None

    best_score = 0.0
    best_path: Optional[str] = None
    target_tokens = set(target.split())

    for fname in os.listdir(LOGS_PERFORMANCE_DIR):
        if not fname.endswith(".csv") or "_performance" not in fname:
            continue
        base = fname.replace(".csv", "")
        base_norm = normalize_for_match(base)
        if not base_norm:
            continue
        b_tokens = set(base_norm.split())
        if not target_tokens or not b_tokens:
            continue
        score = len(target_tokens & b_tokens) / len(target_tokens | b_tokens)
        if score > best_score:
            best_score = score
            best_path = os.path.join(LOGS_PERFORMANCE_DIR, fname)

    # threshold for “good enough” match
    if best_score < 0.2:
        MATCH_DEBUG["performance"][wave_name] = None
        return None

    MATCH_DEBUG["performance"][wave_name] = best_path
    return best_path


def find_latest_positions_path(wave_name: str) -> Optional[str]:
    """
    Fuzzy match LATEST positions file for a wave.
    Stores result in MATCH_DEBUG['positions'][wave_name].
    """
    MATCH_DEBUG["positions"][wave_name] = None

    if not os.path.isdir(LOGS_POSITIONS_DIR):
        return None
    target = normalize_for_match(wave_name)
    if not target:
        return None

    candidates: List[Tuple[datetime, str, float]] = []
    target_tokens = set(target.split())

    for fname in os.listdir(LOGS_POSITIONS_DIR):
        if not fname.endswith(".csv") or "_positions" not in fname:
            continue
        base = fname.replace(".csv", "")
        base_norm = normalize_for_match(base)
        if not base_norm:
            continue
        b_tokens = set(base_norm.split())
        if not target_tokens or not b_tokens:
            continue
        score = len(target_tokens & b_tokens) / len(target_tokens | b_tokens)
        if score < 0.2:
            continue

        full_path = os.path.join(LOGS_POSITIONS_DIR, fname)
        dt = parse_date_from_positions_filename(fname)
        if dt is None:
            try:
                dt = datetime.fromtimestamp(os.path.getmtime(full_path))
            except Exception:
                continue
        candidates.append((dt, fname, score))

    if not candidates:
        return None

    # choose latest date; if tie, highest score
    latest = max(candidates, key=lambda x: (x[0], x[2]))
    latest_fname = latest[1]
    best_path = os.path.join(LOGS_POSITIONS_DIR, latest_fname)
    MATCH_DEBUG["positions"][wave_name] = best_path
    return best_path


# ---------- Data loaders ----------

def get_available_waves() -> List[str]:
    """Waves from engine weights, falling back to metadata."""
    waves: set[str] = set()

    if HAS_ENGINE and hasattr(waves_engine, "load_wave_weights"):
        try:
            weights_df = waves_engine.load_wave_weights()  # type: ignore
            if weights_df is not None and not weights_df.empty:
                cols = {c.lower(): c for c in weights_df.columns}
                wave_col = cols.get("wave")
                if wave_col:
                    waves.update(
                        weights_df[wave_col].dropna().astype(str).unique().tolist()
                    )
        except Exception:
            pass

    if not waves:
        waves.update(WAVE_METADATA.keys())

    return sorted(waves)


def load_performance_history(wave: str) -> Optional[pd.DataFrame]:
    path = find_best_performance_path(wave)
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
    path = find_latest_positions_path(wave)
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


def load_human_overrides() -> Optional[pd.DataFrame]:
    candidates = [
        os.path.join(HUMAN_OVERRIDE_DIR, "overrides.csv"),
        os.path.join(HUMAN_OVERRIDE_DIR, "human_overrides.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except Exception:
                continue
    return None


# ---------- SANDBOX data generator (code-only fallback) ----------

def demo_positions_for_wave(wave: str) -> pd.DataFrame:
    """Generate synthetic positions for a wave if none exist."""
    if wave == "Clean Transit-Infrastructure Wave":
        tickers = ["TSLA", "NIO", "RIVN", "CHPT", "BLNK", "F", "GM", "CAT", "DE", "UNP"]
    elif wave == "Quantum Computing Wave":
        tickers = ["NVDA", "AMD", "IBM", "QCOM", "AVGO", "TSM", "MSFT", "GOOGL"]
    elif wave == "S&P 500 Wave":
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "JPM", "JNJ", "XOM", "PG"]
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]

    n = len(tickers)
    raw = np.abs(np.random.rand(n))
    weights = raw / raw.sum()

    df = pd.DataFrame({
        "wave": [wave] * n,
        "ticker": tickers,
        "name": tickers,  # simple name = ticker
        "weight": weights,
    })
    return df


def demo_performance_for_wave(wave: str, days: int = 90) -> pd.DataFrame:
    """Generate synthetic performance history for a wave if none exist."""
    end_date = datetime.today().date()
    dates = pd.bdate_range(end=end_date, periods=days)
    n = len(dates)

    # simple random walk around 0.08 annualized, 15% vol (approx)
    daily_mu = 0.08 / 252.0
    daily_sigma = 0.15 / np.sqrt(252.0)
    daily_ret = np.random.normal(daily_mu, daily_sigma, size=n)

    nav = 100 * (1 + daily_ret).cumprod()

    df = pd.DataFrame({
        "date": dates,
        "nav": nav,
        "return_1d": daily_ret,
    })

    # trailing returns
    df["return_30d"] = np.nan
    df["return_60d"] = np.nan
    for i in range(n):
        # 30 business days
        if i >= 21:
            df.loc[df.index[i], "return_30d"] = nav[i] / nav[i - 21] - 1.0
        if i >= 42:
            df.loc[df.index[i], "return_60d"] = nav[i] / nav[i - 42] - 1.0

    # simple alpha: 30d alpha = 30d return minus 2%/yr baseline
    baseline_daily = 0.02 / 252.0
    df["alpha_30d"] = df["return_30d"] - (baseline_daily * 21)

    df["wave"] = wave
    df["regime"] = "SANDBOX"

    return df


def generate_sandbox_logs_if_missing(wave: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    If a wave has no positions/performance files at all,
    generate synthetic SANDBOX data and write logs to /logs.
    """
    perf_path = find_best_performance_path(wave)
    pos_path = find_latest_positions_path(wave)

    if perf_path is not None or pos_path is not None:
        # Something already exists; don't overwrite
        perf_df = load_performance_history(wave)
        pos_df = load_latest_positions(wave)
        return perf_df, pos_df

    # No files at all: generate SANDBOX
    prefix = wave.replace(" ", "_")

    perf_df = demo_performance_for_wave(wave)
    pos_df = demo_positions_for_wave(wave)

    # Write to logs so future reads pick it up
    perf_log_path = os.path.join(LOGS_PERFORMANCE_DIR, f"{prefix}_performance_daily.csv")
    pos_log_path = os.path.join(
        LOGS_POSITIONS_DIR,
        f"{prefix}_positions_{datetime.today().strftime('%Y%m%d')}.csv",
    )

    try:
        perf_df.to_csv(perf_log_path, index=False)
        MATCH_DEBUG["performance"][wave] = perf_log_path
    except Exception:
        pass

    try:
        pos_df.to_csv(pos_log_path, index=False)
        MATCH_DEBUG["positions"][wave] = pos_log_path
    except Exception:
        pass

    return perf_df, pos_df


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
        st.info(f"No holdings available for {wave_name}. (No matching positions file found.)")
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
        # if still missing, allow SANDBOX fill-in for snapshot as well
        if perf_df is None or perf_df.empty:
            perf_df, _ = generate_sandbox_logs_if_missing(wave)
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


# ---------- Sub-tabs ----------

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


def render_human_override_tab(selected_wave: str) -> None:
    st.subheader("Human Override — View Only")
    df = load_human_overrides()
    if df is None or df.empty:
        st.info(
            "No human overrides found. If overrides are used, they will be read from "
            "`logs/human_overrides/overrides.csv`."
        )
        return

    st.markdown("#### All Overrides")
    st.dataframe(df, use_container_width=True)

    cols = {c.lower(): c for c in df.columns}
    wave_col = cols.get("wave")
    if wave_col:
        st.markdown(f"#### Overrides for {selected_wave}")
        this_wave_df = df[df[wave_col] == selected_wave]
        if not this_wave_df.empty:
            st.dataframe(this_wave_df, use_container_width=True)
        else:
            st.info(f"No overrides currently filed for {selected_wave}.")


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
            st.success("waves_engine module loaded — WAVES engine is AVAILABLE.")
        else:
            st.error("waves_engine module NOT found — console is running in SANDBOX mode only.")
        st.markdown("#### Directories")
        st.write(f"Logs - Positions: `{LOGS_POSITIONS_DIR}`")
        st.write(f"Logs - Performance: `{LOGS_PERFORMANCE_DIR}`")
        st.write(f"Human Overrides: `{HUMAN_OVERRIDE_DIR}`")

    with logs_col:
        st.markdown("#### Latest Updates per Wave")
        rows = []
        for wave in waves:
            perf_path = MATCH_DEBUG["performance"].get(wave) or find_best_performance_path(wave)
            pos_path = MATCH_DEBUG["positions"].get(wave) or find_latest_positions_path(wave)
            rows.append({
                "Wave": wave,
                "Perf File": perf_path or "(none)",
                "Positions File": pos_path or "(none)",
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

    # Engine status + control buttons
    st.sidebar.markdown("---")
    if HAS_ENGINE:
        st.sidebar.success("Engine status: ON (waves_engine loaded).")
    else:
        st.sidebar.warning("Engine status: SANDBOX (no waves_engine.py on this host).")

    if HAS_ENGINE and hasattr(waves_engine, "run_wave"):
        if st.sidebar.button(f"Run Engine for THIS Wave ({selected_wave})"):
            try:
                waves_engine.run_wave(selected_wave)  # type: ignore
                st.sidebar.success(f"Engine run triggered for {selected_wave}.")
            except Exception as e:
                st.sidebar.error(f"Error running engine for {selected_wave}: {e}")

    if HAS_ENGINE and hasattr(waves_engine, "run_all_waves"):
        if st.sidebar.button("Run Engine for ALL Waves"):
            try:
                waves_engine.run_all_waves(mode=mode, debug=True)  # type: ignore
                st.sidebar.success("Engine run triggered for ALL Waves.")
            except Exception as e:
                st.sidebar.error(f"Error running engine for ALL Waves: {e}")
    elif HAS_ENGINE:
        st.sidebar.info("run_all_waves() not available on waves_engine.")
    else:
        st.sidebar.info("Engine not loaded; using SANDBOX data where needed.")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Active Wave: **{selected_wave}**")
    if selected_wave in WAVE_METADATA and WAVE_METADATA[selected_wave].get("tagline"):
        st.sidebar.caption(WAVE_METADATA[selected_wave]["tagline"])

    # Load data (may be empty if engine never ran for this wave)
    perf_df = load_performance_history(selected_wave)
    positions_df = load_latest_positions(selected_wave)

    # Auto-kick engine ONCE if there is no data for this wave
    if HAS_ENGINE and hasattr(waves_engine, "run_wave") and (perf_df is None or positions_df is None):
        try:
            waves_engine.run_wave(selected_wave)  # type: ignore
            st.sidebar.info(f"Auto-run: engine kicked for {selected_wave}.")
            perf_df = load_performance_history(selected_wave)
            positions_df = load_latest_positions(selected_wave)
        except Exception as e:
            st.sidebar.error(f"Auto-run error for {selected_wave}: {e}")

    # If still no data, generate SANDBOX logs via code only
    if (perf_df is None or perf_df.empty) and (positions_df is None or positions_df.empty):
        st.sidebar.info(f"Generating SANDBOX data for {selected_wave} (code-only demo).")
        perf_df, positions_df = generate_sandbox_logs_if_missing(selected_wave)

    metrics = compute_summary_metrics(perf_df)

    col_left, col_right = st.columns([2, 1])

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
        else:
            st.caption("No performance history yet for this Wave.")

        # Show which files this wave is currently using
        st.markdown("###### Debug — Matched Files for This Wave")
        st.write("Performance file:", MATCH_DEBUG["performance"].get(selected_wave) or "(none)")
        st.write("Positions file:", MATCH_DEBUG["positions"].get(selected_wave) or "(none)")

    with col_right:
        render_top10_holdings(positions_df, selected_wave)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Wave Details",
            "Alpha Capture",
            "WaveScore",
            "Human Override",
            "System Status",
            "All Waves Snapshot",
        ]
    )

    with tab1:
        st.subheader(f"{selected_wave} — Detailed Positions")
        if positions_df is not None and not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No detailed positions available for this Wave.")
        st.markdown("### Raw Performance History")
        if perf_df is not None and not perf_df.empty:
            st.dataframe(perf_df, use_container_width=True)
        else:
            st.info("No performance history yet for this Wave.")

    with tab2:
        render_alpha_capture_tab()

    with tab3:
        render_wavescore_tab()

    with tab4:
        render_human_override_tab(selected_wave)

    with tab5:
        render_system_status_tab(waves)

    with tab6:
        st.subheader("All Waves — Snapshot")
        snapshot_df = compute_multi_wave_snapshot(waves)
        if not snapshot_df.empty:
            display_df = snapshot_df.copy()
            for c in ["Intraday", "30d Return", "60d Return", "30d Alpha"]:
                display_df[c] = display_df[c].apply(format_pct)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No data available yet for multi-Wave snapshot.")


if __name__ == "__main__":
    main()
    