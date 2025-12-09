import os
from datetime import datetime
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

# Try to import your engine (optional)
try:
    import waves_engine  # type: ignore
    HAS_ENGINE = True
except Exception:
    waves_engine = None
    HAS_ENGINE = False

APP_TITLE = "WAVES Intelligence™ Institutional Console — Vector1 (Lite)"
APP_SUBTITLE = "11-Wave Rotation • Alpha Capture • Top 10 Holdings"

LOGS_POSITIONS_DIR = os.path.join("logs", "positions")
LOGS_PERFORMANCE_DIR = os.path.join("logs", "performance")

MATCH_DEBUG: Dict[str, Dict[str, Optional[str]]] = {
    "performance": {},
    "positions": {},
}

EQUITY_WAVES: List[str] = [
    "S&P 500 Wave",
    "Growth Wave",
    "Small Cap Growth Wave",
    "Small to Mid Cap Growth Wave",
    "Future Power & Energy Wave",
    "Quantum Computing Wave",
    "Clean Transit-Infrastructure Wave",
    "AI Wave",
    "Infinity Wave",
    "International Developed Wave",
    "Emerging Markets Wave",
]

WAVE_METADATA: Dict[str, Dict[str, str]] = {
    "S&P 500 Wave": {"category": "Core Equity", "benchmark": "SPY"},
    "Growth Wave": {"category": "Growth Equity", "benchmark": "QQQ"},
    "Small Cap Growth Wave": {"category": "Small Cap Growth", "benchmark": "IWM"},
    "Small to Mid Cap Growth Wave": {"category": "SMID Growth", "benchmark": "IJH"},
    "Future Power & Energy Wave": {"category": "Thematic", "benchmark": "XLE"},
    "Quantum Computing Wave": {"category": "Thematic", "benchmark": "QQQ"},
    "Clean Transit-Infrastructure Wave": {"category": "Thematic", "benchmark": "IDEV"},
    "AI Wave": {"category": "Thematic", "benchmark": "AI Basket"},
    "Infinity Wave": {"category": "Flagship", "benchmark": "ACWI"},
    "International Developed Wave": {"category": "Global", "benchmark": "EFA"},
    "Emerging Markets Wave": {"category": "Global", "benchmark": "EEM"},
}

def ensure_dirs() -> None:
    os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)

def normalize_for_match(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    for ch in ["&", "-", "_"]:
        s = s.replace(ch, " ")
    for token in ["wave", "portfolio", "positions", "position", "performance", "daily"]:
        s = s.replace(token, " ")
    s = " ".join(s.split())
    return s

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

def find_best_performance_path(wave_name: str) -> Optional[str]:
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
    if best_score < 0.2:
        return None
    MATCH_DEBUG["performance"][wave_name] = best_path
    return best_path

def find_latest_positions_path(wave_name: str) -> Optional[str]:
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
    latest = max(candidates, key=lambda x: (x[0], x[2]))
    best_path = os.path.join(LOGS_POSITIONS_DIR, latest[1])
    MATCH_DEBUG["positions"][wave_name] = best_path
    return best_path

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

def demo_positions_for_wave(wave: str) -> pd.DataFrame:
    if wave == "Clean Transit-Infrastructure Wave":
        tickers = ["TSLA", "NIO", "RIVN", "CHPT", "BLNK", "F", "GM", "CAT", "DE", "UNP"]
    elif wave == "Quantum Computing Wave":
        tickers = ["NVDA", "AMD", "IBM", "QCOM", "AVGO", "TSM", "MSFT", "GOOGL"]
    elif wave == "S&P 500 Wave":
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "JPM", "JNJ", "XOM", "PG"]
    elif wave == "AI Wave":
        tickers = ["NVDA", "MSFT", "GOOGL", "META", "AVGO", "CRM", "SNOW", "ADBE"]
    elif wave == "Future Power & Energy Wave":
        tickers = ["NEE", "ENPH", "FSLR", "XOM", "CVX", "PLUG", "SEDG"]
    elif wave == "Infinity Wave":
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "AVGO"]
    elif wave == "International Developed Wave":
        tickers = ["NOVO-B.CO", "NESN.SW", "ASML", "SONY", "BP", "BHP", "RIO"]
    elif wave == "Emerging Markets Wave":
        tickers = ["TSM", "BABA", "PDD", "INFY", "VALE", "PBR", "MELI"]
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    n = len(tickers)
    raw = np.abs(np.random.rand(n))
    weights = raw / raw.sum()
    df = pd.DataFrame({
        "wave": [wave] * n,
        "ticker": tickers,
        "name": tickers,
        "weight": weights,
    })
    return df

def demo_performance_for_wave(wave: str, days: int = 260) -> pd.DataFrame:
    end_date = datetime.today().date()
    dates = pd.bdate_range(end=end_date, periods=days)
    n = len(dates)
    bench_mu = 0.08 / 252.0
    bench_sigma = 0.15 / np.sqrt(252.0)
    if wave == "S&P 500 Wave":
        alpha_mu = 0.01 / 252.0
        alpha_sigma = 0.03 / np.sqrt(252.0)
    elif wave == "Infinity Wave":
        alpha_mu = 0.04 / 252.0
        alpha_sigma = 0.08 / np.sqrt(252.0)
    elif wave in ["Growth Wave", "Small Cap Growth Wave", "Small to Mid Cap Growth Wave"]:
        alpha_mu = 0.03 / 252.0
        alpha_sigma = 0.07 / np.sqrt(252.0)
    elif wave in ["Quantum Computing Wave", "Future Power & Energy Wave", "Clean Transit-Infrastructure Wave", "AI Wave"]:
        alpha_mu = 0.05 / 252.0
        alpha_sigma = 0.10 / np.sqrt(252.0)
    elif wave in ["International Developed Wave", "Emerging Markets Wave"]:
        alpha_mu = 0.02 / 252.0
        alpha_sigma = 0.05 / np.sqrt(252.0)
    else:
        alpha_mu = 0.03 / 252.0
        alpha_sigma = 0.06 / np.sqrt(252.0)
    bench_ret = np.random.normal(bench_mu, bench_sigma, size=n)
    alpha_noise = np.random.normal(alpha_mu, alpha_sigma, size=n)
    wave_ret = bench_ret + alpha_noise
    bench_nav = 100 * (1 + bench_ret).cumprod()
    wave_nav = 100 * (1 + wave_ret).cumprod()
    df = pd.DataFrame({
        "date": dates,
        "nav": wave_nav,
        "return_1d": wave_ret,
        "bench_nav": bench_nav,
        "bench_return_1d": bench_ret,
    })
    df["return_30d"] = np.nan
    df["return_60d"] = np.nan
    df["return_252d"] = np.nan
    df["bench_return_30d"] = np.nan
    df["bench_return_60d"] = np.nan
    df["bench_return_252d"] = np.nan
    for i in range(n):
        if i >= 21:
            df.loc[df.index[i], "return_30d"] = wave_nav[i] / wave_nav[i - 21] - 1.0
            df.loc[df.index[i], "bench_return_30d"] = bench_nav[i] / bench_nav[i - 21] - 1.0
        if i >= 42:
            df.loc[df.index[i], "return_60d"] = wave_nav[i] / wave_nav[i - 42] - 1.0
            df.loc[df.index[i], "bench_return_60d"] = bench_nav[i] / bench_nav[i - 42] - 1.0
        if i >= 252:
            df.loc[df.index[i], "return_252d"] = wave_nav[i] / wave_nav[i - 252] - 1.0
            df.loc[df.index[i], "bench_return_252d"] = bench_nav[i] / bench_nav[i - 252] - 1.0
    df["alpha_1d"] = df["return_1d"] - df["bench_return_1d"]
    df["alpha_30d"] = df["return_30d"] - df["bench_return_30d"]
    df["alpha_60d"] = df["return_60d"] - df["bench_return_60d"]
    df["alpha_1y"] = df["return_252d"] - df["bench_return_252d"]
    df["wave"] = wave
    return df

def generate_sandbox_logs_if_missing(wave: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perf_path = find_best_performance_path(wave)
    pos_path = find_latest_positions_path(wave)
    if perf_path is not None or pos_path is not None:
        perf_df = load_performance_history(wave)
        pos_df = load_latest_positions(wave)
        return perf_df, pos_df
    prefix = wave.replace(" ", "_")
    perf_df = demo_performance_for_wave(wave)
    pos_df = demo_positions_for_wave(wave)
    os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
    perf_log_path = os.path.join(LOGS_PERFORMANCE_DIR, f"{prefix}_performance_daily.csv")
    pos_log_path = os.path.join(LOGS_POSITIONS_DIR, f"{prefix}_positions_{datetime.today().strftime('%Y%m%d')}.csv")
    perf_df.to_csv(perf_log_path, index=False)
    pos_df.to_csv(pos_log_path, index=False)
    MATCH_DEBUG["performance"][wave] = perf_log_path
    MATCH_DEBUG["positions"][wave] = pos_log_path
    return perf_df, pos_df

def google_quote_url(ticker: str) -> str:
    if not isinstance(ticker, str):
        return "https://www.google.com/finance"
    clean = ticker.strip().upper()
    if not clean:
        return "https://www.google.com/finance"
    return f"https://www.google.com/finance?q={quote_plus(clean)}"

def render_top10_holdings(df: Optional[pd.DataFrame], wave_name: str) -> None:
    if df is None or df.empty:
        st.info(f"No holdings available for {wave_name}.")
        return
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ticker")
    wcol = cols.get("weight")
    ncol = cols.get("name")
    if tcol is None or wcol is None:
        st.warning("Missing ticker/weight columns.")
        return
    df[tcol] = df[tcol].astype(str).str.upper()
    try:
        df[wcol] = df[wcol].astype(float)
    except Exception:
        pass
    df = df.sort_values(wcol, ascending=False).head(10)
    df["Google"] = df[tcol].apply(lambda t: f"[{t}]({google_quote_url(t)})")
    cols_show = ["Google"]
    if ncol:
        cols_show.append(ncol)
    cols_show.append(wcol)
    st.subheader(f"Top 10 Holdings — {wave_name}")
    st.markdown(df[cols_show].to_markdown(index=False), unsafe_allow_html=True)

def compute_summary_metrics(perf_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {"alpha_1d": None, "alpha_30d": None, "alpha_60d": None, "alpha_1y": None}
    if perf_df is None or perf_df.empty:
        return metrics
    last = perf_df.iloc[-1]
    for key, col in [("alpha_1d", "alpha_1d"), ("alpha_30d", "alpha_30d"), ("alpha_60d", "alpha_60d"), ("alpha_1y", "alpha_1y")]:
        try:
            metrics[key] = float(last[col])
        except Exception:
            metrics[key] = None
    return metrics

def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "N/A"

def compute_snapshot(waves: List[str]) -> pd.DataFrame:
    rows = []
    for w in waves:
        perf_df = load_performance_history(w)
        if perf_df is None or perf_df.empty:
            perf_df, _ = generate_sandbox_logs_if_missing(w)
        m = compute_summary_metrics(perf_df)
        meta = WAVE_METADATA.get(w, {})
        rows.append({
            "Wave": w,
            "Category": meta.get("category", ""),
            "Benchmark": meta.get("benchmark", ""),
            "Intraday Alpha": m["alpha_1d"],
            "30d Alpha": m["alpha_30d"],
            "60d Alpha": m["alpha_60d"],
            "1y Alpha": m["alpha_1y"],
        })
    return pd.DataFrame(rows)

def main() -> None:
    ensure_dirs()
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    waves = sorted(EQUITY_WAVES)
    # Make sure we have data for each wave
    for w in waves:
        generate_sandbox_logs_if_missing(w)

    wave = st.sidebar.selectbox("Select Wave", waves, index=0)

    perf_df = load_performance_history(wave)
    pos_df = load_latest_positions(wave)
    if perf_df is None or perf_df.empty or pos_df is None or pos_df.empty:
        perf_df, pos_df = generate_sandbox_logs_if_missing(wave)

    metrics = compute_summary_metrics(perf_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Intraday Alpha", format_pct(metrics["alpha_1d"]))
    col2.metric("30-Day Alpha", format_pct(metrics["alpha_30d"]))
    col3.metric("60-Day Alpha", format_pct(metrics["alpha_60d"]))
    col4.metric("1-Year Alpha", format_pct(metrics["alpha_1y"]))

    if perf_df is not None and not perf_df.empty and "date" in perf_df.columns:
        chart_cols = [c for c in ["alpha_30d", "alpha_60d", "alpha_1y"] if c in perf_df.columns]
        if chart_cols:
            st.line_chart(perf_df.set_index("date")[chart_cols])

    render_top10_holdings(pos_df, wave)

    st.markdown("---")
    tab1, tab2 = st.tabs(["Wave Details", "All Waves Snapshot"])

    with tab1:
        st.subheader(f"{wave} — Positions")
        st.dataframe(pos_df, use_container_width=True)
        st.subheader(f"{wave} — Raw Performance")
        st.dataframe(perf_df, use_container_width=True)

    with tab2:
        snap = compute_snapshot(waves)
        if not snap.empty:
            disp = snap.copy()
            for c in ["Intraday Alpha", "30d Alpha", "60d Alpha", "1y Alpha"]:
                disp[c] = disp[c].apply(format_pct)
            st.dataframe(disp, use_container_width=True)

if __name__ == "__main__":
    main()