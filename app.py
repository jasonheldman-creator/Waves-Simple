import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from urllib.parse import quote_plus

try:
    import streamlit as st
except Exception:
    class _Dummy:
        def __getattr__(self, name):
            def f(*a, **k):
                pass
            return f
    st = _Dummy()

# Try to import the engine (optional but preferred)
try:
    import waves_engine  # type: ignore
    HAS_ENGINE = True
except Exception:
    waves_engine = None
    HAS_ENGINE = False

APP_TITLE = "WAVES Intelligence™ Institutional Console — Vector 1.5"
APP_SUBTITLE = (
    "11-Wave Rotation • Alpha-Minus-Beta • Private Logic™ • WaveScore™ • SmartSafe™ • UAPV™ Preview"
)

LOGS_DIR = "logs"
LOGS_POSITIONS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_PERFORMANCE_DIR = os.path.join(LOGS_DIR, "performance")
HUMAN_OVERRIDE_DIR = os.path.join(LOGS_DIR, "human_overrides")
ENGINE_LOG_DIR = os.path.join(LOGS_DIR, "engine")

WAVESCORE_CANDIDATES = [
    os.path.join(LOGS_PERFORMANCE_DIR, "wavescore_summary.csv"),
    "wavescore_summary.csv",
]

ENGINE_ACTIVITY_CANDIDATES = [
    os.path.join(ENGINE_LOG_DIR, "engine_activity.csv"),
    os.path.join(ENGINE_LOG_DIR, "engine_log.csv"),
    "engine_activity.csv",
]

MATCH_DEBUG: Dict[str, Dict[str, Optional[str]]] = {
    "performance": {},
    "positions": {},
}

# ----------------- Wave lineup -----------------

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
    "Small Cap Growth Wave": {
        "category": "Small Cap Growth",
        "benchmark": "IWM",
        "tagline": "High-octane small-cap growth with disciplined risk controls.",
    },
    "Small to Mid Cap Growth Wave": {
        "category": "SMID Growth",
        "benchmark": "IJH",
        "tagline": "Blended small–mid cap growth with smoother ride.",
    },
    "Future Power & Energy Wave": {
        "category": "Thematic Equity",
        "benchmark": "XLE",
        "tagline": "Future power, infrastructure, renewables, and next-gen energy.",
    },
    "Quantum Computing Wave": {
        "category": "Thematic Equity",
        "benchmark": "QQQ",
        "tagline": "Quantum, AI, and deep-tech acceleration.",
    },
    "Clean Transit-Infrastructure Wave": {
        "category": "Thematic Equity",
        "benchmark": "IDEV",
        "tagline": "Clean transit, infrastructure, and mobility.",
    },
    "AI Wave": {
        "category": "Thematic Equity",
        "benchmark": "AI Basket",
        "tagline": "Pure AI exposure across chips, cloud, and software leaders.",
    },
    "Infinity Wave": {
        "category": "Flagship Multi-Theme",
        "benchmark": "ACWI",
        "tagline": "Flagship multi-theme alpha engine — “Tesla Roadster” Wave.",
    },
    "International Developed Wave": {
        "category": "Global Equity",
        "benchmark": "EFA",
        "tagline": "Developed international exposure with adaptive overlays.",
    },
    "Emerging Markets Wave": {
        "category": "Global Equity",
        "benchmark": "EEM",
        "tagline": "Emerging markets growth engine with risk discipline.",
    },
}

# ----------------- Basic helpers -----------------


def ensure_dirs() -> None:
    os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(HUMAN_OVERRIDE_DIR, exist_ok=True)
    os.makedirs(ENGINE_LOG_DIR, exist_ok=True)


def normalize_for_match(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    for ch in ["&", "-", "_"]:
        s = s.replace(ch, " ")
    for token in ["wave", "portfolio", "positions", "position", "performance", "daily"]:
        s = s.replace(token, " ")
    return " ".join(s.split())


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


def canonical_prefix(wave_name: str) -> str:
    return wave_name.replace(" ", "_")


def find_best_performance_path(wave_name: str) -> Optional[str]:
    MATCH_DEBUG["performance"][wave_name] = None

    prefix = canonical_prefix(wave_name)
    exact = os.path.join(LOGS_PERFORMANCE_DIR, f"{prefix}_performance_daily.csv")
    if os.path.exists(exact):
        MATCH_DEBUG["performance"][wave_name] = exact
        return exact

    if not os.path.isdir(LOGS_PERFORMANCE_DIR):
        return None

    target = normalize_for_match(wave_name)
    if not target:
        return None
    target_tokens = set(target.split())
    best_score = 0.0
    best_path: Optional[str] = None

    for fname in os.listdir(LOGS_PERFORMANCE_DIR):
        if not fname.endswith(".csv"):
            continue
        base = fname.replace(".csv", "")
        base_norm = normalize_for_match(base)
        b_tokens = set(base_norm.split())
        if not b_tokens:
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

    prefix = canonical_prefix(wave_name) + "_positions_"
    candidates = [
        f for f in os.listdir(LOGS_POSITIONS_DIR)
        if f.startswith(prefix) and f.endswith(".csv")
    ]
    if not candidates:
        return None

    best_dt = None
    best_file = None
    for fname in candidates:
        full = os.path.join(LOGS_POSITIONS_DIR, fname)
        dt = parse_date_from_positions_filename(fname)
        if dt is None:
            try:
                dt = datetime.fromtimestamp(os.path.getmtime(full))
            except Exception:
                continue
        if best_dt is None or dt > best_dt:
            best_dt = dt
            best_file = full

    if best_file is None:
        return None

    MATCH_DEBUG["positions"][wave_name] = best_file
    return best_file


def get_available_waves() -> List[str]:
    """
    Discover waves from the engine / wave_weights, but ALWAYS include
    the full EQUITY_WAVES lineup so all equity Waves appear in the UI.
    """
    waves: Set[str] = set()

    # 1) Try to get waves from the engine / weights file
    if HAS_ENGINE and hasattr(waves_engine, "load_wave_weights"):
        try:
            weights_df = waves_engine.load_wave_weights()  # type: ignore
            if weights_df is not None and not weights_df.empty:
                cols = {c.lower(): c for c in weights_df.columns}
                wave_col = cols.get("wave")
                if wave_col:
                    discovered = (
                        weights_df[wave_col].dropna().astype(str).unique().tolist()
                    )
                    waves.update(discovered)
        except Exception:
            pass

    # 2) ALWAYS include the full equity lineup
    waves = waves.union(set(EQUITY_WAVES))

    # 3) Sort and return
    return sorted(list(waves))


# ----------------- CSV loaders -----------------


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


def load_engine_activity() -> Optional[pd.DataFrame]:
    for path in ENGINE_ACTIVITY_CANDIDATES:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except Exception:
                continue
    return None


# ----------------- Sandbox fallback (if no logs) -----------------


def demo_positions_for_wave(wave: str) -> pd.DataFrame:
    if wave == "Clean Transit-Infrastructure Wave":
        tickers = ["TSLA", "NIO", "RIVN", "CHPT", "BLNK", "F", "GM", "CAT", "DE", "UNP"]
    elif wave == "Quantum Computing Wave":
        tickers = ["NVDA", "AMD", "IBM", "QCOM", "AVGO", "TSM", "MSFT", "GOOGL"]
    elif wave == "S&P 500 Wave":
        tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "BRK.B",
            "JPM",
            "JNJ",
            "XOM",
            "PG",
        ]
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

    return pd.DataFrame(
        {
            "wave": [wave] * n,
            "ticker": tickers,
            "name": tickers,
            "weight": weights,
        }
    )


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
    elif wave in [
        "Quantum Computing Wave",
        "Future Power & Energy Wave",
        "Clean Transit-Infrastructure Wave",
        "AI Wave",
    ]:
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

    df = pd.DataFrame(
        {
            "date": dates,
            "nav": wave_nav,
            "nav_risk": wave_nav,
            "return_1d": wave_ret,
            "bench_nav": bench_nav,
            "bench_return_1d": bench_ret,
        }
    )

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
    df["regime"] = "SANDBOX"
    return df


def generate_sandbox_logs_if_missing(
    wave: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    perf_path = find_best_performance_path(wave)
    pos_path = find_latest_positions_path(wave)

    if perf_path is not None or pos_path is not None:
        return load_performance_history(wave), load_latest_positions(wave)

    prefix = canonical_prefix(wave)
    perf_df = demo_performance_for_wave(wave)
    pos_df = demo_positions_for_wave(wave)

    os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)

    perf_log_path = os.path.join(
        LOGS_PERFORMANCE_DIR, f"{prefix}_performance_daily.csv"
    )
    pos_log_path = os.path.join(
        LOGS_POSITIONS_DIR,
        f"{prefix}_positions_{datetime.today().strftime('%Y%m%d')}.csv",
    )

    perf_df.to_csv(perf_log_path, index=False)
    pos_df.to_csv(pos_log_path, index=False)

    MATCH_DEBUG["performance"][wave] = perf_log_path
    MATCH_DEBUG["positions"][wave] = pos_log_path

    return perf_df, pos_df


# ----------------- Mode overlay -----------------


def apply_mode_overlay(
    perf_df: Optional[pd.DataFrame], mode: str
) -> Optional[pd.DataFrame]:
    if perf_df is None or perf_df.empty:
        return perf_df

    df = perf_df.copy()
    if mode == "Standard":
        return df

    alpha_cols = [c for c in df.columns if c.lower().startswith("alpha_")]
    ret_cols = [c for c in df.columns if c.lower().startswith("return_")]

    if mode == "Alpha-Minus-Beta":
        scale_alpha = 0.75
        scale_return = 0.85
    elif mode == "Private Logic™":
        scale_alpha = 1.25
        scale_return = 1.10
    else:
        return df

    for col in alpha_cols:
        try:
            df[col] = df[col].astype(float) * scale_alpha
        except Exception:
            pass
    for col in ret_cols:
        try:
            df[col] = df[col].astype(float) * scale_return
        except Exception:
            pass
    return df


# ----------------- Metrics & formatting -----------------


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
        st.warning("Missing Ticker/Weight columns for holdings.")
        return
    df[tcol] = df[tcol].astype(str).str.upper()
    try:
        df[wcol] = df[wcol].astype(float)
    except Exception:
        pass
    df = df.sort_values(wcol, ascending=False).head(10)
    df["Google"] = df[tcol].apply(google_quote_url)
    show_cols: List[str] = ["Google"]
    if ncol:
        show_cols.append(ncol)
    show_cols.append(wcol)
    display_df = df[show_cols].rename(
        columns={ncol: "Name" if ncol else ncol, wcol: "Weight"}
    )
    st.subheader(f"Top 10 Holdings — {wave_name}")
    st.table(display_df)


def compute_wave_metrics(perf_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "alpha_1d": None,
        "alpha_30d": None,
        "alpha_60d": None,
        "alpha_1y": None,
        "ret_30d": None,
        "ret_60d": None,
        "ret_1y": None,
        "vol_ann": None,
        "alpha_ir": None,
        "hit_rate": None,
    }
    if perf_df is None or perf_df.empty:
        return out

    df = perf_df.copy()
    last = df.iloc[-1]
    mapping = [
        ("alpha_1d", "alpha_1d"),
        ("alpha_30d", "alpha_30d"),
        ("alpha_60d", "alpha_60d"),
        ("alpha_1y", "alpha_1y"),
        ("return_30d", "ret_30d"),
        ("return_60d", "ret_60d"),
        ("return_252d", "ret_1y"),
    ]
    for col, key in mapping:
        if col in df.columns and pd.notnull(last.get(col)):
            try:
                out[key] = float(last[col])
            except Exception:
                pass

    if "return_1d" in df.columns:
        try:
            vol = float(df["return_1d"].dropna().std()) * np.sqrt(252.0)
            out["vol_ann"] = vol
        except Exception:
            pass

    if "alpha_1d" in df.columns:
        series = df["alpha_1d"].dropna()
        if len(series) > 5:
            try:
                mean_a = float(series.mean())
                std_a = float(series.std())
                if std_a > 1e-8:
                    out["alpha_ir"] = (mean_a / std_a) * np.sqrt(252.0)
            except Exception:
                pass
            try:
                out["hit_rate"] = float((series > 0).mean())
            except Exception:
                pass

    return out


def format_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    try:
        return f"{x * 100:.2f}%"
    except Exception:
        return "N/A"


def format_number(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    try:
        return f"{x:.2f}"
    except Exception:
        return "N/A"


def compute_multi_wave_metrics(waves: List[str], mode: str) -> pd.DataFrame:
    rows = []
    for w in waves:
        perf_df = load_performance_history(w)
        if perf_df is None or perf_df.empty:
            perf_df, _ = generate_sandbox_logs_if_missing(w)
        mode_df = apply_mode_overlay(perf_df, mode)
        m = compute_wave_metrics(mode_df)
        meta = WAVE_METADATA.get(w, {})
        rows.append(
            {
                "Wave": w,
                "Category": meta.get("category", ""),
                "Benchmark": meta.get("benchmark", ""),
                "Alpha_1d": m["alpha_1d"],
                "Alpha_30d": m["alpha_30d"],
                "Alpha_60d": m["alpha_60d"],
                "Alpha_1y": m["alpha_1y"],
                "Return_30d": m["ret_30d"],
                "Return_60d": m["ret_60d"],
                "Return_1y": m["ret_1y"],
                "Vol_Ann": m["vol_ann"],
                "Alpha_IR": m["alpha_ir"],
                "Hit_Rate": m["hit_rate"],
            }
        )
    return pd.DataFrame(rows)


# ----------------- UAPV preview -----------------


def compute_uapv_preview(
    perf_df: Optional[pd.DataFrame], initial_capital: float = 100000.0
) -> Optional[pd.DataFrame]:
    if perf_df is None or perf_df.empty or "nav" not in perf_df.columns:
        return None

    df = perf_df.copy().sort_values("date")
    first_nav = float(df["nav"].iloc[0])
    if first_nav == 0:
        return None
    df["unit_price"] = df["nav"] / first_nav * 100.0

    first_unit_price = float(df["unit_price"].iloc[0])
    units_held = initial_capital / first_unit_price

    df["model_units"] = units_held
    df["model_account_value"] = df["model_units"] * df["unit_price"]

    return df[["date", "unit_price", "model_units", "model_account_value"]]


def render_uapv_section(perf_df_raw: Optional[pd.DataFrame], wave_name: str) -> None:
    st.subheader("UAPV™ Preview — Units & Flows (Model Account)")
    uapv_df = compute_uapv_preview(perf_df_raw)
    if uapv_df is None or uapv_df.empty:
        st.info("UAPV preview not available (missing NAV data).")
        return
    latest = uapv_df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Unit Price (Index = 100)", f"{latest['unit_price']:.2f}")
    col2.metric("Model Units Held", f"{latest['model_units']:.2f}")
    col3.metric("Model Account Value", f"${latest['model_account_value']:,.2f}")
    st.markdown(
        "_Preview only: 100,000 starting capital, no flows. "
        "Full UAPV uses real subscriptions/redemptions._"
    )
    st.dataframe(uapv_df.tail(30), use_container_width=True)


# ----------------- Tabs & status -----------------


def render_wavescore_tab() -> None:
    st.subheader("WAVESCORE™ v1.0 — Wave Quality Dashboard")
    df = load_wavescore_summary()
    if df is None or df.empty:
        st.info("WaveScore summary file not found yet.")
        return
    st.dataframe(df, use_container_width=True)


def render_alpha_capture_tab(waves: List[str], mode: str) -> None:
    st.subheader(f"Alpha Capture Matrix — {mode}")
    df = compute_multi_wave_metrics(waves, mode)
    if df.empty:
        st.info("No data available yet for Alpha Capture Matrix.")
        return
    disp = df.copy()
    pct_cols = [
        "Alpha_1d",
        "Alpha_30d",
        "Alpha_60d",
        "Alpha_1y",
        "Return_30d",
        "Return_60d",
        "Return_1y",
        "Hit_Rate",
    ]
    for c in pct_cols:
        disp[c] = disp[c].apply(format_pct)
    disp["Vol_Ann"] = disp["Vol_Ann"].apply(format_pct)
    disp["Alpha_IR"] = disp["Alpha_IR"].apply(format_number)
    st.dataframe(
        disp[
            [
                "Wave",
                "Category",
                "Benchmark",
                "Alpha_1d",
                "Alpha_30d",
                "Alpha_60d",
                "Alpha_1y",
                "Return_30d",
                "Return_60d",
                "Return_1y",
                "Vol_Ann",
                "Alpha_IR",
                "Hit_Rate",
            ]
        ],
        use_container_width=True,
    )


def render_human_override_tab(selected_wave: str) -> None:
    st.subheader("Human Override — View Only")
    df = load_human_overrides()
    if df is None or df.empty:
        st.info("No human overrides found.")
        return
    st.markdown("#### All Overrides")
    st.dataframe(df, use_container_width=True)
    cols = {c.lower(): c for c in df.columns}
    wave_col = cols.get("wave")
    if wave_col:
        st.markdown(f"#### Overrides for {selected_wave}")
        this_wave = df[df[wave_col] == selected_wave]
        if not this_wave.empty:
            st.dataframe(this_wave, use_container_width=True)
        else:
            st.info(f"No overrides for {selected_wave}.")


def get_last_update_time(path: Optional[str]) -> Optional[datetime]:
    if path is None or not os.path.exists(path):
        return None
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None


def render_system_status_tab(waves: List[str]) -> None:
    st.subheader("System Status — Engine & Data Health")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Engine")
        if HAS_ENGINE:
            st.success("waves_engine module loaded — engine AVAILABLE.")
        else:
            st.warning("waves_engine module NOT found — running SANDBOX only.")
        st.markdown("#### Directories")
        st.write(f"Positions: `{LOGS_POSITIONS_DIR}`")
        st.write(f"Performance: `{LOGS_PERFORMANCE_DIR}`")
        st.write(f"Overrides: `{HUMAN_OVERRIDE_DIR}`")
        st.write(f"Engine Logs: `{ENGINE_LOG_DIR}`")
    with col2:
        st.markdown("#### Latest Files per Wave")
        rows = []
        for w in waves:
            perf_path = MATCH_DEBUG["performance"].get(w) or find_best_performance_path(
                w
            )
            pos_path = MATCH_DEBUG["positions"].get(w) or find_latest_positions_path(w)
            rows.append(
                {
                    "Wave": w,
                    "Perf File": perf_path or "(none)",
                    "Positions File": pos_path or "(none)",
                    "Last Perf Update": get_last_update_time(perf_path),
                    "Last Pos Update": get_last_update_time(pos_path),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No log files detected yet.")
    st.markdown("---")
    st.markdown("#### All log CSVs")
    files = []
    for root, _, fs in os.walk("logs"):
        for f in fs:
            if f.endswith(".csv"):
                full = os.path.join(root, f)
                files.append(
                    {"File": full, "Last Modified": get_last_update_time(full)}
                )
    if files:
        st.dataframe(pd.DataFrame(files), use_container_width=True)
    else:
        st.info("No CSV files present in /logs folder.")


def render_mode_logic_tab(selected_mode: str) -> None:
    st.subheader("Mode Logic Viewer — Alpha-Minus-Beta / Private Logic™")
    st.write(f"**Mode:** {selected_mode}")
    rows = [
        {
            "Mode": "Standard",
            "Beta Target": "~1.00",
            "Risk Focus": "Balanced",
            "Notes": "Default benchmark-style exposure with moderate alpha target.",
        },
        {
            "Mode": "Alpha-Minus-Beta",
            "Beta Target": "0.78–0.90",
            "Risk Focus": "Downside control",
            "Notes": "Scaled exposure, capped beta; smoother ride with alpha over time.",
        },
        {
            "Mode": "Private Logic™",
            "Beta Target": "0.90–1.10 (dynamic)",
            "Risk Focus": "Aggressive alpha capture",
            "Notes": "Allows more tracking error and turnover when conditions warrant.",
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_engine_activity_tab() -> None:
    st.subheader("Engine Activity Log")
    df = load_engine_activity()
    if df is None or df.empty:
        st.info("No engine activity log detected yet.")
        return
    st.dataframe(df, use_container_width=True)


# ----------------- Main app -----------------


def main() -> None:
    ensure_dirs()
    st.set_page_config(page_title="WAVES Console", layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    waves = get_available_waves()
    if not waves:
        st.error("No Waves discovered (11-wave rotation).")
        return

    # Ensure logs exist for each Wave
    for w in waves:
        perf_path = find_best_performance_path(w)
        pos_path = find_latest_positions_path(w)
        if perf_path is None and pos_path is None:
            generate_sandbox_logs_if_missing(w)

    st.sidebar.header("Wave & Mode")
    selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)
    mode = st.sidebar.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    if HAS_ENGINE:
        st.sidebar.success("Engine module detected (waves_engine).")
    else:
        st.sidebar.info("Running in SANDBOX demo mode (no engine module).")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Active Wave: **{selected_wave}**")
    meta = WAVE_METADATA.get(selected_wave, {})
    if meta.get("tagline"):
        st.sidebar.caption(meta["tagline"])
    if meta.get("benchmark"):
        st.sidebar.caption(f"Benchmark: **{meta['benchmark']}**")

    if HAS_ENGINE and hasattr(waves_engine, "get_strategy_recipe"):
        try:
            cfg = waves_engine.get_strategy_recipe(selected_wave)  # type: ignore
            st.sidebar.markdown("#### Strategy Snapshot")
            st.sidebar.caption(f"Style: {cfg.get('style')}")
            st.sidebar.caption(f"Universe: {cfg.get('universe')}")
            st.sidebar.caption(f"Target β: {cfg.get('target_beta')}")
        except Exception:
            pass

    perf_df_raw = load_performance_history(selected_wave)
    positions_df = load_latest_positions(selected_wave)
    if (perf_df_raw is None or perf_df_raw.empty) or (
        positions_df is None or positions_df.empty
    ):
        perf_df_raw, positions_df = generate_sandbox_logs_if_missing(selected_wave)

    perf_df_mode = apply_mode_overlay(perf_df_raw, mode)
    summary = compute_wave_metrics(perf_df_mode)

    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.subheader(f"{selected_wave} — Overview ({mode})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intraday Alpha", format_pct(summary["alpha_1d"]))
        c2.metric("30-Day Alpha", format_pct(summary["alpha_30d"]))
        c3.metric("60-Day Alpha", format_pct(summary["alpha_60d"]))
        c4.metric("1-Year Alpha", format_pct(summary["alpha_1y"]))

        if perf_df_raw is not None and not perf_df_raw.empty:
            vix_val = None
            regime_val = None
            if "vix" in perf_df_raw.columns:
                try:
                    vix_val = float(perf_df_raw["vix"].iloc[-1])
                except Exception:
                    vix_val = None
            if "risk_regime" in perf_df_raw.columns:
                regime_val = str(perf_df_raw["risk_regime"].iloc[-1])
            if vix_val is not None:
                st.caption(
                    f"VIX Ladder: **{vix_val:.1f}** → Regime: **{regime_val or 'Unknown'}**"
                )

        if (
            perf_df_raw is not None
            and not perf_df_raw.empty
            and "smartsafe_weight" in perf_df_raw.columns
        ):
            try:
                latest_safe = float(perf_df_raw["smartsafe_weight"].iloc[-1])
            except Exception:
                latest_safe = None
            latest_yield = None
            if "smartsafe_yield_annual" in perf_df_raw.columns:
                try:
                    latest_yield = float(
                        perf_df_raw["smartsafe_yield_annual"].iloc[-1]
                    )
                except Exception:
                    latest_yield = None
            if latest_safe is not None:
                if latest_yield is not None and not np.isnan(latest_yield):
                    st.caption(
                        f"SmartSafe allocation: **{latest_safe*100:.1f}%** "
                        f"(cash-like sleeve, ~{latest_yield*100:.1f}% yield)"
                    )
                else:
                    st.caption(
                        f"SmartSafe allocation: **{latest_safe*100:.1f}%** "
                        "(cash-like sleeve)"
                    )

        if (
            perf_df_mode is not None
            and not perf_df_mode.empty
            and "date" in perf_df_mode.columns
        ):
            alpha_cols = [
                c
                for c in ["alpha_30d", "alpha_60d", "alpha_1y"]
                if c in perf_df_mode.columns
            ]
            if alpha_cols:
                st.line_chart(perf_df_mode.set_index("date")[alpha_cols])

        st.markdown("###### Debug: Matched Files (raw logs)")
        st.write(
            "Performance file:",
            MATCH_DEBUG["performance"].get(selected_wave) or "(none)",
        )
        st.write(
            "Positions file:",
            MATCH_DEBUG["positions"].get(selected_wave) or "(none)",
        )

    with col_side:
        render_top10_holdings(positions_df, selected_wave)

    st.markdown("---")

    (
        tab1,
        tab2,
        tab3,
        tab4,
        tab5,
        tab6,
        tab7,
        tab8,
    ) = st.tabs(
        [
            "Wave Details",
            "Alpha Capture",
            "WaveScore",
            "System Status",
            "Human Override",
            "All Waves Snapshot",
            "Mode Logic Viewer",
            "Engine Activity Log",
        ]
    )

    with tab1:
        st.subheader(f"{selected_wave} — Detailed Positions")
        if positions_df is not None and not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No detailed positions available.")
        st.subheader(f"{selected_wave} — Raw Performance (unadjusted)")
        if perf_df_raw is not None and not perf_df_raw.empty:
            st.dataframe(perf_df_raw, use_container_width=True)
        else:
            st.info("No performance history yet for this Wave.")
        render_uapv_section(perf_df_raw, selected_wave)

    with tab2:
        render_alpha_capture_tab(waves, mode)

    with tab3:
        render_wavescore_tab()

    with tab4:
        render_system_status_tab(waves)

    with tab5:
        render_human_override_tab(selected_wave)

    with tab6:
        st.subheader(f"All Waves — Snapshot ({mode})")
        snap = compute_multi_wave_metrics(waves, mode)
        if not snap.empty:
            disp = snap.copy()
            for c in ["Alpha_1d", "Alpha_30d", "Alpha_60d", "Alpha_1y"]:
                disp[c] = disp[c].apply(format_pct)
            for c in ["Return_30d", "Return_60d", "Return_1y"]:
                disp[c] = disp[c].apply(format_pct)
            disp["Vol_Ann"] = disp["Vol_Ann"].apply(format_pct)
            disp["Alpha_IR"] = disp["Alpha_IR"].apply(format_number)
            disp["Hit_Rate"] = disp["Hit_Rate"].apply(format_pct)
            st.dataframe(disp, use_container_width=True)
        else:
            st.info("No snapshot data available.")

    with tab7:
        render_mode_logic_tab(mode)

    with tab8:
        render_engine_activity_tab()


if __name__ == "__main__":
    main()