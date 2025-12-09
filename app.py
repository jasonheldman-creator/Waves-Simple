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

# Try to import your engine
try:
    import waves_engine  # type: ignore
    HAS_ENGINE = True
except Exception:
    waves_engine = None
    HAS_ENGINE = False

APP_TITLE = "WAVES Intelligence™ Institutional Console — Vector1"
APP_SUBTITLE = "Adaptive Portfolio Waves™ • Alpha-Minus-Beta • Private Logic™"

LOGS_POSITIONS_DIR = os.path.join("logs", "positions")
LOGS_PERFORMANCE_DIR = os.path.join("logs", "performance")
HUMAN_OVERRIDE_DIR = os.path.join("logs", "human_overrides")
ENGINE_LOG_DIR = os.path.join("logs", "engine")

ALPHA_CAPTURE_CANDIDATES = [
    os.path.join(LOGS_PERFORMANCE_DIR, "alpha_capture_matrix.csv"),
    "alpha_capture_matrix.csv",
]

WAVESCORE_CANDIDATES = [
    os.path.join(LOGS_PERFORMANCE_DIR, "wavescore_summary.csv"),
    "wavescore_summary.csv",
]

ENGINE_ACTIVITY_CANDIDATES = [
    os.path.join(ENGINE_LOG_DIR, "engine_activity.csv"),
    os.path.join(ENGINE_LOG_DIR, "engine_log.csv"),
    "engine_activity.csv",
]

# Debug tracker: which files each wave is using
MATCH_DEBUG: Dict[str, Dict[str, Optional[str]]] = {
    "performance": {},
    "positions": {},
}

# ---------- Equity + Global Rotation (11 Waves) ----------

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
        "tagline": "Blended small–mid cap growth engine with smoother risk profile.",
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
        "tagline": "The flagship “Tesla Roadster” Wave — multi-theme alpha engine.",
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

# ---------- Basic helpers ----------

def ensure_dirs() -> None:
    os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(HUMAN_OVERRIDE_DIR, exist_ok=True)
    os.makedirs(ENGINE_LOG_DIR, exist_ok=True)


def normalize_for_match(s: str) -> str:
    """Normalize names / filenames for fuzzy matching."""
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
    """Fuzzy match performance file for a wave."""
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
        MATCH_DEBUG["performance"][wave_name] = None
        return None

    MATCH_DEBUG["performance"][wave_name] = best_path
    return best_path


def find_latest_positions_path(wave_name: str) -> Optional[str]:
    """Fuzzy match LATEST positions file for a wave."""
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


# ---------- Data loaders ----------

def get_available_waves() -> List[str]:
    """
    Waves from engine weights, but filtered to the 11-wave rotation.
    If engine has more, they’re ignored; if engine is missing, fallback to EQUITY_WAVES.
    """
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

    # Intersect with the 11-wave rotation
    if waves:
        waves = waves.intersection(set(EQUITY_WAVES))
    else:
        waves = set(EQUITY_WAVES)

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


# ---------- SANDBOX generator (11-wave universe) ----------

def demo_positions_for_wave(wave: str) -> pd.DataFrame:
    """Generate synthetic positions for a wave if none exist."""
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
    """
    Generate synthetic performance + benchmark history for a wave.
    Alpha = Wave return − Benchmark return over each window.
    """
    end_date = datetime.today().date()
    dates = pd.bdate_range(end=end_date, periods=days)
    n = len(dates)

    # Benchmark: ~8%/yr, ~15% vol
    bench_mu = 0.08 / 252.0
    bench_sigma = 0.15 / np.sqrt(252.0)

    # Wave alpha profile
    if wave == "S&P 500 Wave":
        alpha_mu = 0.01 / 252.0   # ~1%/yr alpha
        alpha_sigma = 0.03 / np.sqrt(252.0)
    elif wave == "Infinity Wave":
        alpha_mu = 0.04 / 252.0   # ~4%/yr alpha
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
            df.loc[df.index[i], "return_30d"] = wave_nav[i] / wave_nav[i - 21]