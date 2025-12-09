"""
WAVES Intelligence™ Engine
Clean rebuild – v1.2.0

- Auto-discovers Waves from list.csv and wave_weights.csv
- Fetches live prices via yfinance
- Computes portfolio & benchmark returns
- Generates alpha capture matrices for:
    • Standard
    • Alpha-Minus-Beta
    • Private Logic™
- Writes simple daily performance logs
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# Engine metadata / constants
# ---------------------------------------------------------------------

ENGINE_VERSION = "1.2.0"
ENGINE_NAME = "WAVES Engine – Clean Rebuild"

ROOT = Path(".")
LOG_ROOT = ROOT / "logs"
POSITIONS_DIR = LOG_ROOT / "positions"
PERF_DIR = LOG_ROOT / "performance"
ENGINE_LOG_DIR = LOG_ROOT / "engine"

TRADING_DAYS_1Y = 252

# Fallback benchmark if none provided
DEFAULT_BENCH = "SPY"

# Optional overrides by Wave name (can be expanded)
BENCHMARK_OVERRIDES = {
    "S&P 500 Wave": "SPY",
    "Infinity Wave": "SPY",
    "AI Wave": "QQQ",
    "Quantum Computing Wave": "QQQ",
    "Clean Transit-Infrastructure Wave": "IYT",
    "Future Power & Energy Wave": "XLE",
    "SmartSafe Wave": "BIL",  # cash-like
}


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class WaveDefinition:
    name: str
    category: str
    benchmark: str


@dataclass
class EngineRunResult:
    as_of: datetime
    engine_version: str
    wave_list: pd.DataFrame
    alpha_capture: Dict[str, pd.DataFrame]
    top_holdings: pd.DataFrame
    system_status: Dict[str, str]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (LOG_ROOT, POSITIONS_DIR, PERF_DIR, ENGINE_LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------
# Loading Waves & Weights
# ---------------------------------------------------------------------

def load_wave_list(list_path: str | Path = "list.csv") -> pd.DataFrame:
    """
    Load wave metadata.

    Expected columns (case-insensitive, any of):
      - Wave / wave / name / wave_name
      - Category (optional)
      - Benchmark (optional)
    """
    path = Path(list_path)
    df = _read_csv_safe(path)
    if df is None:
        # Minimal fallback – this should almost never be used
        return pd.DataFrame(
            [
                {"Wave": "Growth Wave", "Category": "Growth Equity", "Benchmark": "SPY"},
            ]
        )

    df = _normalize_colnames(df)

    # Wave name column
    wave_col_candidates = ["wave", "name", "wave_name"]
    wave_col = next((c for c in wave_col_candidates if c in df.columns), None)
    if wave_col is None:
        raise ValueError(
            f"list.csv must have a wave/name column (one of {wave_col_candidates})."
        )
    df.rename(columns={wave_col: "wave"}, inplace=True)

    # Category
    if "category" not in df.columns:
        df["category"] = "Uncategorized"

    # Benchmark
    if "benchmark" not in df.columns:
        df["benchmark"] = DEFAULT_BENCH

    # Apply overrides
    def apply_bench_override(row):
        wname = str(row["wave"]).strip()
        return BENCHMARK_OVERRIDES.get(wname, row["benchmark"])

    df["benchmark"] = df.apply(apply_bench_override, axis=1)

    # Clean
    df["wave"] = df["wave"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["benchmark"] = df["benchmark"].astype(str).str.strip().str.upper()

    df = df[["wave", "category", "benchmark"]].drop_duplicates().reset_index(drop=True)
    return df


def load_wave_weights(weights_path: str | Path = "wave_weights.csv") -> pd.DataFrame:
    """
    Load holdings for all Waves.

    Expected columns (case-insensitive, any of):
      - wave / portfolio
      - ticker / symbol
      - weight / w
    """
    path = Path(weights_path)
    df = _read_csv_safe(path)
    if df is None:
        raise ValueError(
            "wave_weights.csv could not be read – make sure it exists and has data."
        )

    df = _normalize_colnames(df)

    # Wave column
    wave_col_candidates = ["wave", "portfolio"]
    wave_col = next((c for c in wave_col_candidates if c in df.columns), None)
    if wave_col is None:
        raise ValueError(
            f"wave_weights.csv must have a wave/portfolio column (one of {wave_col_candidates})."
        )
    df.rename(columns={wave_col: "wave"}, inplace=True)

    # Ticker column
    ticker_col_candidates = ["ticker", "symbol"]
    ticker_col = next((c for c in ticker_col_candidates if c in df.columns), None)
    if ticker_col is None:
        raise ValueError(
            f"wave_weights.csv must have a ticker/symbol column (one of {ticker_col_candidates})."
        )
    df.rename(columns={ticker_col: "ticker"}, inplace=True)

    # Weight column
    weight_col_candidates = ["weight", "w"]
    weight_col = next((c for c in weight_col_candidates if c in df.columns), None)
    if weight_col is None:
        raise ValueError(
            f"wave_weights.csv must have a weight column (one of {weight_col_candidates})."
        )
    df.rename(columns={weight_col: "weight"}, inplace=True)

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    # Drop 0 weights
    df = df[df["weight"] != 0].copy()

    # Normalize weights per Wave
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda x: x / x.abs().sum() if x.abs().sum() != 0 else x
    )

    return df[["wave", "ticker", "weight"]]


# ---------------------------------------------------------------------
# Price & Return calculations
# ---------------------------------------------------------------------

def _fetch_prices(tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.Series]:
    """
    Fetch daily adjusted close for all tickers from yfinance.
    Returns a dict: {ticker: price_series}
    """
    if not tickers:
        return {}

    # yfinance can download multiple tickers at once
    data = yf.download(
        tickers=list(set(tickers)),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    prices: Dict[str, pd.Series] = {}

    # If it's a MultiIndex (ticker, field)
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                s = data[ticker]["Close"].dropna()
                s.name = ticker
                prices[ticker] = s
            except Exception:
                continue
    else:
        # Single ticker case
        s = data["Close"].dropna()
        for ticker in tickers:
            prices[ticker] = s

    return prices


def _portfolio_returns(
    weights: pd.DataFrame, prices: Dict[str, pd.Series], wave_name: str
) -> pd.Series:
    """
    Build daily portfolio return series for given wave.
    """
    w_df = weights[weights["wave"] == wave_name]
    tickers = w_df["ticker"].unique().tolist()
    if not tickers:
        return pd.Series(dtype=float)

    aligned: List[pd.Series] = []
    w_vec: List[float] = []

    for _, row in w_df.iterrows():
        t = row["ticker"]
        w = row["weight"]
        s = prices.get(t)
        if s is None or s.empty:
            continue
        aligned.append(s)
        w_vec.append(w)

    if not aligned:
        return pd.Series(dtype=float)

    df = pd.concat(aligned, axis=1)
    df.columns = list(range(df.shape[1]))  # simple integer columns
    w_arr = np.array(w_vec).reshape(-1, 1)

    # Daily percentage returns
    rets = df.pct_change().dropna()
    # Weighted sum across columns
    port_ret = rets.dot(w_arr).squeeze()
    port_ret.name = wave_name
    return port_ret


def _window_return(series: pd.Series, days: int) -> float:
    if series is None or series.empty or len(series) < 2:
        return np.nan
    end = series.index.max()
    start = end - timedelta(days=days)
    # Find first index >= start
    s = series.loc[series.index >= start]
    if s.empty:
        return np.nan
    first = s.iloc[0]
    last = s.iloc[-1]
    if first == 0:
        return np.nan
    return float(last / first - 1.0)


def _mode_transform(port_ret: pd.Series, mode: str) -> pd.Series:
    """
    Simple deterministic transformation so modes are distinct.

    - Standard: 1.00x exposure
    - Alpha-Minus-Beta: 0.85x exposure (defensive)
    - Private Logic™: 1.15x exposure, lightly clipped
    """
    if port_ret is None or port_ret.empty:
        return port_ret

    if mode == "Standard":
        return port_ret
    elif mode == "Alpha-Minus-Beta":
        return 0.85 * port_ret
    elif mode == "Private Logic":
        scaled = 1.15 * port_ret
        # clip to avoid absurd backtest tails
        return scaled.clip(lower=-0.2, upper=0.2)
    else:
        return port_ret


# ---------------------------------------------------------------------
# Alpha capture computation
# ---------------------------------------------------------------------

def compute_alpha_matrices(
    waves: pd.DataFrame,
    weights: pd.DataFrame,
    as_of: Optional[datetime] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Core engine: given wave list + weights, compute alpha matrices
    for each mode.

    Returns:
        {
          "Standard": df,
          "Alpha-Minus-Beta": df,
          "Private Logic": df,
        }
    """
    if as_of is None:
        as_of = datetime.utcnow()

    start = as_of - timedelta(days=400)  # enough history for 1y windows
    all_tickers: List[str] = sorted(weights["ticker"].unique().tolist())

    # Include benchmark tickers
    bench_tickers = waves["benchmark"].unique().tolist()
    all_tickers = sorted(set(all_tickers + bench_tickers))

    prices = _fetch_prices(all_tickers, start=start, end=as_of + timedelta(days=1))

    # Precompute benchmark return series
    bench_series: Dict[str, pd.Series] = {}
    for _, row in waves.iterrows():
        bench = row["benchmark"]
        if bench not in bench_series:
            s = prices.get(bench)
            if s is None or s.empty:
                continue
            bench_series[bench] = s.pct_change().dropna()

    modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

    matrices: Dict[str, List[Dict[str, object]]] = {m: [] for m in modes}

    for _, row in waves.iterrows():
        wave_name = row["wave"]
        bench = row["benchmark"]
        category = row["category"]

        port_ret = _portfolio_returns(weights, prices, wave_name)
        b_ret = bench_series.get(bench)
        if port_ret.empty or b_ret is None or b_ret.empty:
            # still append a placeholder row so UI stays aligned
            for mode in modes:
                matrices[mode].append(
                    {
                        "Wave": wave_name,
                        "Category": category,
                        "Alpha_1d": np.nan,
                        "Alpha_30d": np.nan,
                        "Alpha_60d": np.nan,
                        "Alpha_1y": np.nan,
                        "Return_1d": np.nan,
                        "Return_30d": np.nan,
                        "Return_60d": np.nan,
                        "Return_1y": np.nan,
                        "Benchmark": bench,
                    }
                )
            continue

        # Align portfolio & benchmark dates
        combined = pd.concat([port_ret, b_ret], axis=1, join="inner").dropna()
        combined.columns = ["port", "bench"]
        if combined.empty:
            continue

        for mode in modes:
            m_ret = _mode_transform(combined["port"], mode)

            r_1d = _window_return(m_ret, 1)
            r_30 = _window_return(m_ret, 30)
            r_60 = _window_return(m_ret, 60)
            r_1y = _window_return(m_ret, 365)

            b_1d = _window_return(combined["bench"], 1)
            b_30 = _window_return(combined["bench"], 30)
            b_60 = _window_return(combined["bench"], 60)
            b_1y = _window_return(combined["bench"], 365)

            matrices[mode].append(
                {
                    "Wave": wave_name,
                    "Category": category,
                    "Alpha_1d": r_1d - b_1d if not np.isnan(r_1d) and not np.isnan(b_1d) else np.nan,
                    "Alpha_30d": r_30 - b_30 if not np.isnan(r_30) and not np.isnan(b_30) else np.nan,
                    "Alpha_60d": r_60 - b_60 if not np.isnan(r_60) and not np.isnan(b_60) else np.nan,
                    "Alpha_1y": r_1y - b_1y if not np.isnan(r_1y) and not np.isnan(b_1y) else np.nan,
                    "Return_1d": r_1d,
                    "Return_30d": r_30,
                    "Return_60d": r_60,
                    "Return_1y": r_1y,
                    "Benchmark": bench,
                }
            )

    alpha_capture: Dict[str, pd.DataFrame] = {}
    for mode in modes:
        df = pd.DataFrame(matrices[mode])
        if not df.empty:
            # Sort by Wave name for stable UI
            df = df.sort_values("Wave").reset_index(drop=True)
        alpha_capture[mode] = df

    return alpha_capture


# ---------------------------------------------------------------------
# Top-10 holdings per Wave
# ---------------------------------------------------------------------

def build_top_holdings(weights: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return a table of top holdings per wave with Google Finance links.
    """
    df = weights.copy()
    df["abs_weight"] = df["weight"].abs()
    df = df.sort_values(["wave", "abs_weight"], ascending=[True, False])

    df = df.groupby("wave").head(top_n).reset_index(drop=True)
    df["Google_Finance_URL"] = df["ticker"].apply(
        lambda t: f"https://www.google.com/finance/quote/{t}:NASDAQ"
    )

    df.rename(columns={"wave": "Wave", "ticker": "Ticker", "weight": "Weight"}, inplace=True)
    return df[["Wave", "Ticker", "Weight", "Google_Finance_URL"]]


# ---------------------------------------------------------------------
# Logging helper (simple)
# ---------------------------------------------------------------------

def _log_engine_run(as_of: datetime, note: str = "") -> None:
    _ensure_dirs()
    log_path = ENGINE_LOG_DIR / "engine_runs.csv"
    row = {
        "timestamp_utc": as_of.isoformat(),
        "engine_version": ENGINE_VERSION,
        "note": note,
    }
    try:
        if log_path.exists():
            df = pd.read_csv(log_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(log_path, index=False)
    except Exception:
        # logging must never break engine
        pass


# ---------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------

def run_full_engine(as_of: Optional[datetime] = None) -> EngineRunResult:
    """
    Main entry point used by app.py.

    - Loads waves + weights
    - Fetches prices
    - Computes alpha matrices for all three modes
    - Builds top-10 holdings table
    - Logs engine run
    """
    if as_of is None:
        as_of = datetime.utcnow()

    _ensure_dirs()

    waves_df = load_wave_list("list.csv")
    weights_df = load_wave_weights("wave_weights.csv")

    alpha_capture = compute_alpha_matrices(waves_df, weights_df, as_of=as_of)
    top_holdings = build_top_holdings(weights_df, top_n=10)

    _log_engine_run(as_of, note="run_full_engine")

    system_status = {
        "Engine": "AVAILABLE",
        "Engine_Version": ENGINE_VERSION,
        "As_Of_UTC": as_of.isoformat(timespec="seconds"),
        "Waves_Discovered": str(len(waves_df)),
    }

    return EngineRunResult(
        as_of=as_of,
        engine_version=ENGINE_VERSION,
        wave_list=waves_df,
        alpha_capture=alpha_capture,
        top_holdings=top_holdings,
        system_status=system_status,
    )


# Convenience for manual testing
if __name__ == "__main__":
    result = run_full_engine()
    print(f"{ENGINE_NAME} {ENGINE_VERSION} – {result.as_of:%Y-%m-%d}")
    for mode, df in result.alpha_capture.items():
        print("\nMode:", mode)
        print(df.head())