"""
WAVES Intelligence™ – Engine v3
--------------------------------
Single-file engine used by the Streamlit Institutional Console.

Responsibilities:
- Load wave_weights.csv
- Normalize Wave names to the official 10-Wave lineup
- Fetch prices via yfinance for holdings + benchmarks
- Compute daily returns and Wave vs. benchmark alpha
- Expose:
    - build_engine()  -> WavesEngine
    - load_metrics()  -> pd.DataFrame for the UI
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Tuple

import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


# ---------------------------------------------------------------------------
# Config – paths
# ---------------------------------------------------------------------------

ROOT = Path(".")
WEIGHTS_CSV = ROOT / "wave_weights.csv"

LOGS_DIR = ROOT / "logs"
PERF_DIR = LOGS_DIR / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Official 10-Wave lineup + normalization
# ---------------------------------------------------------------------------

STANDARD_WAVES = {
    "S&P Wave": "S&P Wave",
    "Growth Wave": "Growth Wave",
    "Small-Mid Cap Growth Wave": "Small-Mid Cap Growth Wave",
    "Clean Transit-Infrastructure Wave": "Clean Transit-Infrastructure Wave",
    "Cloud & Enterprise Software Growth Wave": "Cloud & Enterprise Software Growth Wave",
    "Crypto Equity Wave (mid/large cap)": "Crypto Equity Wave (mid/large cap)",
    "Income Wave": "Income Wave",
    "Quantum Computing Wave": "Quantum Computing Wave",
    "AI Wave": "AI Wave",
    "SmartSafe Wave": "SmartSafe Wave",
}


def normalize_wave_name(name: str) -> str:
    """Normalize raw wave name from CSV to the official 10-Wave label."""
    raw = name.strip()
    for key in STANDARD_WAVES:
        if raw.lower() == key.lower():
            return STANDARD_WAVES[key]
    raise RuntimeError(f"Unknown wave name in CSV: '{name}'")


# ---------------------------------------------------------------------------
# Benchmarks – ETF blends per Wave (weights sum to 1.0)
# NOTE: only using tickers that exist in yfinance.
# ---------------------------------------------------------------------------

WAVE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    # Broad market
    "S&P Wave": {"SPY": 1.0},

    # Large-cap growth (megacap tech tilt)
    # User preference: QQQ + IWF blend
    "Growth Wave": {"QQQ": 0.6, "IWF": 0.4},

    # Small / mid-cap growth – Russell 2000 growth proxies
    "Small-Mid Cap Growth Wave": {"VTWG": 0.5, "SLYG": 0.5},

    # Clean transit / infra – Industrials + Consumer Discretionary + SPY
    "Clean Transit-Infrastructure Wave": {"FIDU": 0.45, "FDIS": 0.45, "SPY": 0.10},

    # Cloud / Enterprise software – IGV + WCLD + SPY
    "Cloud & Enterprise Software Growth Wave": {"IGV": 0.6, "WCLD": 0.2, "SPY": 0.2},

    # Crypto Equity (mid / large cap) – 70% spot BTC proxy, 30% digital assets equity
    "Crypto Equity Wave (mid/large cap)": {"FBTC": 0.7, "DAPP": 0.3},

    # Dividend / quality income
    "Income Wave": {"SCHD": 1.0},

    # Quantum computing – tech growth proxies
    "Quantum Computing Wave": {"IYW": 0.7, "QQQ": 0.3},

    # AI Wave – diversified AI / software growth blend
    "AI Wave": {"IGV": 0.5, "VGT": 0.25, "QQQ": 0.15, "SPY": 0.10},

    # SmartSafe – ultra-short ladder (matches your SGOV/BIL/SHY holdings)
    "SmartSafe Wave": {"SGOV": 0.7, "BIL": 0.2, "SHY": 0.1},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WaveMetrics:
    wave: str
    start_date: dt.date
    end_date: dt.date

    # Return metrics
    ret_60d: float
    ret_1y: float
    ret_intraday: float

    # Benchmark metrics
    bench_ret_60d: float
    bench_ret_1y: float
    bench_ret_intraday: float

    # Alpha metrics
    alpha_60d: float
    alpha_1y: float
    alpha_intraday: float

    # Diagnostics
    n_holdings: int
    n_days: int


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class WavesEngine:
    def __init__(
        self,
        weights_df: pd.DataFrame,
        price_start: dt.date,
        price_end: dt.date,
    ):
        """
        weights_df: columns [wave, ticker, weight] with normalized wave names
        """
        self.weights_df = weights_df.copy()
        self.price_start = price_start
        self.price_end = price_end

        # Build wave -> {ticker: weight}
        self.wave_to_weights: Dict[str, Dict[str, float]] = (
            self._build_wave_weight_map(self.weights_df)
        )

        # Universe of tickers to download (holdings + all benchmarks)
        universe = set(self.weights_df["ticker"].unique())

        for wave, bm in WAVE_BENCHMARKS.items():
            for tkr in bm.keys():
                universe.add(tkr)

        self.universe: List[str] = sorted(universe)

        # Lazy-loaded price frame (Adj Close)
        self._price_df: pd.DataFrame | None = None
        self._ret_df: pd.DataFrame | None = None

    # ---------------------- utility builders ---------------------- #

    @staticmethod
    def _build_wave_weight_map(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for wave, grp in df.groupby("wave"):
            tickers = grp["ticker"].tolist()
            weights = grp["weight"].astype(float).tolist()
            w = np.array(weights, dtype=float)
            if w.sum() <= 0:
                raise RuntimeError(f"Wave '{wave}' has non-positive total weight.")
            w = w / w.sum()
            result[wave] = {tkr: float(wi) for tkr, wi in zip(tickers, w)}
        return result

    # ---------------------- price loading ---------------------- #

    def _load_prices(self) -> pd.DataFrame:
        """Download adjusted close prices for the full universe."""
        if self._price_df is not None:
            return self._price_df

        data = yf.download(
            tickers=self.universe,
            start=self.price_start,
            end=self.price_end + dt.timedelta(days=1),
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )

        # yfinance returns a different shape depending on ticker count
        if isinstance(data.columns, pd.MultiIndex):
            # Extract only "Adj Close"
            close_frames = []
            for tkr in self.universe:
                if (tkr, "Adj Close") in data.columns:
                    s = data[(tkr, "Adj Close")].rename(tkr)
                    close_frames.append(s)
            if not close_frames:
                raise RuntimeError("No price data found for any tickers in universe.")
            prices = pd.concat(close_frames, axis=1)
        else:
            # Single ticker
            prices = data.rename(columns={"Adj Close": self.universe[0]})

        prices = prices.dropna(how="all")
        if prices.empty:
            raise RuntimeError("Price DataFrame is empty after dropping NaNs.")

        self._price_df = prices
        return prices

    def _load_returns(self) -> pd.DataFrame:
        if self._ret_df is not None:
            return self._ret_df
        prices = self._load_prices()
        rets = prices.pct_change().dropna(how="all")
        self._ret_df = rets
        return rets

    # ---------------------- helpers ---------------------- #

    def _weighted_series(
        self, weights: Dict[str, float], ret_df: pd.DataFrame
    ) -> pd.Series:
        """Return a weighted return series given ticker weights."""
        cols = [t for t in weights.keys() if t in ret_df.columns]
        if not cols:
            raise RuntimeError(
                f"No price data found for tickers: {list(weights.keys())}"
            )
        w = np.array([weights[t] for t in cols], dtype=float)
        w = w / w.sum()
        sub = ret_df[cols]
        series = (sub * w).sum(axis=1)
        return series

    def _window_return(self, series: pd.Series, days: int) -> float:
        if series.empty:
            return np.nan
        if days <= 0:
            return np.nan
        tail = series.tail(days)
        if tail.empty:
            return np.nan
        cum = (1 + tail).prod() - 1
        return float(cum)

    # ---------------------- core metric computation ---------------------- #

    def compute_wave_metrics(self, wave: str) -> WaveMetrics:
        if wave not in self.wave_to_weights:
            raise RuntimeError(f"Wave '{wave}' not found in weights map.")

        rets = self._load_returns()

        # holdings series
        wave_weights = self.wave_to_weights[wave]
        wave_series = self._weighted_series(wave_weights, rets)

        # benchmark series
        bm_cfg = WAVE_BENCHMARKS.get(wave)
        if bm_cfg is None:
            raise RuntimeError(f"No benchmark configured for wave '{wave}'.")
        bench_series = self._weighted_series(bm_cfg, rets)

        # Align indexes
        wave_series, bench_series = wave_series.align(bench_series, join="inner")
        if wave_series.empty:
            raise RuntimeError(f"Empty aligned series for wave '{wave}'.")

        n_days = len(wave_series)
        start_date = wave_series.index[0].date()
        end_date = wave_series.index[-1].date()

        # Intraday = 1 trading day
        ret_intraday = self._window_return(wave_series, 1)
        bench_intraday = self._window_return(bench_series, 1)
        alpha_intraday = ret_intraday - bench_intraday

        # 60D
        ret_60d = self._window_return(wave_series, 60)
        bench_60d = self._window_return(bench_series, 60)
        alpha_60d = ret_60d - bench_60d

        # 1Y (252 trading days)
        ret_1y = self._window_return(wave_series, 252)
        bench_1y = self._window_return(bench_series, 252)
        alpha_1y = ret_1y - bench_1y

        return WaveMetrics(
            wave=wave,
            start_date=start_date,
            end_date=end_date,
            ret_60d=ret_60d,
            ret_1y=ret_1y,
            ret_intraday=ret_intraday,
            bench_ret_60d=bench_60d,
            bench_ret_1y=bench_1y,
            bench_ret_intraday=bench_intraday,
            alpha_60d=alpha_60d,
            alpha_1y=alpha_1y,
            alpha_intraday=alpha_intraday,
            n_holdings=len(wave_weights),
            n_days=n_days,
        )

    def compute_all_metrics(self) -> Dict[str, WaveMetrics]:
        results: Dict[str, WaveMetrics] = {}
        for wave in sorted(self.wave_to_weights.keys()):
            metrics = self.compute_wave_metrics(wave)
            results[wave] = metrics
        return results


# ---------------------------------------------------------------------------
# Engine builder + public helpers for Streamlit
# ---------------------------------------------------------------------------

def _load_weights_csv() -> pd.DataFrame:
    if not WEIGHTS_CSV.exists():
        raise RuntimeError(f"weights file not found at {WEIGHTS_CSV}")

    df = pd.read_csv(WEIGHTS_CSV)

    required_cols = {"wave", "ticker", "weight"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(
            f"wave_weights.csv must contain columns {required_cols}, "
            f"found {list(df.columns)}"
        )

    # Normalize names
    df["wave"] = df["wave"].apply(normalize_wave_name)

    # Strip ticker whitespace
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    return df


def build_engine() -> WavesEngine:
    """Create a WavesEngine with a reasonable price window."""
    today = dt.date.today()
    start = today - dt.timedelta(days=365 * 3)  # 3y window for now

    weights_df = _load_weights_csv()
    engine = WavesEngine(weights_df=weights_df, price_start=start, price_end=today)
    return engine


def load_metrics() -> pd.DataFrame:
    """
    Public entry for Streamlit.

    Returns a DataFrame with:
        Wave | 60D | Alpha 1Y | Ret 60D | Ret 1Y | Intraday | Alpha Intraday | ...
    UI can pick/format whichever columns it wants.
    """
    engine = build_engine()
    metrics_dict = engine.compute_all_metrics()

    rows: List[Dict] = []
    for m in metrics_dict.values():
        d = asdict(m)
        rows.append(d)

    df = pd.DataFrame(rows)

    # Human-friendly columns used by the dashboard table
    df["60D"] = df["alpha_60d"]     # main alpha column
    df["Alpha 1Y"] = df["alpha_1y"]

    # Sort in a stable, nice order
    df = df.sort_values("wave").reset_index(drop=True)

    # Optional: write daily snapshot log
    snap_path = PERF_DIR / f"wave_metrics_snapshot_{dt.date.today().isoformat()}.csv"
    try:
        df.to_csv(snap_path, index=False)
    except Exception:
        # Logging failure should never break the app
        pass

    return df