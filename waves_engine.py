"""
waves_engine.py

Core WAVES Intelligence™ engine:
- Loads wave_weights.csv + list.csv
- Normalizes / aliases Wave names
- Computes Wave & benchmark return series
- Computes Intraday, 30D, 60D, 1Y returns & alpha
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PRICE_LOOKBACK_DAYS = 420  # ~1Y + cushion


# Wave name aliases: old CSV names -> new official names
WAVE_NAME_ALIASES: Dict[str, str] = {
    # legacy -> current
    "Crypto Income Wave": "Crypto Equity Wave (mid/large cap)",
    "Small Cap Growth Wave": "Cloud & Enterprise Software Growth Wave",
}

# Waves we have retired (rows will be ignored when loading weights)
REMOVED_WAVES = {
    "Future Power & Energy Wave",
}

# Official Wave list (10 total once AI + SmartSafe weights exist in CSV)
OFFICIAL_WAVES: List[str] = [
    "Clean Transit-Infrastructure Wave",
    "Cloud & Enterprise Software Growth Wave",
    "Crypto Equity Wave (mid/large cap)",
    "Growth Wave",
    "Income Wave",
    "Quantum Computing Wave",
    "S&P Wave",
    "Small-Mid Cap Growth Wave",
    "AI Wave",
    "SmartSafe Wave",
]

# Blended ETF benchmarks per Wave (ticker -> weight)
# NOTE: You can tweak tickers/weights here without touching the UI.
WAVE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    # Clean transit / diversified industrials + autos
    "Clean Transit-Infrastructure Wave": {
        "FIDU": 0.45,  # Industrials
        "FDIS": 0.45,  # Consumer Discretionary (autos)
        "SPY": 0.10,   # Broad market context
    },
    # Cloud & enterprise software / small-cap growth tech
    "Cloud & Enterprise Software Growth Wave": {
        "IGV": 0.60,   # Software sector
        "WCLD": 0.20,  # Cloud computing
        "SPY": 0.20,   # Broad market
    },
    # Crypto equities + spot proxy
    "Crypto Equity Wave (mid/large cap)": {
        "FBTC": 0.70,  # Spot BTC proxy (or IBIT)
        "DAPP": 0.30,  # Digital asset equity index
    },
    # General large-cap growth / diversified tech
    "Growth Wave": {
        "QQQ": 0.50,
        "IWF": 0.50,
    },
    # Dividend income / quality income
    "Income Wave": {
        "SCHD": 1.00,
    },
    # Quantum computing & advanced tech (ETF proxy)
    "Quantum Computing Wave": {
        "QTUM": 0.70,  # Defiance Quantum ETF
        "QQQ": 0.30,
    },
    # Core market beta
    "S&P Wave": {
        "SPY": 1.00,
    },
    # Small–mid cap growth
    "Small-Mid Cap Growth Wave": {
        "IWO": 0.60,  # Russell 2000 Growth
        "VOT": 0.40,  # Mid-cap growth
    },
    # AI Wave – using your earlier IGV/VGT/QQQ/SPY blend
    "AI Wave": {
        "IGV": 0.50,  # Software / AI infra
        "VGT": 0.25,  # Broad tech
        "QQQ": 0.15,  # Large-cap growth tech
        "SPY": 0.10,  # Market context
    },
    # SmartSafe Wave – short-term Treasuries / cash-like
    "SmartSafe Wave": {
        "SGOV": 0.70,  # 0–3M Treasuries
        "BIL": 0.20,   # 1–3M Treasuries
        "SHY": 0.10,   # Short-term Treasuries
    },
}


# ---------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------

@dataclass
class WaveMetrics:
    wave: str
    intraday_return: float
    intraday_alpha: float
    return_30d: float
    alpha_30d: float
    return_60d: float
    alpha_60d: float
    return_1y: float
    alpha_1y: float


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def normalize_wave_name(name: str) -> str:
    """Normalize / alias wave names so CSV + engine stay in sync."""
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    return WAVE_NAME_ALIASES.get(name, name)


def _ann_ret(series: pd.Series) -> float:
    """Convert cumulative 1Y (252 trading days) return to annualized."""
    if series.empty:
        return np.nan
    return (1.0 + series).prod() - 1.0


# ---------------------------------------------------------
# Engine
# ---------------------------------------------------------

class WavesEngine:
    def __init__(
        self,
        list_path: str = "list.csv",
        weights_path: str = "wave_weights.csv",
    ) -> None:
        self.list_path = list_path
        self.weights_path = weights_path

        self.universe: List[str] = self._load_universe()
        self.wave_weights: Dict[str, pd.Series] = self._load_wave_weights(
            self.weights_path
        )

    # ---------- Loading ----------

    def _load_universe(self) -> List[str]:
        try:
            df = pd.read_csv(self.list_path)
        except FileNotFoundError:
            return []

        if "ticker" in df.columns:
            tickers = df["ticker"]
        else:
            tickers = df.iloc[:, 0]

        return (
            tickers.astype(str)
            .str.strip()
            .dropna()
            .unique()
            .tolist()
        )

    def _load_wave_weights(self, path: str) -> Dict[str, pd.Series]:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise RuntimeError(f"wave_weights.csv not found at {path}")

        required = {"wave", "ticker", "weight"}
        if not required.issubset(df.columns):
            raise RuntimeError(
                f"wave_weights.csv must have columns {required}, "
                f"found {set(df.columns)}"
            )

        # Normalize & clean
        df["wave"] = df["wave"].apply(normalize_wave_name)
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

        df = df.dropna(subset=["ticker", "weight"])
        df = df[df["weight"] != 0.0]

        # Drop retired waves
        df = df[~df["wave"].isin(REMOVED_WAVES)]

        # Group duplicates (wave, ticker)
        df = (
            df.groupby(["wave", "ticker"], as_index=False)["weight"]
            .sum()
        )

        # Build mapping
        wave_weights: Dict[str, pd.Series] = {}
        for wave_name, sub in df.groupby("wave"):
            weights = sub.set_index("ticker")["weight"].astype(float)
            total = weights.sum()
            if total <= 0:
                continue
            weights = weights / total
            wave_weights[wave_name] = weights

        # Filter to known / official waves only, if present
        if wave_weights:
            ordered = {}
            for w in OFFICIAL_WAVES:
                if w in wave_weights:
                    ordered[w] = wave_weights[w]
            # Keep any extra custom waves at the end
            for w, s in wave_weights.items():
                if w not in ordered:
                    ordered[w] = s
            wave_weights = ordered

        if not wave_weights:
            raise RuntimeError(
                "No valid waves found in wave_weights.csv after normalization."
            )

        return wave_weights

    # ---------- Price history ----------

    @lru_cache(maxsize=1)
    def get_price_history(self) -> pd.DataFrame:
        """
        Download daily adjusted-close prices for:
        - All tickers in list.csv
        - All tickers in wave_weights
        - All benchmark ETF tickers
        """
        end = datetime.utcnow().date()
        start = end - timedelta(days=PRICE_LOOKBACK_DAYS)

        tickers = set(self.universe)

        for weights in self.wave_weights.values():
            tickers.update(weights.index.tolist())

        for blend in WAVE_BENCHMARKS.values():
            tickers.update(blend.keys())

        tickers = sorted(t for t in tickers if isinstance(t, str) and t.strip())

        if not tickers:
            return pd.DataFrame()

        data = yf.download(
            tickers,
            start=start,
            end=end + timedelta(days=1),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        # yfinance returns different shapes depending on number of tickers
        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            prices = data["Adj Close"]
        else:
            # MultiIndex: columns: (ticker, field)
            try:
                prices = data.xs("Adj Close", axis=1, level=1)
            except Exception:
                prices = data

        # Ensure 2D
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        prices = prices.ffill().bfill()
        prices.index.name = "date"

        return prices

    # ---------- Core computations ----------

    def _compute_returns(self, price_series: pd.Series) -> pd.Series:
        return price_series.pct_change().dropna()

    def _wave_return_series(
        self, wave: str, prices: pd.DataFrame
    ) -> pd.Series:
        weights = self.wave_weights[wave]

        tickers = [t for t in weights.index if t in prices.columns]
        if not tickers:
            raise RuntimeError(
                f"No price data found for any tickers in wave '{wave}'"
            )

        w = weights.loc[tickers]
        w = w / w.sum()

        sub = prices[tickers]

        # Construct synthetic NAV: weighted sum of normalized price series
        nav = (sub / sub.iloc[0] * w).sum(axis=1)

        return self._compute_returns(nav)

    def _benchmark_return_series(
        self, wave: str, prices: pd.DataFrame
    ) -> pd.Series:
        blend = WAVE_BENCHMARKS.get(wave, {"SPY": 1.0})
        tickers = [t for t in blend.keys() if t in prices.columns]

        if not tickers:
            # Fallback to SPY only if everything is missing
            if "SPY" not in prices.columns:
                raise RuntimeError(
                    f"No benchmark price data for wave '{wave}' (tickers={list(blend.keys())})"
                )
            tickers = ["SPY"]
            weights = np.array([1.0], dtype=float)
        else:
            weights = np.array([blend[t] for t in tickers], dtype=float)

        weights = weights / weights.sum()
        sub = prices[tickers]

        nav = (sub / sub.iloc[0]) @ weights
        nav = pd.Series(nav, index=sub.index)

        return self._compute_returns(nav)

    def _window_stats(
        self,
        wave_rets: pd.Series,
        bench_rets: pd.Series,
        days: int,
    ) -> Tuple[float, float]:
        """Return (wave cumulative return, alpha) over last N trading days."""
        if wave_rets.empty or bench_rets.empty:
            return np.nan, np.nan

        wr = wave_rets.iloc[-days:]
        br = bench_rets.iloc[-days:]

        wr, br = wr.align(br, join="inner")
        if wr.empty:
            return np.nan, np.nan

        w_cum = (1.0 + wr).prod() - 1.0
        b_cum = (1.0 + br).prod() - 1.0
        return float(w_cum), float(w_cum - b_cum)

    # ---------- Public API ----------

    def compute_all_metrics(self) -> Dict[str, WaveMetrics]:
        prices = self.get_price_history()
        if prices.empty:
            raise RuntimeError("No price history available.")

        metrics: Dict[str, WaveMetrics] = {}

        for wave in self.wave_weights.keys():
            # Compute return series
            wave_rets = self._wave_return_series(wave, prices)
            bench_rets = self._benchmark_return_series(wave, prices)

            # 1D (intraday), 30D, 60D, 252D (~1Y)
            intraday_ret, intraday_alpha = self._window_stats(
                wave_rets, bench_rets, 1
            )
            ret_30, alpha_30 = self._window_stats(wave_rets, bench_rets, 30)
            ret_60, alpha_60 = self._window_stats(wave_rets, bench_rets, 60)
            ret_252, alpha_252 = self._window_stats(wave_rets, bench_rets, 252)

            metrics[wave] = WaveMetrics(
                wave=wave,
                intraday_return=intraday_ret,
                intraday_alpha=intraday_alpha,
                return_30d=ret_30,
                alpha_30d=alpha_30,
                return_60d=ret_60,
                alpha_60d=alpha_60,
                return_1y=ret_252,
                alpha_1y=alpha_252,
            )

        return metrics