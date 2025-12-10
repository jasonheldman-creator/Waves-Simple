# waves_engine.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf


# --------- Configuration ---------

WEIGHTS_CSV = "wave_weights.csv"

# Benchmark blends per Wave (can be tuned later without touching core logic)
# Weights must sum to ~1.0; they will be renormalized defensively.
BENCHMARK_MAP: Dict[str, Dict[str, float]] = {
    # Core beta
    "S&P Wave": {"SPY": 1.0},

    # Growth / style
    # You had asked for a QQQ + IWF blend for truer large-cap growth.
    "Growth Wave": {"QQQ": 0.5, "IWF": 0.5},

    # Small / mid-cap growth proxies
    "Small-Mid Cap Growth Wave": {"VTWG": 0.5, "SLYG": 0.5},

    # Clean transit & infrastructure – industrials + consumer disc + SPY context
    "Clean Transit-Infrastructure Wave": {"FIDU": 0.45, "FDIS": 0.45, "SPY": 0.10},

    # Cloud & enterprise software growth
    "Cloud & Enterprise Software Growth Wave": {"IGV": 0.60, "WCLD": 0.20, "SPY": 0.20},

    # Crypto equity (mid / large cap): 70% spot BTC proxy + 30% digital asset equities
    "Crypto Equity Wave (mid/large cap)": {"FBTC": 0.70, "DAPP": 0.30},

    # Quality income – SCHD as anchor
    "Income Wave": {"SCHD": 1.0},

    # Quantum computing – QTUM is closest liquid proxy
    "Quantum Computing Wave": {"QTUM": 1.0},

    # AI Wave – software + broad tech + QQQ growth tilt
    "AI Wave": {"IGV": 0.40, "VGT": 0.30, "QQQ": 0.20, "SPY": 0.10},

    # SmartSafe Wave – cash-like ladder (treated as its own benchmark)
    "SmartSafe Wave": {"SGOV": 0.70, "BIL": 0.20, "SHY": 0.10},
}


@dataclass
class WaveMetrics:
    name: str
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    alpha_daily: pd.Series
    stats: Dict[str, float]


class WavesEngine:
    """
    Loads wave weights, fetches price data, and computes Wave-level
    returns + benchmark-relative alpha.
    """

    def __init__(
        self,
        weights_csv: str = WEIGHTS_CSV,
        min_history_days: int = 252 * 3,  # request ~3y of history
    ) -> None:
        self.weights_csv = Path(weights_csv)
        self.min_history_days = min_history_days
        self._warnings: List[str] = []

    # ---------------- Core public API ---------------- #

    def compute_all_metrics(self) -> Tuple[Dict[str, WaveMetrics], List[str]]:
        """
        Main entry point.

        Returns
        -------
        metrics_dict: dict wave_name -> WaveMetrics
        warnings: list of human-readable diagnostics
        """
        self._warnings = []

        weights_df = self._load_and_clean_weights()
        if weights_df.empty:
            self._warnings.append("wave_weights.csv is empty or could not be parsed.")
            return {}, self._warnings

        all_price_tickers = sorted(set(weights_df["ticker"].tolist()) |
                                   set(self._all_benchmark_tickers()))

        prices = self._download_prices(all_price_tickers)
        if prices.empty:
            self._warnings.append(
                "No usable price data returned from yfinance for any ticker."
            )
            return {}, self._warnings

        returns = prices.pct_change().dropna(how="all")

        metrics: Dict[str, WaveMetrics] = {}
        for wave_name, group in weights_df.groupby("wave"):
            wm = self._compute_wave_metrics(wave_name, group, returns)
            if wm is not None:
                metrics[wave_name] = wm

        return metrics, self._warnings

    # ---------------- Internal helpers ---------------- #

    def _load_and_clean_weights(self) -> pd.DataFrame:
        if not self.weights_csv.exists():
            self._warnings.append(
                f"weights file {self.weights_csv} not found in repository."
            )
            return pd.DataFrame(columns=["wave", "ticker", "weight"])

        raw = pd.read_csv(self.weights_csv)

        expected_cols = {"wave", "ticker", "weight"}
        missing = expected_cols - set(raw.columns)
        if missing:
            self._warnings.append(
                f"wave_weights.csv is missing columns: {', '.join(sorted(missing))}"
            )
            return pd.DataFrame(columns=["wave", "ticker", "weight"])

        df = raw.copy()
        df["wave"] = df["wave"].astype(str).str.strip()
        df["ticker"] = (
            df["ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B, etc.
        )
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

        # Drop rows with obviously bad data
        df = df.dropna(subset=["wave", "ticker", "weight"])
        df = df[df["weight"] > 0]

        # Renormalize weights within each wave defensively
        df["weight"] = df.groupby("wave")["weight"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else x
        )
        return df

    def _all_benchmark_tickers(self) -> List[str]:
        tickers: List[str] = []
        for blend in BENCHMARK_MAP.values():
            tickers.extend(blend.keys())
        # Normalize dots to dashes, upper-case
        tickers = [t.upper().replace(".", "-") for t in tickers]
        return tickers

    def _download_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Download adjusted close prices for all tickers.
        Any all-NaN columns are dropped; missing tickers are logged as warnings.
        """
        if not tickers:
            return pd.DataFrame()

        try:
            data = yf.download(
                tickers=list(set(tickers)),
                period="max",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            self._warnings.append(f"yfinance.download failed: {e}")
            return pd.DataFrame()

        # yfinance returns either a DataFrame with columns or a Series-like "Adj Close"
        if isinstance(data, pd.DataFrame) and ("Adj Close" in data.columns):
            prices = data["Adj Close"].copy()
        else:
            prices = data.copy()

        # Single-ticker edge case
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        # Standardize columns: upper-case, dot->dash
        prices.columns = [
            str(c).upper().replace(".", "-") for c in prices.columns
        ]

        # Drop all-NaN columns, log which tickers vanished
        valid_cols = prices.columns[prices.notna().any(axis=0)]
        missing = sorted(set(tickers) - set(valid_cols))
        if missing:
            self._warnings.append(
                f"No usable price data for tickers (dropped): {', '.join(missing)}"
            )
        prices = prices[valid_cols]

        return prices

    def _compute_wave_metrics(
        self,
        wave_name: str,
        weights_df: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> WaveMetrics | None:
        # Map tickers for this Wave to columns we actually have
        tickers = [t for t in weights_df["ticker"].tolist() if t in returns.columns]

        if not tickers:
            self._warnings.append(
                f"Wave '{wave_name}' has no tickers with price data; skipping."
            )
            return None

        w = weights_df.set_index("ticker")["weight"].reindex(tickers).fillna(0.0)
        if w.sum() <= 0:
            self._warnings.append(
                f"Wave '{wave_name}' has non-positive total weights after cleaning; skipping."
            )
            return None
        w = w / w.sum()

        wave_rets = (returns[tickers] * w).sum(axis=1)

        # Align benchmark
        bench_rets = self._compute_benchmark_returns(wave_name, returns, wave_rets.index)

        alpha = wave_rets - bench_rets

        stats = self._compute_stats(alpha, wave_rets, bench_rets)
        return WaveMetrics(
            name=wave_name,
            portfolio_returns=wave_rets,
            benchmark_returns=bench_rets,
            alpha_daily=alpha,
            stats=stats,
        )

    def _compute_benchmark_returns(
        self,
        wave_name: str,
        returns: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        blend = BENCHMARK_MAP.get(wave_name)
        if not blend:
            # No specific benchmark: flat zero benchmark (alpha = raw return)
            self._warnings.append(
                f"No benchmark blend defined for '{wave_name}'. Using 0% line."
            )
            return pd.Series(0.0, index=index)

        bench_tickers_all = [t.upper().replace(".", "-") for t in blend.keys()]
        bench_tickers = [t for t in bench_tickers_all if t in returns.columns]

        if not bench_tickers:
            self._warnings.append(
                f"Benchmark tickers for '{wave_name}' not found in price data; using 0% line."
            )
            return pd.Series(0.0, index=index)

        weights = (
            pd.Series(blend)
            .rename(index=lambda x: x.upper().replace(".", "-"))
            .reindex(bench_tickers)
            .fillna(0.0)
        )
        if weights.sum() <= 0:
            self._warnings.append(
                f"Benchmark weights for '{wave_name}' sum to 0; using 0% line."
            )
            return pd.Series(0.0, index=index)

        weights = weights / weights.sum()
        bench_rets = (returns[bench_tickers] * weights).sum(axis=1)
        bench_rets = bench_rets.reindex(index).fillna(0.0)
        return bench_rets

    def _compute_stats(
        self,
        alpha: pd.Series,
        wave_rets: pd.Series,
        bench_rets: pd.Series,
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {}

        if len(wave_rets) >= 60:
            r_60 = (1.0 + wave_rets.iloc[-60:]).prod() - 1.0
            stats["60D"] = float(r_60)
        else:
            stats["60D"] = float("nan")

        if len(alpha) >= 252:
            a_1y = (1.0 + alpha.iloc[-252:]).prod() - 1.0
            stats["Alpha 1Y"] = float(a_1y)
        else:
            stats["Alpha 1Y"] = float("nan")

        # Optional: you can add more here (vol, maxDD, IR, etc.)
        return stats


# Convenience factory used by app.py
def build_engine() -> WavesEngine:
    return WavesEngine()