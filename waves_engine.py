"""
waves_engine.py

WAVES Intelligence™ – simplified institutional engine

- Loads wave_weights.csv
- Normalizes tickers (e.g., BRK.B -> BRK-B)
- Downloads price data via yfinance with retry + error handling
- Computes wave NAV, benchmark NAV, and alpha stats
- Exposes WavesEngine + WaveMetrics for use in Streamlit app.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------
# Ticker normalization / aliases
# ---------------------------------------------------------------------

TICKER_ALIASES: Dict[str, str] = {
    "BRK.B": "BRK-B",
    "BRK B": "BRK-B",
    # Add any other odd tickers here if yfinance is picky
}


def normalize_ticker(ticker: str) -> str:
    """Normalize raw CSV tickers to Yahoo Finance-compatible ones."""
    return TICKER_ALIASES.get(ticker.strip(), ticker.strip())


# ---------------------------------------------------------------------
# Safe yfinance download helper
# ---------------------------------------------------------------------

def safe_download(
    tickers: Set[str],
    start: datetime,
    end: datetime,
    max_retries: int = 3,
    sleep_base: float = 1.0,
) -> pd.DataFrame:
    """
    Wrapper around yf.download with:
      - ticker normalization
      - MultiIndex column cleanup
      - Retry logic for rate-limit / transient errors
      - Dropping tickers that come back all-NaN
    """
    normed = sorted({normalize_ticker(t) for t in tickers})
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[safe_download] Attempt {attempt} for tickers: {normed}")
            data = yf.download(
                normed,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )

            if data is None or data.empty:
                raise RuntimeError("yf.download returned empty DataFrame")

            # Handle MultiIndex columns: (field, ticker)
            if isinstance(data.columns, pd.MultiIndex):
                level0 = list(data.columns.levels[0])
                if "Adj Close" in level0:
                    data = data["Adj Close"]
                elif "Close" in level0:
                    data = data["Close"]
                else:
                    # Fallback to first field level
                    data = data.xs(level0[0], level=0, axis=1)

            # At this point columns should be tickers
            all_nan = [c for c in data.columns if data[c].isna().all()]
            if all_nan:
                print(f"[safe_download] Dropping all-NaN tickers: {all_nan}")
                data = data.drop(columns=all_nan)

            if data.empty:
                raise RuntimeError("All tickers dropped; no usable price data.")

            return data

        except Exception as e:
            last_err = e
            print(f"[safe_download] Attempt {attempt} failed: {e}")
            time.sleep(sleep_base * attempt)

    raise RuntimeError(f"Failed to download price data for {normed}. Last error: {last_err}")


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class WaveMetrics:
    wave_name: str
    nav_series: pd.Series
    benchmark_nav: pd.Series
    total_return_60d: float
    bench_return_60d: float
    alpha_60d: float
    total_return_1y: float
    bench_return_1y: float
    alpha_1y: float


# ---------------------------------------------------------------------
# Benchmarks per Wave
# These are the benchmark blends we discussed (and you’ve screenshotted).
# We compute a weighted benchmark return series from these.
# ---------------------------------------------------------------------

BENCHMARK_MAP: Dict[str, Dict[str, float]] = {
    # S&P Wave – simple, classic
    "S&P Wave": {
        "SPY": 1.0,
    },

    # Growth Wave – 50% QQQ, 50% IWF (your chosen blend)
    "Growth Wave": {
        "QQQ": 0.50,
        "IWF": 0.50,
    },

    # Small-Mid Cap Growth Wave – 60% IGV, 20% WCLD/CLOU, 20% SPY (example tech/growth blend)
    "Small-Mid Cap Growth Wave": {
        "IGV": 0.60,
        "WCLD": 0.20,   # or CLOU – using WCLD here
        "SPY": 0.20,
    },

    # Clean Transit-Infrastructure Wave – FIDU/FDIS/SPY style blend (industrials + consumer + context)
    "Clean Transit-Infrastructure Wave": {
        "FIDU": 0.45,
        "FDIS": 0.45,
        "SPY": 0.10,
    },

    # Cloud & Enterprise Software Growth Wave – IGV + cloud + SPY
    "Cloud & Enterprise Software Growth Wave": {
        "IGV": 0.60,
        "WCLD": 0.20,   # WCLD or CLOU
        "SPY": 0.20,
    },

    # Crypto Equity Wave (mid/large cap) – 70% spot BTC proxy + 30% DAPP
    "Crypto Equity Wave (mid/large cap)": {
        "FBTC": 0.70,   # FBTC or IBIT
        "DAPP": 0.30,
    },

    # Income Wave – SCHD
    "Income Wave": {
        "SCHD": 1.0,
    },

    # Quantum Computing Wave – use a specialized tech / growth proxy (approx)
    "Quantum Computing Wave": {
        "QQQ": 0.50,
        "IGV": 0.30,
        "SPY": 0.20,
    },

    # AI Wave – software + cloud + broad tech context
    "AI Wave": {
        "IGV": 0.50,
        "VGT": 0.25,
        "QQQ": 0.15,
        "SPY": 0.10,
    },

    # SmartSafe Wave – short-term Treasuries / bills
    "SmartSafe Wave": {
        "SGOV": 0.70,
        "BIL": 0.20,
        "SHY": 0.10,
    },
}


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------

class WavesEngine:
    """
    Core engine:

    - Loads wave weights from CSV (wave_weights.csv)
    - Downloads & caches price data for all tickers
    - Builds wave NAV series and benchmark NAV series
    - Computes alpha statistics
    """

    def __init__(
        self,
        weights_csv_path: str = "wave_weights.csv",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> None:
        self.weights_csv_path = weights_csv_path
        self.start_date = start_date or (datetime.today() - timedelta(days=365 * 3))
        self.end_date = end_date or datetime.today()

        self.wave_weights: Dict[str, Dict[str, float]] = {}
        self.price_data: pd.DataFrame | None = None

        self._load_wave_weights()
        self._load_price_data()

    # -----------------------------------------------------------------
    # Loading weights & prices
    # -----------------------------------------------------------------

    def _load_wave_weights(self) -> None:
        """
        Read wave_weights.csv into a nested dict:
        { wave_name: { ticker: weight, ... }, ... }
        """
        df = pd.read_csv(self.weights_csv_path)

        expected_cols = {"wave", "ticker", "weight"}
        if not expected_cols.issubset(set(df.columns)):
            raise ValueError(
                f"wave_weights.csv must have columns {expected_cols}, got {df.columns.tolist()}"
            )

        weights: Dict[str, Dict[str, float]] = {}
        for _, row in df.iterrows():
            wave = str(row["wave"]).strip()
            ticker = normalize_ticker(str(row["ticker"]))
            weight = float(row["weight"])

            if wave not in weights:
                weights[wave] = {}

            # If duplicate ticker within a wave, sum weights (then we’ll renormalize)
            weights[wave][ticker] = weights[wave].get(ticker, 0.0) + weight

        # Renormalize each wave to sum to 1.0
        for wave, tickers in weights.items():
            total = sum(tickers.values())
            if total <= 0:
                raise ValueError(f"Wave {wave} has non-positive total weight.")
            for t in list(tickers.keys()):
                tickers[t] = tickers[t] / total

        self.wave_weights = weights

    def _load_price_data(self) -> pd.DataFrame:
        """
        Download historical prices for all unique tickers across all waves.
        Uses safe_download() wrapper.
        """
        ticker_set: Set[str] = set()
        for wave, tickers in self.wave_weights.items():
            for t in tickers.keys():
                ticker_set.add(normalize_ticker(t))

        if not ticker_set:
            raise RuntimeError("No tickers found in wave_weights; cannot download prices.")

        price_df = safe_download(
            tickers=ticker_set,
            start=self.start_date,
            end=self.end_date,
        )

        # Clean up missing values a bit
        price_df = price_df.sort_index().ffill().bfill()

        if price_df.empty:
            raise RuntimeError(
                f"No price data found for any ticker. Tickers requested: {sorted(list(ticker_set))}"
            )

        self.price_data = price_df
        return price_df

    # -----------------------------------------------------------------
    # Core computation helpers
    # -----------------------------------------------------------------

    def _wave_return_series(self, wave_name: str) -> pd.Series:
        """
        Build a daily return series for a given Wave from ticker prices + weights.
        """
        if self.price_data is None:
            raise RuntimeError("Price data not loaded.")

        if wave_name not in self.wave_weights:
            raise KeyError(f"Unknown Wave: {wave_name}")

        weights = self.wave_weights[wave_name]
        tickers = list(weights.keys())

        missing = [t for t in tickers if t not in self.price_data.columns]
        if missing:
            raise RuntimeError(
                f"Missing price data for tickers {missing} in Wave {wave_name}. "
                f"Check ticker aliases or wave_weights.csv."
            )

        prices = self.price_data[tickers]
        rets = prices.pct_change().fillna(0.0)

        weight_vec = np.array([weights[t] for t in tickers])
        wave_rets = (rets * weight_vec).sum(axis=1)
        return wave_rets

    def _benchmark_return_series(self, wave_name: str) -> pd.Series:
        """
        Build a daily benchmark return series from BENCHMARK_MAP for the given Wave.
        """
        if wave_name not in BENCHMARK_MAP:
            # Fallback: use SPY for anything not specified
            bench_def = {"SPY": 1.0}
        else:
            bench_def = BENCHMARK_MAP[wave_name]

        tickers = set(bench_def.keys())
        bench_prices = safe_download(
            tickers=tickers,
            start=self.start_date,
            end=self.end_date,
        )

        bench_prices = bench_prices.sort_index().ffill().bfill()
        if bench_prices.empty:
            raise RuntimeError(f"No benchmark price data for Wave {wave_name}")

        weight_vec = np.array([bench_def[t] for t in bench_prices.columns])
        bench_rets = bench_prices.pct_change().fillna(0.0)
        bench_series = (bench_rets * weight_vec).sum(axis=1)
        return bench_series

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    @staticmethod
    def _nav_from_returns(rets: pd.Series, start_nav: float = 100.0) -> pd.Series:
        nav = (1.0 + rets).cumprod() * start_nav
        return nav

    @staticmethod
    def _total_return(rets: pd.Series, window_days: int) -> float:
        if len(rets) == 0:
            return np.nan
        window_rets = rets.iloc[-window_days:]
        total = (1.0 + window_rets).prod() - 1.0
        return float(total)

    def compute_wave_metrics(self, wave_name: str) -> WaveMetrics:
        wave_rets = self._wave_return_series(wave_name)
        bench_rets = self._benchmark_return_series(wave_name)

        # Align indices
        idx = wave_rets.index.intersection(bench_rets.index)
        wave_rets = wave_rets.loc[idx]
        bench_rets = bench_rets.loc[idx]

        nav = self._nav_from_returns(wave_rets)
        bench_nav = self._nav_from_returns(bench_rets)

        # 60-day and 1-year windows (approx trading days)
        r_60 = self._total_return(wave_rets, 60)
        b_60 = self._total_return(bench_rets, 60)
        a_60 = r_60 - b_60

        r_1y = self._total_return(wave_rets, 252)
        b_1y = self._total_return(bench_rets, 252)
        a_1y = r_1y - b_1y

        return WaveMetrics(
            wave_name=wave_name,
            nav_series=nav,
            benchmark_nav=bench_nav,
            total_return_60d=r_60,
            bench_return_60d=b_60,
            alpha_60d=a_60,
            total_return_1y=r_1y,
            bench_return_1y=b_1y,
            alpha_1y=a_1y,
        )

    def compute_all_metrics(self) -> Dict[str, WaveMetrics]:
        metrics: Dict[str, WaveMetrics] = {}
        for wave in sorted(self.wave_weights.keys()):
            try:
                metrics[wave] = self.compute_wave_metrics(wave)
            except Exception as e:
                print(f"[compute_all_metrics] Failed for {wave}: {e}")
        return metrics

    def metrics_snapshot_df(self) -> pd.DataFrame:
        """
        Convenience: return a DataFrame with Wave-level summary stats
        for the "All Waves Snapshot" table in the console.
        """
        metrics = self.compute_all_metrics()
        rows = []
        for wave, m in metrics.items():
            rows.append(
                {
                    "Wave": wave,
                    "60D Return": m.total_return_60d,
                    "60D Bench Return": m.bench_return_60d,
                    "60D Alpha": m.alpha_60d,
                    "1Y Return": m.total_return_1y,
                    "1Y Bench Return": m.bench_return_1y,
                    "1Y Alpha": m.alpha_1y,
                }
            )
        df = pd.DataFrame(rows).set_index("Wave")
        # format as decimals; Streamlit will format to %
        return df.sort_index()


# ---------------------------------------------------------------------
# If you want a quick manual test (not used by Streamlit)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    eng = WavesEngine()
    snap = eng.metrics_snapshot_df()
    print(snap)