"""
waves_engine.py

WAVES Intelligence™ Engine
- Uses list.csv as the TOTAL MARKET universe (no wave column required)
- Uses wave_weights.csv as the authoritative Wave definition file
- Auto-detects Waves from wave_weights.csv
- Normalizes weights per Wave
- Merges universe metadata (company, sector, etc.)
- Computes intraday + 30-day returns and alpha vs benchmark
- Writes simple logs for positions and performance
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class WavesEngine:
    def __init__(
        self,
        list_path: str | Path = "list.csv",
        weights_path: str | Path = "wave_weights.csv",
        logs_root: str | Path = "logs",
        default_lookback_days: int = 30,
    ) -> None:
        self.list_path = Path(list_path)
        self.weights_path = Path(weights_path)
        self.logs_root = Path(logs_root)
        self.default_lookback_days = default_lookback_days

        self._prepare_log_dirs()

        # Core datasets
        self.universe = self._load_universe()
        self.wave_weights = self._load_wave_weights()
        self.wave_definitions = self._build_wave_definitions()

        # Benchmarks per wave (EDIT these tickers if you want different benchmarks)
        self.benchmark_map = self._default_benchmark_map()

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def _prepare_log_dirs(self) -> None:
        (self.logs_root / "positions").mkdir(parents=True, exist_ok=True)
        (self.logs_root / "performance").mkdir(parents=True, exist_ok=True)

    def _load_universe(self) -> pd.DataFrame:
        """
        Load list.csv as the TOTAL MARKET universe.

        Requirements:
        - Must contain a ticker column (Ticker/ticker/symbol)
        - Other columns are optional and treated as metadata:
          Company, Sector, Weight, etc.
        """
        if not self.list_path.exists():
            raise FileNotFoundError(f"Universe file not found: {self.list_path}")

        df = pd.read_csv(self.list_path)

        # Identify ticker column
        ticker_cols = [c for c in df.columns if c.lower() in ("ticker", "symbol")]
        if not ticker_cols:
            raise ValueError(
                "list.csv must contain a 'Ticker' or 'ticker' (or 'symbol') column. "
                f"Found columns: {list(df.columns)}"
            )
        ticker_col = ticker_cols[0]
        df = df.rename(columns={ticker_col: "ticker"})

        # Standardize a few common metadata columns
        rename_map = {}
        for col in df.columns:
            cl = col.lower()
            if cl == "company":
                rename_map[col] = "company"
            elif cl == "weight":
                rename_map[col] = "universe_weight"
            elif cl == "sector":
                rename_map[col] = "sector"
        if rename_map:
            df = df.rename(columns=rename_map)

        # Clean tickers
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

        return df

    def _load_wave_weights(self) -> pd.DataFrame:
        """
        Load wave_weights.csv which defines:
        - wave
        - ticker
        - weight
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Wave weights file not found: {self.weights_path}")

        df = pd.read_csv(self.weights_path)

        # Normalize column names
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if cl == "wave":
                col_map[col] = "wave"
            elif cl == "ticker":
                col_map[col] = "ticker"
            elif cl == "weight":
                col_map[col] = "weight"

        df = df.rename(columns=col_map)

        required = ["wave", "ticker", "weight"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"wave_weights.csv must have columns {required}, missing {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        df["wave"] = df["wave"].astype(str).str.strip()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

        df = df.dropna(subset=["weight"])
        df = df[df["weight"] > 0]

        return df

    def _build_wave_definitions(self) -> pd.DataFrame:
        """
        Merge wave_weights with universe metadata and normalize weights
        within each Wave.
        """
        df = self.wave_weights.copy()

        # Normalize weights inside each wave to sum to 1.0
        df["weight"] = df.groupby("wave")["weight"].transform(
            lambda x: x / x.sum() if x.sum() != 0 else x
        )

        # Merge with universe metadata (left join: keep weights even if no metadata)
        merged = df.merge(self.universe, how="left", on="ticker", suffixes=("", "_u"))

        return merged

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_wave_names(self) -> List[str]:
        return sorted(self.wave_definitions["wave"].unique())

    def get_wave_holdings(self, wave: str) -> pd.DataFrame:
        df = self.wave_definitions[self.wave_definitions["wave"] == wave].copy()
        if df.empty:
            raise ValueError(f"No holdings found for wave: {wave}")
        return df.sort_values("weight", ascending=False)

    def get_top_holdings(self, wave: str, n: int = 10) -> pd.DataFrame:
        holdings = self.get_wave_holdings(wave)
        return holdings.head(n)

    # ------------------------------------------------------------------
    # Benchmarks & performance
    # ------------------------------------------------------------------
    def _default_benchmark_map(self) -> Dict[str, str]:
        """
        Default benchmark tickers per Wave.

        You can EDIT this mapping as needed.
        """
        return {
            "S&P Wave": "SPY",  # S&P 500
            "Growth Wave": "QQQ",  # Nasdaq 100 / growth proxy
            "Future Power & Energy Wave": "XLE",
            "Clean Transit-Infrastructure Wave": "ICLN",
            # Fallback for all others: SPY
        }

    def get_benchmark(self, wave: str) -> str:
        return self.benchmark_map.get(wave, "SPY")

    def _fetch_history(
        self,
        tickers: List[str],
        days: int,
    ) -> pd.DataFrame:
        """
        Fetch daily adjusted close prices for a list of tickers over the last N days.
        """
        if not tickers:
            raise ValueError("No tickers provided to _fetch_history")

        end = datetime.now()
        start = end - timedelta(days=days + 3)  # small buffer for weekends/holidays

        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        # data["Close"] yields:
        # - DataFrame with columns as tickers (for multiple tickers)
        # - Series (for a single ticker)
        close = data["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])

        close = close.dropna(how="all")
        return close

    def get_wave_performance(
        self,
        wave: str,
        days: Optional[int] = None,
        log: bool = True,
    ) -> Dict[str, object]:
        """
        Compute intraday and 30-day performance + alpha vs benchmark.

        Returns dict:
        {
            "wave": wave,
            "benchmark": benchmark,
            "intraday_return": float,
            "intraday_alpha": float,
            "return_30d": float,
            "alpha_30d": float,
            "history": DataFrame[w_* and benchmark_* columns]
        }
        """
        if days is None:
            days = self.default_lookback_days

        holdings = self.get_wave_holdings(wave)
        benchmark = self.get_benchmark(wave)

        tickers = sorted(holdings["ticker"].unique().tolist())
        all_tickers = sorted(set(tickers + [benchmark]))

        prices = self._fetch_history(all_tickers, days=days)
        if prices.empty:
            raise RuntimeError("No price history returned from yfinance.")

        returns = prices.pct_change().dropna(how="all")

        # Align weights & returns
        weight_series = holdings.set_index("ticker")["weight"]
        common_tickers = [t for t in tickers if t in returns.columns]

        if not common_tickers:
            raise RuntimeError(f"No overlapping tickers between weights and price data for wave {wave}")

        w = weight_series.loc[common_tickers]
        w = w / w.sum()  # re-normalize on common universe

        wave_daily = (returns[common_tickers] * w).sum(axis=1)

        if benchmark in returns.columns:
            bm_daily = returns[benchmark]
        else:
            # Fallback: zero benchmark if we cannot find it
            bm_daily = pd.Series(
                index=wave_daily.index,
                data=0.0,
                name="benchmark",
            )

        # Cumulative curves
        wave_curve = (1.0 + wave_daily).cumprod()
        bm_curve = (1.0 + bm_daily).cumprod()

        intraday_return = float(wave_daily.iloc[-1])
        bm_intraday = float(bm_daily.iloc[-1])
        intraday_alpha = intraday_return - bm_intraday

        total_wave_ret = float(wave_curve.iloc[-1] - 1.0)
        total_bm_ret = float(bm_curve.iloc[-1] - 1.0)
        alpha_30d = total_wave_ret - total_bm_ret

        history = pd.DataFrame(
            {
                "wave_return": wave_daily,
                "benchmark_return": bm_daily,
                "wave_value": wave_curve,
                "benchmark_value": bm_curve,
            }
        )

        result = {
            "wave": wave,
            "benchmark": benchmark,
            "intraday_return": intraday_return,
            "intraday_alpha": intraday_alpha,
            "return_30d": total_wave_ret,
            "alpha_30d": alpha_30d,
            "history": history,
        }

        if log:
            self._log_positions(wave, holdings)
            self._log_performance(wave, result)

        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_positions(self, wave: str, holdings: pd.DataFrame) -> None:
        today = datetime.now().strftime("%Y%m%d")
        path = self.logs_root / "positions" / f"{wave.replace(' ', '_')}_positions_{today}.csv"

        df = holdings.copy()
        df.insert(0, "date", datetime.now().strftime("%Y-%m-%d"))
        df.to_csv(path, index=False)

    def _log_performance(self, wave: str, perf: Dict[str, object]) -> None:
        path = self.logs_root / "performance" / f"{wave.replace(' ', '_')}_performance_daily.csv"
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "wave": perf["wave"],
            "benchmark": perf["benchmark"],
            "intraday_return": perf["intraday_return"],
            "intraday_alpha": perf["intraday_alpha"],
            "return_30d": perf["return_30d"],
            "alpha_30d": perf["alpha_30d"],
        }

        if path.exists():
            df = pd.read_csv(path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_csv(path, index=False)