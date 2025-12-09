"""
waves_engine.py — WAVES Intelligence™ Vector 2.0 Engine (clean, 1-D safe)

Responsibilities
  • Load list.csv (total universe) and wave_weights.csv (wave definitions)
  • Build Wave portfolios from constituent tickers + weights
  • Fetch prices via yfinance for:
      - each Wave’s holdings
      - each Wave’s benchmark
      - VIX (^VIX) for risk-gating
  • Compute:
      - daily Wave returns
      - daily Benchmark returns
      - Wave / Benchmark value curves
      - daily Alpha Captured (Wave − Benchmark)
      - Intraday, 30D, 60D, 1Y Alpha Captured
      - 30D, 60D, 1Y Wave & Benchmark returns
      - realized beta (≈60d) via covariance / variance
      - mode-aware, VIX-gated exposure (Standard / Alpha-Minus-Beta / Private Logic)
  • Provide accessors used by the Streamlit app:
      - get_wave_names()
      - get_benchmark(wave)
      - get_wave_holdings(wave)
      - get_top_holdings(wave, n)
      - get_wave_performance(wave, mode, days, log=False)

All array operations are strictly 1-dimensional to avoid
“Data must be 1-dimensional, got shape (N, 1)” errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class WavesEngine:
    # ---------------------------------------------------------
    # INIT
    # ---------------------------------------------------------
    def __init__(
        self,
        list_path: str | Path = "list.csv",
        weights_path: str | Path = "wave_weights.csv",
        logs_root: str | Path = "logs",
    ):
        self.list_path = Path(list_path)
        self.weights_path = Path(weights_path)
        self.logs_root = Path(logs_root)

        self.universe = self._load_list()
        self.weights = self._load_weights()

        # simple benchmark map; fall back to SPY if not listed
        self._benchmark_map: Dict[str, str] = {
            "S&P Wave": "SPY",
            "S&P 500 Wave": "SPY",
            "S&P Wave ": "SPY",
            "Growth Wave": "QQQ",
            "Small Cap Growth Wave": "IWM",
            "Small to Mid Cap Growth Wave": "VO",
            "Income Wave": "SCHD",
            "Dividend Income Wave": "SCHD",
            "Future Power & Energy Wave": "ICLN",
            "Clean Transit-Infrastructure Wave": "ICLN",
            "Quantum Computing Wave": "IYW",
            "Total Market Wave": "VTI",
            "SmartSafe Wave": "SHV",
            "SmartSafe": "SHV",
            "SmartSafe™": "SHV",
            "Crypto Wave": "BTC-USD",
            "Crypto Income Wave": "BTC-USD",
        }

    # ---------------------------------------------------------
    # CSV LOADERS
    # ---------------------------------------------------------
    def _load_list(self) -> pd.DataFrame:
        if not self.list_path.exists():
            raise FileNotFoundError(f"Universe file not found: {self.list_path}")

        df = pd.read_csv(self.list_path)
        df.columns = [c.strip().lower() for c in df.columns]

        if "ticker" not in df.columns:
            raise ValueError("list.csv must include a 'Ticker' column (case-insensitive).")

        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

        # optional company / sector / description
        for col in ["company", "sector", "name"]:
            if col not in df.columns:
                df[col] = None

        return df

    def _load_weights(self) -> pd.DataFrame:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Wave weights file not found: {self.weights_path}")

        df = pd.read_csv(self.weights_path)
        df.columns = [c.strip().lower() for c in df.columns]

        required = {"wave", "ticker", "weight"}
        if not required.issubset(df.columns):
            raise ValueError("wave_weights.csv must include columns: wave,ticker,weight")

        df["wave"] = df["wave"].astype(str).str.strip()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

        # drop invalid rows
        df = df.dropna(subset=["ticker", "weight"])
        # normalize per Wave so weights sum to 1
        df["weight"] = df["weight"].astype(float)
        weight_sum = df.groupby("wave")["weight"].transform("sum")
        weight_sum = weight_sum.replace(0, np.nan)
        df["weight"] = df["weight"] / weight_sum

        return df

    # ---------------------------------------------------------
    # PUBLIC ACCESSORS
    # ---------------------------------------------------------
    def get_wave_names(self) -> List[str]:
        return sorted(self.weights["wave"].unique())

    def get_benchmark(self, wave: str) -> str:
        return self._benchmark_map.get(wave, "SPY")

    def get_wave_holdings(self, wave: str) -> pd.DataFrame:
        w = self.weights[self.weights["wave"] == wave].copy()
        if w.empty:
            return w

        uni = self.universe[["ticker", "company", "sector"]].copy()
        merged = w.merge(uni, on="ticker", how="left")
        # sort descending by weight for display
        merged = merged.sort_values("weight", ascending=False).reset_index(drop=True)
        return merged

    def get_top_holdings(self, wave: str, n: int = 10) -> pd.DataFrame:
        holdings = self.get_wave_holdings(wave)
        if holdings.empty:
            return holdings
        return holdings.sort_values("weight", ascending=False).head(n).reset_index(drop=True)

    # ---------------------------------------------------------
    # PRICE HELPERS (1-D SAFE)
    # ---------------------------------------------------------
    def _get_price_series(self, ticker: str, period: str = "1y") -> pd.Series:
        """
        Fetches adjusted close for a single ticker as a 1-D Series.
        """
        data = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            raise ValueError(f"No price data for {ticker}")
        close = data["Close"]
        # ensure 1-D Series
        if isinstance(close, pd.DataFrame):
            # sometimes yfinance returns a DataFrame with one column
            close = close.iloc[:, 0]
        close = close.astype(float)
        close.name = ticker
        return close

    def _get_price_matrix(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Fetches adjusted close for multiple tickers as a 2-D DataFrame [date x ticker].
        """
        if len(tickers) == 0:
            raise ValueError("No tickers provided to _get_price_matrix()")

        # yfinance supports list of tickers
        data = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            raise ValueError(f"No price data for tickers: {tickers}")

        # For multiple tickers, data['Close'] is DataFrame with one column per ticker
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            closes = data["Close"]
        else:
            # fallback — some yfinance versions behave differently
            closes = data.xs("Close", axis=1, level=0)

        # Keep only our requested tickers, drop all-NaN columns
        closes = closes[[c for c in closes.columns if c in tickers]]
        closes = closes.dropna(how="all")
        closes = closes.astype(float)
        return closes

    # ---------------------------------------------------------
    # CORE PERFORMANCE
    # ---------------------------------------------------------
    def get_wave_performance(
        self,
        wave: str,
        mode: str = "standard",
        days: int = 30,
        log: bool = False,
    ) -> Optional[dict]:
        """
        Compute full metrics for a given Wave + mode.
        All returns are daily; alpha is Wave − Benchmark.

        Returns dict with keys used by the Streamlit app:
          - benchmark
          - beta_realized
          - exposure_final
          - intraday_alpha_captured
          - alpha_30d, alpha_60d, alpha_1y
          - return_30d_wave, return_30d_benchmark
          - return_60d_wave, return_60d_benchmark
          - return_1y_wave, return_1y_benchmark
          - history_30d (DataFrame)
        """
        holdings = self.get_wave_holdings(wave)
        if holdings.empty:
            return None

        # unique tickers for the Wave
        tickers = sorted(holdings["ticker"].unique())
        weights = holdings.set_index("ticker")["weight"]

        # choose enough history to cover 1Y
        history_period = "1y"

        # --- Wave prices & returns (portfolio) ---
        try:
            price_matrix = self._get_price_matrix(tickers, period=history_period)
        except Exception:
            return None

        # align weights to available price columns
        common = [t for t in price_matrix.columns if t in weights.index]
        if not common:
            return None

        price_matrix = price_matrix[common]
        w_vec = weights.loc[common].values.astype(float)

        # daily returns per ticker
        ret_matrix = price_matrix.pct_change().dropna(how="all")

        # ensure 2-D (date x ticker)
        ret_matrix = ret_matrix.astype(float)

        # portfolio daily return: (ret * weights).sum(axis=1)
        wave_ret_series = ret_matrix.mul(w_vec, axis=1).sum(axis=1)

        # ensure 1-D Series
        wave_ret_series = pd.Series(wave_ret_series, index=ret_matrix.index, name="wave_return")

        wave_value = (1.0 + wave_ret_series).cumprod()
        wave_value.name = "wave_value"

        # --- Benchmark prices & returns ---
        benchmark = self.get_benchmark(wave)
        try:
            bm_price = self._get_price_series(benchmark, period=history_period)
        except Exception:
            return None

        bm_ret = bm_price.pct_change()
        bm_ret.name = "benchmark_return"
        bm_value = (1.0 + bm_ret).cumprod()
        bm_value.name = "benchmark_value"

        # --- Merge Wave + Benchmark ---
        df = pd.concat([wave_ret_series, bm_ret, wave_value, bm_value], axis=1).dropna()
        if df.empty:
            return None

        # daily alpha captured
        df["alpha_captured"] = df["wave_return"] - df["benchmark_return"]

        # --- alpha windows ---
        def alpha_window(series: pd.Series, window: int) -> Optional[float]:
            if series is None or series.empty:
                return None
            n = min(window, len(series))
            if n <= 0:
                return None
            return float(series.tail(n).sum())

        intraday_alpha = float(df["alpha_captured"].iloc[-1])
        alpha_30d = alpha_window(df["alpha_captured"], 30)
        alpha_60d = alpha_window(df["alpha_captured"], 60)
        alpha_1y = alpha_window(df["alpha_captured"], len(df))  # full period ≈1y

        # --- return windows ---
        def window_return(curve: pd.Series, window: int) -> Optional[float]:
            if curve is None or curve.empty:
                return None
            n = min(window, len(curve))
            if n <= 1:
                return None
            start = float(curve.iloc[-n])
            end = float(curve.iloc[-1])
            if start == 0:
                return None
            return (end / start) - 1.0

        ret_30_wave = window_return(df["wave_value"], 30)
        ret_30_bm = window_return(df["benchmark_value"], 30)
        ret_60_wave = window_return(df["wave_value"], 60)
        ret_60_bm = window_return(df["benchmark_value"], 60)
        ret_1y_wave = window_return(df["wave_value"], min(len(df), 252))
        ret_1y_bm = window_return(df["benchmark_value"], min(len(df), 252))

        # --- realized beta (≈60d) ---
        beta_realized = np.nan
        tail_n = min(60, len(df))
        if tail_n >= 20:
            x = df["benchmark_return"].tail(tail_n).values.flatten()
            y = df["wave_return"].tail(tail_n).values.flatten()
            if np.var(x) > 0:
                cov_xy = np.cov(x, y)[0, 1]
                beta_realized = float(cov_xy / np.var(x))

        # --- VIX-gated exposure by mode ---
        exposure_final = self._compute_exposure(mode, beta_realized)

        # --- history_30d slice for the app ---
        history_30d = df.tail(30).copy()

        result = {
            "benchmark": benchmark,
            "beta_realized": beta_realized,
            "exposure_final": exposure_final,
            "intraday_alpha_captured": intraday_alpha,
            "alpha_30d": alpha_30d,
            "alpha_60d": alpha_60d,
            "alpha_1y": alpha_1y,
            "return_30d_wave": ret_30_wave,
            "return_30d_benchmark": ret_30_bm,
            "return_60d_wave": ret_60_wave,
            "return_60d_benchmark": ret_60_bm,
            "return_1y_wave": ret_1y_wave,
            "return_1y_benchmark": ret_1y_bm,
            "history_30d": history_30d,
        }

        # optional logging hook (currently off in app)
        if log:
            self._log_performance_row(wave, result)

        return result

    # ---------------------------------------------------------
    # EXPOSURE MODEL (Mode + VIX)
    # ---------------------------------------------------------
    def _compute_exposure(self, mode: str, beta_realized: float) -> float:
        """
        Mode-aware + VIX-gated exposure.

        Modes:
          • standard        – baseline 1.0x, mild VIX throttling
          • alpha-minus-beta– target 0.7-0.9 beta, stronger VIX throttling
          • private_logic   – 1.1x in calm markets, but still cut in high VIX
        """
        # base by mode
        mode = (mode or "standard").lower()
        beta = beta_realized if beta_realized is not None and not np.isnan(beta_realized) else 1.0

        if mode == "alpha-minus-beta":
            base = max(0.3, min(1.0, 1.0 - 0.4 * (beta - 0.8)))  # steer toward ~0.8 beta
        elif mode == "private_logic":
            base = 1.1
        else:  # standard
            base = 1.0

        # VIX regime
        vix_level = self._get_vix_level()  # float or None

        # regime multipliers
        if vix_level is None:
            vix_mult = 1.0
        elif vix_level < 14:
            vix_mult = 1.1 if mode == "private_logic" else 1.0
        elif vix_level < 22:
            vix_mult = 0.9
        else:
            vix_mult = 0.7 if mode != "alpha-minus-beta" else 0.6

        exposure = base * vix_mult
        # clamp to a reasonable range
        exposure = float(np.clip(exposure, 0.2, 1.4))
        return exposure

    def _get_vix_level(self) -> Optional[float]:
        """
        Fetch recent VIX level (^VIX). Returns last close as float or None on failure.
        """
        try:
            vix = self._get_price_series("^VIX", period="3mo")
        except Exception:
            return None
        if vix.empty:
            return None
        return float(vix.iloc[-1])

    # ---------------------------------------------------------
    # LOGGING (optional)
    # ---------------------------------------------------------
    def _log_performance_row(self, wave: str, result: dict) -> None:
        """
        Append a single performance row to logs/performance/<Wave>_performance_daily.csv.
        The app currently calls get_wave_performance(..., log=False), so this is a
        future-proof hook.
        """
        try:
            perf_dir = self.logs_root / "performance"
            perf_dir.mkdir(parents=True, exist_ok=True)

            fname = perf_dir / f"{wave.replace(' ', '_')}_performance_daily.csv"

            row = {
                "benchmark": result.get("benchmark"),
                "beta_realized": result.get("beta_realized"),
                "exposure_final": result.get("exposure_final"),
                "intraday_alpha_captured": result.get("intraday_alpha_captured"),
                "alpha_30d": result.get("alpha_30d"),
                "alpha_60d": result.get("alpha_60d"),
                "alpha_1y": result.get("alpha_1y"),
                "return_30d_wave": result.get("return_30d_wave"),
                "return_30d_benchmark": result.get("return_30d_benchmark"),
                "return_60d_wave": result.get("return_60d_wave"),
                "return_60d_benchmark": result.get("return_60d_benchmark"),
                "return_1y_wave": result.get("return_1y_wave"),
                "return_1y_benchmark": result.get("return_1y_benchmark"),
            }

            df_row = pd.DataFrame([row])
            if fname.exists():
                df_existing = pd.read_csv(fname)
                df_out = pd.concat([df_existing, df_row], ignore_index=True)
            else:
                df_out = df_row
            df_out.to_csv(fname, index=False)
        except Exception:
            # logging must never break the engine
            pass