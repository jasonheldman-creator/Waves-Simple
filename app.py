"""
waves_engine.py — WAVES Intelligence™ Vector 2.0 Engine (auto-cleaning, 1-D safe)

Key behaviors
  • Loads list.csv (universe) and wave_weights.csv (Wave definitions).
  • Builds each Wave from its tickers + weights.
  • Fetches data via yfinance for:
      - each Wave’s constituents
      - each Wave’s benchmark
      - VIX (^VIX) for risk-gating
  • Computes:
      - daily Wave & benchmark returns
      - Wave / benchmark value curves
      - daily alpha captured (Wave − BM)
      - Intraday, 30D, 60D, 1Y alpha captured
      - 30D, 60D, 1Y Wave & benchmark returns
      - realized beta (≈60d)
      - mode-aware VIX-gated exposure
  • AUTO-CLEANS:
      - Drops tickers that can’t be priced
      - Drops tickers whose returns are all NaN
      - Re-normalizes weights to the surviving tickers
      - So you never see “weight alignment mismatch” again.

Exported API used by app.py:
  - get_wave_names()
  - get_benchmark(wave)
  - get_wave_holdings(wave)
  - get_top_holdings(wave, n)
  - get_wave_performance(wave, mode, days, log=False)
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

        # basic benchmark map; fall back to SPY
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

        # optional metadata
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

        df = df.dropna(subset=["ticker", "weight"])
        df["weight"] = df["weight"].astype(float)

        # normalize per Wave so weights sum to 1
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
        Fetch adjusted close for a single ticker as a 1-D Series.
        Raises if no data.
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
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.astype(float)
        close.name = ticker
        return close

    def _get_price_matrix(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Fetch adjusted close for multiple tickers as DataFrame [date x ticker].

        AUTO-CLEANS:
          • Skips tickers with no data
          • Returns only the tickers that produced a price Series
        """
        frames = []
        valid_tickers: List[str] = []
        for t in tickers:
            try:
                s = self._get_price_series(t, period=period)
                frames.append(s)
                valid_tickers.append(t)
            except Exception:
                # skip tickers we can't price at all
                continue

        if not frames:
            raise ValueError(f"No price data for any tickers in: {tickers}")

        closes = pd.concat(frames, axis=1)
        closes.columns = valid_tickers
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
        AUTO-CLEANS ticker set so weights and returns always align.
        """
        holdings = self.get_wave_holdings(wave)
        if holdings.empty:
            raise ValueError(f"No holdings defined for Wave: {wave}")

        # original tickers + weights
        tickers = sorted(holdings["ticker"].unique())
        weights = holdings.set_index("ticker")["weight"]

        history_period = "1y"

        # --- Wave prices & returns (portfolio) ---
        price_matrix = self._get_price_matrix(tickers, period=history_period)

        # build daily returns and drop first NaN row only (not columns)
        ret_matrix = price_matrix.pct_change().iloc[1:, :].astype(float)

        # drop columns (tickers) where returns are all NaN
        ret_matrix = ret_matrix.dropna(axis=1, how="all")

        if ret_matrix.empty:
            raise ValueError(f"No usable return history for Wave {wave}")

        # active tickers are the ones that survived
        active_tickers = list(ret_matrix.columns)

        # align weights to active tickers and renormalize
        w_vec = weights.reindex(active_tickers).fillna(0.0).values.astype(float)
        w_sum = w_vec.sum()
        if w_sum <= 0:
            raise ValueError(f"No positive weights for active tickers in Wave {wave}")
        w_vec = w_vec / w_sum

        # portfolio daily return: (ret * weights).sum(axis=1)
        wave_ret_series = ret_matrix.mul(w_vec, axis=1).sum(axis=1)
        wave_ret_series.name = "wave_return"

        wave_value = (1.0 + wave_ret_series).cumprod()
        wave_value.name = "wave_value"

        # --- Benchmark prices & returns ---
        benchmark = self.get_benchmark(wave)
        bm_price = self._get_price_series(benchmark, period=history_period)

        bm_ret = bm_price.pct_change().iloc[1:].astype(float)
        bm_ret.name = "benchmark_return"
        bm_value = (1.0 + bm_ret).cumprod()
        bm_value.name = "benchmark_value"

        # --- Merge Wave + Benchmark ---
        df = pd.concat([wave_ret_series, bm_ret, wave_value, bm_value], axis=1).dropna()
        if df.empty:
            raise ValueError(f"No overlapping Wave/benchmark history for {wave}")

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

        if log:
            self._log_performance_row(wave, result)

        return result

    # ---------------------------------------------------------
    # EXPOSURE MODEL (Mode + VIX)
    # ---------------------------------------------------------
    def _compute_exposure(self, mode: str, beta_realized: float) -> float:
        """
        Mode-aware + VIX-gated exposure.
        """
        mode = (mode or "standard").lower()
        beta = beta_realized if beta_realized is not None and not np.isnan(beta_realized) else 1.0

        if mode == "alpha-minus-beta":
            base = max(0.3, min(1.0, 1.0 - 0.4 * (beta - 0.8)))  # steer toward ~0.8 beta
        elif mode == "private_logic":
            base = 1.1
        else:  # standard
            base = 1.0

        vix_level = self._get_vix_level()

        if vix_level is None:
            vix_mult = 1.0
        elif vix_level < 14:
            vix_mult = 1.1 if mode == "private_logic" else 1.0
        elif vix_level < 22:
            vix_mult = 0.9
        else:
            vix_mult = 0.7 if mode != "alpha-minus-beta" else 0.6

        exposure = base * vix_mult
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
    # LOGGING (optional, safe)
    # ---------------------------------------------------------
    def _log_performance_row(self, wave: str, result: dict) -> None:
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