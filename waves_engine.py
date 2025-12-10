"""
waves_engine.py — WAVES Intelligence™ Vector 2.0 Engine + Strategy Overlay
Benchmark Map v1.1 (LOCKED)

Changes in this version:
• Rename "Crypto Income Wave" → "Crypto Equity Wave"
• Remove "Future Power & Energy Wave"
• Add "AI Wave" with an AI-focused blended benchmark

What this does
--------------
1) Loads list.csv (universe) and wave_weights.csv (Wave definitions).
2) Aggregates duplicate tickers per Wave (sums weights).
3) Fetches 1-year daily prices via yfinance.
4) Computes:
   • daily Wave & benchmark returns
   • cumulative value curves
   • intraday, 30D, 60D, 1Y alpha captured
   • 30D / 60D / 1Y Wave & benchmark returns
   • realised beta (≈60 trading days)
5) Applies a DAILY TRADING OVERLAY:
   • per-day exposure is determined by:
       - Mode: standard / alpha-minus-beta / private_logic
       - VIX level on that day
   • daily Wave returns are scaled by exposure_t
   → approximates live risk-controls/algos instead of naive buy & hold.
6) Supports blended benchmarks:
   • Growth Wave              → 50% QQQ + 50% IWF
   • Small-Mid Cap Growth     → 50% VTWG + 50% VO
   • AI Wave                  → 40% QQQ + 60% SOXX
   • Clean Transit-Infrastructure → 50% ICLN + 50% IGF
   • Quantum Computing Wave   → 70% IYW + 30% SOXX
   • Crypto Equity Wave       → 50% BTC-USD + 30% ETH-USD + 20% SOL-USD
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf


class WavesEngine:
    # ---------------------------------------------------------
    # INIT
    # ---------------------------------------------------------
    def __init__(
        self,
        list_path: Union[str, Path] = "list.csv",
        weights_path: Union[str, Path] = "wave_weights.csv",
        logs_root: Union[str, Path] = "logs",
    ):
        self.list_path = Path(list_path)
        self.weights_path = Path(weights_path)
        self.logs_root = Path(logs_root)

        self.universe = self._load_list()
        self.weights = self._load_weights()

        # Single-ticker benchmark defaults (used when not overridden by blend logic)
        self._benchmark_map: Dict[str, str] = {
            "S&P Wave": "SPY",
            "S&P 500 Wave": "SPY",
            "S&P Wave ": "SPY",
            "Income Wave": "SCHD",
            "Dividend Income Wave": "SCHD",
            "Small Cap Growth Wave": "VTWG",
            "Small-Mid Cap Growth Wave": "VO",  # overridden by blend logic
            "Small to Mid Cap Growth Wave": "VO",
            "Clean Transit-Infrastructure Wave": "ICLN",  # overridden by blend logic
            "Quantum Computing Wave": "IYW",              # overridden by blend logic
            "AI Wave": "QQQ",                             # overridden by blend logic
            "Total Market Wave": "VTI",
            "SmartSafe Wave": "SHV",
            "SmartSafe": "SHV",
            "SmartSafe™": "SHV",
            "Crypto Equity Wave": "BTC-USD",              # overridden by blend logic
            "Crypto Wave": "BTC-USD",
            "Growth Wave": "QQQ",                         # overridden by blend logic
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

        # Optional metadata columns
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

        # Normalise weights per Wave so they sum to 1.0
        weight_sum = df.groupby("wave")["weight"].transform("sum")
        weight_sum = weight_sum.replace(0, np.nan)
        df["weight"] = df["weight"] / weight_sum

        return df

    # ---------------------------------------------------------
    # PUBLIC ACCESSORS
    # ---------------------------------------------------------
    def get_wave_names(self) -> List[str]:
        return sorted(self.weights["wave"].unique())

    def get_benchmark(self, wave: str) -> Union[str, Dict[str, float]]:
        """
        Returns one of:
            • Single ticker string, e.g. "SPY"
            • Dict of {ticker: weight} for blended benchmarks
        """
        wave = wave.strip()

        # --------- Custom blended benchmarks (Benchmark Map v1.1) ----------
        if wave == "Growth Wave":
            # 50% QQQ + 50% IWF
            return {"QQQ": 0.50, "IWF": 0.50}

        if wave == "Small-Mid Cap Growth Wave" or wave == "Small to Mid Cap Growth Wave":
            # 50% VTWG + 50% VO
            return {"VTWG": 0.50, "VO": 0.50}

        if wave == "AI Wave":
            # AI megacap + semis: 40% QQQ + 60% SOXX
            return {"QQQ": 0.40, "SOXX": 0.60}

        if wave == "Clean Transit-Infrastructure Wave":
            # 50% ICLN + 50% IGF
            return {"ICLN": 0.50, "IGF": 0.50}

        if wave == "Quantum Computing Wave":
            # 70% IYW + 30% SOXX
            return {"IYW": 0.70, "SOXX": 0.30}

        if wave == "Crypto Equity Wave":
            # Broad crypto index approximation
            return {"BTC-USD": 0.50, "ETH-USD": 0.30, "SOL-USD": 0.20}

        # --------- Standard single-ticker benchmarks ---------
        return self._benchmark_map.get(wave, "SPY")

    def get_wave_holdings(self, wave: str) -> pd.DataFrame:
        """
        Holdings for a Wave, with duplicate tickers aggregated (weights summed).
        """
        w = self.weights[self.weights["wave"] == wave].copy()
        if w.empty:
            return w

        # Aggregate duplicates within Wave
        w = (
            w.groupby(["wave", "ticker"], as_index=False)["weight"]
            .sum()
        )

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

        Auto-clean:
          • skips tickers with no data
        """
        frames = []
        valid_tickers: List[str] = []
        for t in tickers:
            try:
                s = self._get_price_series(t, period=period)
                frames.append(s)
                valid_tickers.append(t)
            except Exception:
                # Skip un-priceable names (delisted, etc.)
                continue

        if not frames:
            raise ValueError(f"No price data for any tickers in: {tickers}")

        closes = pd.concat(frames, axis=1)
        closes.columns = valid_tickers
        closes = closes.dropna(how="all")
        closes = closes.astype(float)
        return closes

    # ---------------------------------------------------------
    # STRATEGY OVERLAY — VIX-GATED DAILY EXPOSURE
    # ---------------------------------------------------------
    def _get_vix_series(self, index_like: pd.Index, period: str = "1y") -> pd.Series:
        """
        Fetch VIX (^VIX) and align it to the given date index (forward/back fill).
        """
        vix = self._get_price_series("^VIX", period=period)
        vix.index = pd.to_datetime(vix.index)
        idx = pd.to_datetime(index_like)
        vix_aligned = vix.reindex(idx, method=None)
        # fill gaps forward/back to avoid NaNs
        vix_aligned = vix_aligned.ffill().bfill()
        vix_aligned.name = "VIX"
        return vix_aligned

    def _compute_exposure_series(self, mode: str, vix_series: pd.Series) -> pd.Series:
        """
        Compute a per-day exposure series based on mode + daily VIX level.
        Core "trading algo" overlay: how much of the raw portfolio return
        to actually take each day.

        Rough rules:
          - standard:
              base = 1.0
          - alpha-minus-beta:
              base = 0.8  (more defensive)
          - private_logic:
              base = 1.1  (slightly more aggressive when VIX calm)

          VIX ladder:
              VIX < 14   → calm, can run full base (or +10% in PL)
              14–22      → moderate, 0.9x base
              > 22       → high stress, 0.6x base (0.5x for alpha-minus-beta)

        Exposure is clipped into [0.2, 1.4].
        """
        mode = (mode or "standard").lower()
        if mode == "alpha-minus-beta":
            base = 0.8
        elif mode == "private_logic":
            base = 1.1
        else:
            base = 1.0

        vix_vals = vix_series.astype(float).values
        exposure_vals = []

        for v in vix_vals:
            if np.isnan(v):
                v_mult = 1.0
            elif v < 14:
                # calm
                v_mult = 1.1 if mode == "private_logic" else 1.0
            elif v < 22:
                # moderate
                v_mult = 0.9
            else:
                # stressed
                v_mult = 0.5 if mode == "alpha-minus-beta" else 0.6

            exp_val = base * v_mult
            exp_val = float(np.clip(exp_val, 0.2, 1.4))
            exposure_vals.append(exp_val)

        exposure_series = pd.Series(exposure_vals, index=vix_series.index, name="exposure")
        return exposure_series

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

        Auto-align:
          • aggregates duplicate tickers
          • uses only tickers with price data
          • drops tickers whose returns are all NaN
          • re-normalises weights to active tickers

        Strategy overlay:
          • builds raw portfolio daily return
          • fetches daily VIX and computes exposure_t per day
          • final Wave return_t = raw_return_t * exposure_t
        """
        holdings = self.get_wave_holdings(wave)
        if holdings.empty:
            raise ValueError(f"No holdings defined for Wave: {wave}")

        # Unique tickers with aggregate weight
        weights_all = (
            holdings.groupby("ticker")["weight"]
            .sum()
            .astype(float)
        )
        all_tickers = list(weights_all.index)

        history_period = "1y"

        # --- Wave prices & returns (portfolio) ---
        price_matrix = self._get_price_matrix(all_tickers, period=history_period)

        # Daily returns (drop first NaN row), then drop all-NaN columns
        ret_matrix = price_matrix.pct_change().iloc[1:, :].astype(float)
        ret_matrix = ret_matrix.dropna(axis=1, how="all")

        if ret_matrix.empty:
            raise ValueError(f"No usable return history for Wave {wave}")

        # Active tickers are columns with usable returns
        active_tickers = list(ret_matrix.columns)

        # Align weights to active tickers and renormalise
        weights_active = weights_all.reindex(active_tickers).fillna(0.0)
        w_sum = weights_active.sum()
        if w_sum <= 0:
            raise ValueError(f"No positive weights for active tickers in Wave {wave}")
        w_vec = (weights_active / w_sum).values.astype(float)

        # Raw portfolio daily return = weighted sum across columns
        raw_wave_ret = ret_matrix.mul(w_vec, axis=1).sum(axis=1)
        raw_wave_ret.name = "wave_return_raw"

        # --- Strategy overlay: daily VIX-gated exposure ---
        vix_series = self._get_vix_series(raw_wave_ret.index, period=history_period)
        exposure_series = self._compute_exposure_series(mode, vix_series)

        # final Wave daily return after overlay
        wave_ret_series = raw_wave_ret * exposure_series
        wave_ret_series.name = "wave_return"

        wave_value = (1.0 + wave_ret_series).cumprod()
        wave_value.name = "wave_value"

        # --- Benchmark prices & returns ---
        benchmark = self.get_benchmark(wave)

        # Blended benchmark
        if isinstance(benchmark, dict):
            bm_prices = self._get_price_matrix(list(benchmark.keys()), period=history_period)
            bm_rets = bm_prices.pct_change().iloc[1:, :].astype(float)

            # align blend weights to tickers actually present
            w_bm = pd.Series(benchmark)
            w_bm = w_bm.reindex(bm_rets.columns).fillna(0.0)
            if w_bm.sum() <= 0:
                # fallback: equal weight
                w_bm = pd.Series(1.0, index=bm_rets.columns)
            w_bm = w_bm / w_bm.sum()

            bm_ret = bm_rets.mul(w_bm.values, axis=1).sum(axis=1)
            bm_ret.name = "benchmark_return"
            bm_value = (1.0 + bm_ret).cumprod()
            bm_value.name = "benchmark_value"
        else:
            # Single-ticker benchmark
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

        # --- Alpha windows ---
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
        alpha_1y = alpha_window(df["alpha_captured"], len(df))  # full ≈1y period

        # --- Return windows ---
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

        # --- Realised beta (≈60d) ---
        beta_realized = np.nan
        tail_n = min(60, len(df))
        if tail_n >= 20:
            x = df["benchmark_return"].tail(tail_n).values.flatten()
            y = df["wave_return"].tail(tail_n).values.flatten()
            if np.var(x) > 0:
                cov_xy = np.cov(x, y)[0, 1]
                beta_realized = float(cov_xy / np.var(x))

        # --- Exposure final (last day) ---
        exposure_final = float(exposure_series.iloc[-1])

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
    # LOGGING (optional, non-blocking)
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
            # Logging should never break the engine
            pass