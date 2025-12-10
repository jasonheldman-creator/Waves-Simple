"""
waves_engine.py — WAVES Intelligence™ Vector 2.7 Engine
Dynamic Weights + Strategy Overlay + TLH Signals + Slippage + UAPV Unit Price
Benchmark Map v1.1 (LOCKED)

New in this version (2.7)
-------------------------
• SmartSafe-aware behavior (low-vol, low-tilt profile).
• Tax-loss harvesting signals:
    - tlh_candidate_count
    - tlh_candidate_weight (share of Wave weight in TLH candidates)
• Slippage and turnover modeling:
    - daily turnover from dynamic weights
    - daily slippage cost (configurable bps)
    - turnover_annual and slippage_annual_drag (approx)
• UAPV-style unit price:
    - uapv_unit_price = latest wave_value (starting from 1.0)
• Wave-specific custom rules:
    - AI Wave: extra tilt in calm/low-VIX regimes (especially Private Logic™)
    - Quantum Computing Wave: similar but slightly gentler
    - Crypto Equity Wave: aggressive in calm/normal, toned down in elevated/extreme VIX
    - SmartSafe Wave: minimal tilt, conservative behavior

Core functionality (unchanged in spirit)
----------------------------------------
1) Loads list.csv (universe) and wave_weights.csv (Wave definitions).
2) Aggregates duplicate tickers per Wave.
3) Fetches ~1-year daily prices via yfinance.
4) Builds daily Wave returns via dynamic weights:
   • Risk-parity base (inverse 60-day vol).
   • Momentum-based signals (30D & 60D).
   • VIX-regime-adjusted signal tilt.
   • Mode-aware behavior (standard / alpha-minus-beta / private_logic).
5) Applies VIX-based exposure overlay to get final Wave returns.
6) Computes alpha and returns vs blended benchmarks (Benchmark Map v1.1).
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
        slippage_bps: float = 0.0005,  # 5 bps per 100% turnover
        tlh_drawdown_threshold: float = 0.10,  # 10% from recent high
    ):
        self.list_path = Path(list_path)
        self.weights_path = Path(weights_path)
        self.logs_root = Path(logs_root)

        self.slippage_bps = float(slippage_bps)
        self.tlh_drawdown_threshold = float(tlh_drawdown_threshold)

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
            "Small-Mid Cap Growth Wave": "VO",  # overridden by blend
            "Small to Mid Cap Growth Wave": "VO",
            "Clean Transit-Infrastructure Wave": "ICLN",  # overridden by blend
            "Quantum Computing Wave": "IYW",              # overridden by blend
            "AI Wave": "QQQ",                             # overridden by blend
            "Total Market Wave": "VTI",
            "SmartSafe Wave": "SHV",
            "SmartSafe": "SHV",
            "SmartSafe™": "SHV",
            "Crypto Equity Wave": "BTC-USD",              # overridden by blend
            "Crypto Wave": "BTC-USD",
            "Growth Wave": "QQQ",                         # overridden by blend
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

        # Normalise weights per Wave so they sum to 1.0 (initial definition)
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
            return {"QQQ": 0.50, "IWF": 0.50}

        if wave in ("Small-Mid Cap Growth Wave", "Small to Mid Cap Growth Wave"):
            return {"VTWG": 0.50, "VO": 0.50}

        if wave == "AI Wave":
            # AI megacap + semis: 40% QQQ + 60% SOXX
            return {"QQQ": 0.40, "SOXX": 0.60}

        if wave == "Clean Transit-Infrastructure Wave":
            return {"ICLN": 0.50, "IGF": 0.50}

        if wave == "Quantum Computing Wave":
            return {"IYW": 0.70, "SOXX": 0.30}

        if wave == "Crypto Equity Wave":
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
    # PRICE HELPERS
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
                continue

        if not frames:
            raise ValueError(f"No price data for any tickers in: {tickers}")

        closes = pd.concat(frames, axis=1)
        closes.columns = valid_tickers
        closes = closes.dropna(how="all")
        closes = closes.astype(float)
        return closes

    # ---------------------------------------------------------
    # VIX + EXPOSURE HELPERS
    # ---------------------------------------------------------
    def _get_vix_series(self, index_like: pd.Index, period: str = "1y") -> pd.Series:
        vix = self._get_price_series("^VIX", period=period)
        vix.index = pd.to_datetime(vix.index)
        idx = pd.to_datetime(index_like)
        vix_aligned = vix.reindex(idx, method=None)
        vix_aligned = vix_aligned.ffill().bfill()
        vix_aligned.name = "VIX"
        return vix_aligned

    def _get_vol_regime(self, v: float) -> str:
        if np.isnan(v):
            return "unknown"
        if v < 14:
            return "calm"
        if v < 22:
            return "normal"
        if v < 30:
            return "elevated"
        return "extreme"

    def _compute_exposure_series(self, wave: str, mode: str, vix_series: pd.Series) -> pd.Series:
        """
        Per-day exposure based on mode + daily VIX level + wave-specific tweaks.
        """
        wave = (wave or "").strip()
        mode = (mode or "standard").lower()

        # Base exposure by mode
        if mode == "alpha-minus-beta":
            base = 0.8
        elif mode == "private_logic":
            base = 1.1
        else:
            base = 1.0

        # SmartSafe: keep exposure stable and conservative
        if "smartsafe" in wave.lower():
            base = 0.9

        vix_vals = vix_series.astype(float).values
        exposure_vals = []

        for v in vix_vals:
            regime = self._get_vol_regime(v)

            # Default regime multiplier
            if regime == "calm":
                v_mult = 1.05 if mode == "private_logic" else 1.0
            elif regime == "normal":
                v_mult = 0.95
            elif regime == "elevated":
                v_mult = 0.85
            else:  # extreme
                v_mult = 0.6 if mode != "alpha-minus-beta" else 0.5

            # Wave-specific tweaks
            if wave == "Crypto Equity Wave":
                # Extra conservative in elevated/extreme regimes
                if regime in ("elevated", "extreme"):
                    v_mult *= 0.9
            if wave == "AI Wave" and mode == "private_logic":
                # Let AI Wave run a bit hotter in calm regimes
                if regime == "calm":
                    v_mult *= 1.10

            exp_val = base * v_mult
            exp_val = float(np.clip(exp_val, 0.2, 1.4))
            exposure_vals.append(exp_val)

        exposure_series = pd.Series(exposure_vals, index=vix_series.index, name="exposure")
        return exposure_series

    # ---------------------------------------------------------
    # DYNAMIC WEIGHT ENGINE
    # ---------------------------------------------------------
    def _compute_dynamic_weights(
        self,
        wave: str,
        ret_matrix: pd.DataFrame,
        vix_series: pd.Series,
        mode: str,
    ) -> pd.DataFrame:
        """
        Build a [date x ticker] weight matrix using:
          • Risk-parity base (inverse 60-day volatility)
          • Momentum-based signals (30D & 60D)
          • Volatility regime adjustment (via VIX)
          • Mode-based tilt strength
          • Wave-specific customizations
        """
        wave_name = (wave or "").strip()
        mode = (mode or "standard").lower()

        # 60-day realized vol
        vol_60 = ret_matrix.rolling(60).std()

        # Momentum signals (30D & 60D cumulative returns)
        def window_return(arr: np.ndarray) -> float:
            return float(np.prod(1.0 + arr) - 1.0)

        mom30 = (1.0 + ret_matrix).rolling(30).apply(window_return, raw=True)
        mom60 = (1.0 + ret_matrix).rolling(60).apply(window_return, raw=True)

        signal_score = 0.6 * mom30 + 0.4 * mom60

        # Base tilt strength by mode (Option 2: moderate tilt)
        if mode == "alpha-minus-beta":
            base_tilt = 0.20
        elif mode == "private_logic":
            base_tilt = 0.50
        else:
            base_tilt = 0.30

        # SmartSafe: near-zero tilt
        if "smartsafe" in wave_name.lower():
            base_tilt = 0.05

        # Constraints (Can be tightened/loosened per wave)
        w_min = 0.0025   # 0.25%
        w_max = 0.10     # 10%

        # More conservative caps for Crypto Equity Wave
        if wave_name == "Crypto Equity Wave":
            w_max = 0.07

        weights_time = pd.DataFrame(index=ret_matrix.index, columns=ret_matrix.columns, dtype=float)

        for dt in ret_matrix.index:
            vol_row = vol_60.loc[dt]
            sig_row = signal_score.loc[dt]
            vix_val = float(vix_series.loc[dt])

            # Risk-parity base weights
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_vol = 1.0 / vol_row
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
            if inv_vol.notna().sum() == 0:
                valid = ret_matrix.loc[dt].replace(0.0, np.nan).notna()
                if valid.sum() == 0:
                    continue
                rp = valid.astype(float) / valid.sum()
            else:
                inv_vol = inv_vol.fillna(0.0)
                if inv_vol.sum() <= 0:
                    valid = ret_matrix.loc[dt].replace(0.0, np.nan).notna()
                    if valid.sum() == 0:
                        continue
                    rp = valid.astype(float) / valid.sum()
                else:
                    rp = inv_vol / inv_vol.sum()

            # Signal z-scores (cross-sectional)
            if sig_row.notna().sum() >= 2 and sig_row.std(skipna=True) > 0:
                z = (sig_row - sig_row.mean(skipna=True)) / sig_row.std(skipna=True)
            else:
                z = pd.Series(0.0, index=sig_row.index)

            z = z.clip(-2.5, 2.5)
            scaled = z / 2.5  # now ∈ [-1, 1]

            # Volatility regime adjustment to tilt strength
            regime = self._get_vol_regime(vix_val)
            if regime == "calm":
                regime_mult = 1.10
            elif regime == "normal":
                regime_mult = 1.00
            elif regime == "elevated":
                regime_mult = 0.90
            else:  # extreme
                regime_mult = 0.75

            # Wave-specific regime tweaks
            if wave_name == "AI Wave" and mode == "private_logic":
                if regime == "calm":
                    regime_mult *= 1.15
            if wave_name == "Quantum Computing Wave" and mode == "private_logic":
                if regime == "calm":
                    regime_mult *= 1.10
            if wave_name == "Crypto Equity Wave":
                if regime in ("elevated", "extreme"):
                    regime_mult *= 0.8
            if "smartsafe" in wave_name.lower():
                regime_mult *= 0.8  # keep tilts subdued for SmartSafe

            tilt_strength = base_tilt * regime_mult

            tilt_factor = 1.0 + (scaled * tilt_strength)
            tilt_factor = tilt_factor.clip(lower=0.10)

            raw_w = rp * tilt_factor
            if raw_w.sum() <= 0:
                weights = rp
            else:
                weights = raw_w / raw_w.sum()

            # Apply min/max caps and renormalise
            weights = weights.clip(lower=w_min, upper=w_max)
            if weights.sum() <= 0:
                weights = rp
            weights = weights / weights.sum()

            weights_time.loc[dt] = weights

        return weights_time

    # ---------------------------------------------------------
    # TLH SIGNALS
    # ---------------------------------------------------------
    def _compute_tlh_signals(
        self,
        price_matrix: pd.DataFrame,
        current_weights: pd.Series,
    ) -> Dict[str, Union[float, int]]:
        """
        Simple TLH signal:
          • Compute 60D rolling high.
          • Measure drawdown from that high on the last date.
          • Flag tickers down more than tlh_drawdown_threshold (e.g. 10%).
        Returns:
          - tlh_candidate_count
          - tlh_candidate_weight (sum of current dynamic weights in these names)
        """
        if price_matrix.empty or current_weights is None or current_weights.empty:
            return {"tlh_candidate_count": 0, "tlh_candidate_weight": 0.0}

        roll_high = price_matrix.rolling(60).max()
        if roll_high.empty:
            return {"tlh_candidate_count": 0, "tlh_candidate_weight": 0.0}

        dd = price_matrix / roll_high - 1.0
        dd_last = dd.iloc[-1]

        threshold = -abs(self.tlh_drawdown_threshold)
        candidates_mask = dd_last <= threshold

        tickers = [t for t in dd_last.index if candidates_mask.get(t, False)]
        count = len(tickers)
        if count == 0:
            return {"tlh_candidate_count": 0, "tlh_candidate_weight": 0.0}

        weights_aligned = current_weights.reindex(dd_last.index).fillna(0.0)
        tlh_weight = float(weights_aligned[tickers].sum())

        return {
            "tlh_candidate_count": int(count),
            "tlh_candidate_weight": float(tlh_weight),
        }

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
        Compute full metrics for a given Wave + mode using:
          • Dynamic per-day weights
          • VIX-based exposure overlay
          • Slippage + turnover model
          • TLH opportunity diagnostics
          • UAPV-style unit price
        """
        holdings = self.get_wave_holdings(wave)
        if holdings.empty:
            raise ValueError(f"No holdings defined for Wave: {wave}")

        weights_all = (
            holdings.groupby("ticker")["weight"]
            .sum()
            .astype(float)
        )
        all_tickers = list(weights_all.index)

        history_period = "1y"

        # --- Wave prices & returns (portfolio) ---
        price_matrix = self._get_price_matrix(all_tickers, period=history_period)

        ret_matrix = price_matrix.pct_change().iloc[1:, :].astype(float)
        ret_matrix = ret_matrix.dropna(axis=1, how="all")
        if ret_matrix.empty:
            raise ValueError(f"No usable return history for Wave {wave}")

        ret_filled = ret_matrix.fillna(0.0)

        # --- VIX series ---
        vix_series = self._get_vix_series(ret_matrix.index, period=history_period)

        # --- Dynamic weights over time ---
        weights_time = self._compute_dynamic_weights(wave, ret_matrix, vix_series, mode)
        weights_time = weights_time.reindex_like(ret_filled).fillna(0.0)

        # --- Portfolio returns (before & after slippage) ---
        gross_wave_ret = (weights_time * ret_filled).sum(axis=1)
        gross_wave_ret.name = "wave_return_gross"

        # Turnover + slippage
        turnover = weights_time.diff().abs().sum(axis=1) * 0.5
        turnover = turnover.fillna(0.0)
        slippage_cost = turnover * self.slippage_bps

        raw_wave_ret = gross_wave_ret - slippage_cost
        raw_wave_ret.name = "wave_return_raw"

        # --- Strategy overlay: VIX-gated exposure ---
        exposure_series = self._compute_exposure_series(wave, mode, vix_series)

        wave_ret_series = raw_wave_ret * exposure_series
        wave_ret_series.name = "wave_return"

        wave_value = (1.0 + wave_ret_series).cumprod()
        wave_value.name = "wave_value"

        # --- Benchmark prices & returns ---
        benchmark = self.get_benchmark(wave)

        if isinstance(benchmark, dict):
            bm_prices = self._get_price_matrix(list(benchmark.keys()), period=history_period)
            bm_rets = bm_prices.pct_change().iloc[1:, :].astype(float)

            w_bm = pd.Series(benchmark)
            w_bm = w_bm.reindex(bm_rets.columns).fillna(0.0)
            if w_bm.sum() <= 0:
                w_bm = pd.Series(1.0, index=bm_rets.columns)
            w_bm = w_bm / w_bm.sum()

            bm_ret = bm_rets.mul(w_bm.values, axis=1).sum(axis=1)
            bm_ret.name = "benchmark_return"
            bm_value = (1.0 + bm_ret).cumprod()
            bm_value.name = "benchmark_value"
        else:
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
        alpha_1y = alpha_window(df["alpha_captured"], len(df))

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

        # --- Last-day exposure, VIX, regime, weights ---
        exposure_final = float(exposure_series.iloc[-1])
        vix_last = float(vix_series.iloc[-1])
        regime_last = self._get_vol_regime(vix_last)

        current_weights = weights_time.iloc[-1].dropna()
        current_weights = current_weights[current_weights > 0.0]

        history_30d = df.tail(30).copy()

        # --- Turnover & slippage (annualized approximations) ---
        turnover_daily_avg = float(turnover.mean())
        turnover_annual = turnover_daily_avg * 252.0
        slippage_daily_avg = float(slippage_cost.mean())
        slippage_annual_drag = slippage_daily_avg * 252.0

        # --- TLH signals ---
        tlh_signals = self._compute_tlh_signals(price_matrix, current_weights)

        # --- UAPV-style unit price (Wave token price) ---
        uapv_unit_price = float(df["wave_value"].iloc[-1])

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
            "vix_last": vix_last,
            "vol_regime": regime_last,
            "current_weights": current_weights,
            "turnover_annual": turnover_annual,
            "slippage_annual_drag": slippage_annual_drag,
            "tlh_candidate_count": tlh_signals["tlh_candidate_count"],
            "tlh_candidate_weight": tlh_signals["tlh_candidate_weight"],
            "uapv_unit_price": uapv_unit_price,
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
                "vix_last": result.get("vix_last"),
                "vol_regime": result.get("vol_regime"),
                "turnover_annual": result.get("turnover_annual"),
                "slippage_annual_drag": result.get("slippage_annual_drag"),
                "tlh_candidate_count": result.get("tlh_candidate_count"),
                "tlh_candidate_weight": result.get("tlh_candidate_weight"),
                "uapv_unit_price": result.get("uapv_unit_price"),
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