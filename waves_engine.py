"""
waves_engine.py — WAVES Intelligence™ Vector 2.8+ Engine

Core features
-------------
• Loads list.csv (universe) and wave_weights.csv.
• Normalizes legacy Wave names to canonical ones (Crypto, Cloud/Software, etc.).
• Aggregates duplicate tickers per Wave & normalizes weights.
• Fetches ~1-year daily prices via yfinance by default.
• Builds dynamic Wave portfolios using:

    - Risk-parity base (inverse 60-day volatility).
    - Momentum signals (30D & 60D).
    - VIX volatility regime (calm / normal / elevated / extreme).
    - Mode-aware tilt:
        * standard
        * alpha-minus-beta
        * private_logic (more aggressive)
    - Wave-specific logic:
        * SmartSafe: minimal tilt, conservative exposure.
        * AI Wave: extra tilt in calm regimes (esp. Private Logic™).
        * Quantum Computing Wave: similar, slightly gentler.
        * Crypto Equity Wave: more conservative in high VIX.

• VIX-based exposure overlay (gross risk dial).
• Blended benchmarks per Wave (Crypto, Future Power & Energy, Clean Transit,
  Cloud & Enterprise Software / Small Cap Growth, Growth, Small-Mid Growth, AI, Quantum, Income, S&P).
• Computes:
    - Intraday alpha
    - 30D / 60D / 1Y alpha
    - 30D / 60D / 1Y Wave & Benchmark returns
    - Realized beta (~60 bars)
    - UAPV unit price (Wave token/unit value)
    - TLH opportunity stats + details
    - Turnover & slippage drag

Additional:
-----------
• SmartSafeSweepEngine: multi-Wave + SmartSafe allocation layer.
• get_wave_performance_with_prices(): run strategy on custom price matrices
  (e.g., intraday or external data), instead of yfinance.
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
        tlh_drawdown_threshold: float = 0.10,  # 10% drawdown from 60D high
    ):
        self.list_path = Path(list_path)
        self.weights_path = Path(weights_path)
        self.logs_root = Path(logs_root)

        self.slippage_bps = float(slippage_bps)
        self.tlh_drawdown_threshold = float(tlh_drawdown_threshold)

        self.universe = self._load_list()
        self.weights = self._load_weights()

        # Baseline single-ticker fallback benchmarks
        self._benchmark_map: Dict[str, str] = {
            "S&P Wave": "SPY",
            "S&P 500 Wave": "SPY",
            "Income Wave": "SCHD",
            "Dividend Income Wave": "SCHD",
            "Small Cap Growth Wave": "VTWG",  # overridden by Cloud/Enterprise blend
            "Small-Mid Cap Growth Wave": "VO",  # overridden by blend
            "Small to Mid Cap Growth Wave": "VO",
            "Total Market Wave": "VTI",
            "SmartSafe Wave": "SHV",
            "SmartSafe": "SHV",
            "SmartSafe™": "SHV",
            "Growth Wave": "QQQ",  # overridden by QQQ/IWF blend
            "AI Wave": "QQQ",      # overridden by QQQ/SOXX blend
            "Quantum Computing Wave": "IYW",  # overridden by IYW/SOXX blend
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

        # Optional metadata
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

        # 🔁 NAME NORMALIZATION: map legacy labels to canonical names
        name_map = {
            # Crypto
            "Crypto Income Wave": "Crypto Equity Wave (mid/large cap)",
            "Crypto Equity Wave": "Crypto Equity Wave (mid/large cap)",

            # Small cap growth → Cloud/Enterprise Software
            "Small Cap Growth Wave": "Cloud & Enterprise Software Growth Wave",
            "Cloud Computing & Enterprise Software Growth Fund": "Cloud & Enterprise Software Growth Wave",

            # Future Power & Energy minor variants
            "Future Power & Energy": "Future Power & Energy Wave",

            # Clean Transit variants
            "Clean Transit - Infrastructure Wave": "Clean Transit-Infrastructure Wave",
            "Clean Transit & Infrastructure Wave": "Clean Transit-Infrastructure Wave",
        }
        df["wave"] = df["wave"].replace(name_map)

        df = df.dropna(subset=["ticker", "weight"])
        df["weight"] = df["weight"].astype(float)

        # Normalize per Wave so weights sum to 1
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
        Returns:
            • Single ticker string, e.g. "SPY"
            • Dict of {ticker: weight} for blended benchmarks
        """
        wave = wave.strip()

        # --- Crypto Equity Wave (mid/large cap) ---
        # 70% FBTC (spot BTC ETF), 30% DAPP (digital asset equities)
        if wave in ("Crypto Equity Wave (mid/large cap)", "Crypto Equity Wave", "Crypto Income Wave"):
            return {"FBTC": 0.70, "DAPP": 0.30}

        # --- Future Power & Energy Wave ---
        # 65% XLE (Energy), 25% XLU (Utilities), 10% SPY (broad market)
        if wave in ("Future Power & Energy Wave", "Future Power & Energy"):
            return {"XLE": 0.65, "XLU": 0.25, "SPY": 0.10}

        # --- Clean Transit-Infrastructure Wave ---
        # 45% FIDU (Industrials), 45% FDIS (Consumer Disc), 10% SPY
        if wave == "Clean Transit-Infrastructure Wave":
            return {"FIDU": 0.45, "FDIS": 0.45, "SPY": 0.10}

        # --- Cloud & Enterprise Software Growth (formerly Small Cap Growth Wave) ---
        # 60% IGV, 20% WCLD, 20% SPY
        if wave in (
            "Cloud & Enterprise Software Growth Wave",
            "Cloud Computing & Enterprise Software Growth Fund",
            "Small Cap Growth Wave",
        ):
            return {"IGV": 0.60, "WCLD": 0.20, "SPY": 0.20}

        # --- Growth Wave (large-cap growth) ---
        # 50% QQQ, 50% IWF
        if wave == "Growth Wave":
            return {"QQQ": 0.50, "IWF": 0.50}

        # --- Small-Mid Cap Growth Wave ---
        # 50% VTWG (small-cap growth), 50% VO (mid-cap)
        if wave in ("Small-Mid Cap Growth Wave", "Small to Mid Cap Growth Wave"):
            return {"VTWG": 0.50, "VO": 0.50}

        # --- AI Wave ---
        # 40% QQQ, 60% SOXX
        if wave == "AI Wave":
            return {"QQQ": 0.40, "SOXX": 0.60}

        # --- Quantum Computing Wave ---
        # 70% IYW (tech), 30% SOXX (semi)
        if wave == "Quantum Computing Wave":
            return {"IYW": 0.70, "SOXX": 0.30}

        # --- Income Wave ---
        if wave == "Income Wave":
            return "SCHD"

        # --- S&P Wave & variants ---
        if wave in ("S&P Wave", "S&P 500 Wave"):
            return "SPY"

        # Fallback single-ticker
        return self._benchmark_map.get(wave, "SPY")

    def get_wave_holdings(self, wave: str) -> pd.DataFrame:
        """
        Holdings for a Wave, with duplicate tickers aggregated (weights summed).
        """
        w = self.weights[self.weights["wave"] == wave].copy()
        if w.empty:
            return w

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
        frames = []
        valid_tickers: List[str] = []
        for t in tickers:
            try:
                s = self._get_price_series(t, period=period)
                frames.append(s)
                valid_tickers.append(t)
            except Exception:
                # Skip tickers with no data
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

        # SmartSafe: keep exposure conservative
        if "smartsafe" in wave.lower():
            base = 0.9

        vix_vals = vix_series.astype(float).values
        exposure_vals = []

        for v in vix_vals:
            regime = self._get_vol_regime(v)

            if regime == "calm":
                v_mult = 1.05 if mode == "private_logic" else 1.0
            elif regime == "normal":
                v_mult = 0.95
            elif regime == "elevated":
                v_mult = 0.85
            else:  # extreme
                v_mult = 0.6 if mode != "alpha-minus-beta" else 0.5

            # Wave-specific tweaks
            if wave in ("Crypto Equity Wave (mid/large cap)", "Crypto Equity Wave"):
                if regime in ("elevated", "extreme"):
                    v_mult *= 0.9
            if wave == "AI Wave" and mode == "private_logic":
                if regime == "calm":
                    v_mult *= 1.10

            exp_val = base * v_mult
            exp_val = float(np.clip(exp_val, 0.2, 1.4))
            exposure_vals.append(exp_val)

        return pd.Series(exposure_vals, index=vix_series.index, name="exposure")

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
          • Momentum signals (30D & 60D)
          • Volatility regime adjustment (via VIX)
          • Mode-based tilt strength
          • Wave-specific customizations
        """
        wave_name = (wave or "").strip()
        mode = (mode or "standard").lower()

        # 60-day realized vol
        vol_60 = ret_matrix.rolling(60).std()

        # Momentum signals
        def window_return(arr: np.ndarray) -> float:
            return float(np.prod(1.0 + arr) - 1.0)

        mom30 = (1.0 + ret_matrix).rolling(30).apply(window_return, raw=True)
        mom60 = (1.0 + ret_matrix).rolling(60).apply(window_return, raw=True)

        signal_score = 0.6 * mom30 + 0.4 * mom60

        # Base tilt strength by mode
        if mode == "alpha-minus-beta":
            base_tilt = 0.20
        elif mode == "private_logic":
            base_tilt = 0.50
        else:
            base_tilt = 0.30

        # SmartSafe: minimal tilt
        if "smartsafe" in wave_name.lower():
            base_tilt = 0.05

        # Base caps
        w_min = 0.0025   # 0.25%
        w_max = 0.10     # 10%

        # Crypto a bit tighter
        if wave_name in ("Crypto Equity Wave (mid/large cap)", "Crypto Equity Wave"):
            w_max = 0.07

        weights_time = pd.DataFrame(index=ret_matrix.index, columns=ret_matrix.columns, dtype=float)

        for dt in ret_matrix.index:
            vol_row = vol_60.loc[dt]
            sig_row = signal_score.loc[dt]
            vix_val = float(vix_series.loc[dt])

            # Risk-parity base
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

            # Signal z-scores
            if sig_row.notna().sum() >= 2 and sig_row.std(skipna=True) > 0:
                z = (sig_row - sig_row.mean(skipna=True)) / sig_row.std(skipna=True)
            else:
                z = pd.Series(0.0, index=sig_row.index)

            z = z.clip(-2.5, 2.5)
            scaled = z / 2.5  # [-1,1]

            # Vol regime adjustment
            regime = self._get_vol_regime(vix_val)
            if regime == "calm":
                regime_mult = 1.10
            elif regime == "normal":
                regime_mult = 1.00
            elif regime == "elevated":
                regime_mult = 0.90
            else:
                regime_mult = 0.75

            # Wave-specific regime tweaks
            if wave_name == "AI Wave" and mode == "private_logic":
                if regime == "calm":
                    regime_mult *= 1.15
            if wave_name == "Quantum Computing Wave" and mode == "private_logic":
                if regime == "calm":
                    regime_mult *= 1.10
            if wave_name in ("Crypto Equity Wave (mid/large cap)", "Crypto Equity Wave"):
                if regime in ("elevated", "extreme"):
                    regime_mult *= 0.8
            if "smartsafe" in wave_name.lower():
                regime_mult *= 0.8

            tilt_strength = base_tilt * regime_mult

            tilt_factor = 1.0 + (scaled * tilt_strength)
            tilt_factor = tilt_factor.clip(lower=0.10)

            raw_w = rp * tilt_factor
            if raw_w.sum() <= 0:
                weights = rp
            else:
                weights = raw_w / raw_w.sum()

            # Caps and renorm
            weights = weights.clip(lower=w_min, upper=w_max)
            if weights.sum() <= 0:
                weights = rp
            weights = weights / weights.sum()

            weights_time.loc[dt] = weights

        return weights_time

    # ---------------------------------------------------------
    # TLH SIGNALS (UPGRADED)
    # ---------------------------------------------------------
    def _compute_tlh_signals(
        self,
        price_matrix: pd.DataFrame,
        current_weights: pd.Series,
    ) -> Dict[str, object]:
        """
        TLH signal engine (still heuristic, NOT legal/tax advice):

        • Looks back 60 days, finds max price for each ticker.
        • Computes current drawdown (% below 60D high).
        • Flags tickers below -self.tlh_drawdown_threshold (e.g., -10%).
        • Summarizes:
            - tlh_candidate_count
            - tlh_candidate_weight
        • Also produces a details DataFrame with suggested 'replacement' tickers.

        NOTE: Replacement tickers are crude, illustrative proxies. A real TLH
        engine would consider wash-sale rules, factor exposure, issuer, etc.
        """
        if price_matrix.empty or current_weights is None or current_weights.empty:
            return {
                "tlh_candidate_count": 0,
                "tlh_candidate_weight": 0.0,
                "tlh_details": None,
            }

        roll_high = price_matrix.rolling(60).max()
        if roll_high.empty:
            return {
                "tlh_candidate_count": 0,
                "tlh_candidate_weight": 0.0,
                "tlh_details": None,
            }

        dd = price_matrix / roll_high - 1.0
        dd_last = dd.iloc[-1]

        threshold = -abs(self.tlh_drawdown_threshold)
        candidates_mask = dd_last <= threshold

        tickers = [t for t in dd_last.index if bool(candidates_mask.get(t, False))]
        count = len(tickers)
        if count == 0:
            return {
                "tlh_candidate_count": 0,
                "tlh_candidate_weight": 0.0,
                "tlh_details": None,
            }

        weights_aligned = current_weights.reindex(dd_last.index).fillna(0.0)
        tlh_weight = float(weights_aligned[tickers].sum())

        # Very simple "replacement" mapping for illustration
        replacement_map: Dict[str, str] = {
            "SPY": "VOO",
            "VOO": "SPY",
            "IVV": "SPY",
            "QQQ": "VUG",
            "IWF": "VUG",
            "VUG": "QQQ",
            "VTWG": "IJT",
            "IJT": "VTWG",
            "VO": "IWR",
            "IWR": "VO",
            "SCHD": "VYM",
            "VYM": "SCHD",
            "IGV": "VGT",
            "VGT": "IGV",
            "WCLD": "CLOU",
            "SOXX": "SMH",
            "SMH": "SOXX",
            "IYW": "XLK",
            "XLK": "IYW",
            "FBTC": "IBIT",
            "IBIT": "FBTC",
            "DAPP": "BKCH",
            "BKCH": "DAPP",
        }

        rows = []
        for t in tickers:
            rows.append(
                {
                    "ticker": t,
                    "current_weight": float(weights_aligned.get(t, 0.0)),
                    "drawdown_from_60d_high": float(dd_last[t]),
                    "suggested_replacement": replacement_map.get(t, None),
                }
            )

        details_df = pd.DataFrame(rows).sort_values(
            "drawdown_from_60d_high"
        ).reset_index(drop=True)

        return {
            "tlh_candidate_count": int(count),
            "tlh_candidate_weight": float(tlh_weight),
            "tlh_details": details_df,
        }

    # ---------------------------------------------------------
    # CORE PERFORMANCE (DEFAULT, USING YFINANCE PRICES)
    # ---------------------------------------------------------
    def get_wave_performance(
        self,
        wave: str,
        mode: str = "standard",
        log: bool = False,
    ) -> Optional[dict]:
        """
        Full metrics for a given Wave + mode, using yfinance data:
          • Dynamic per-day weights
          • VIX-based exposure overlay
          • Slippage + turnover model
          • TLH diagnostics
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

        # Prices & returns
        price_matrix = self._get_price_matrix(all_tickers, period=history_period)
        ret_matrix = price_matrix.pct_change().iloc[1:, :].astype(float)
        ret_matrix = ret_matrix.dropna(axis=1, how="all")
        if ret_matrix.empty:
            raise ValueError(f"No usable return history for Wave {wave}")
        ret_filled = ret_matrix.fillna(0.0)

        vix_series = self._get_vix_series(ret_matrix.index, period=history_period)

        # Dynamic weights
        weights_time = self._compute_dynamic_weights(wave, ret_matrix, vix_series, mode)
        weights_time = weights_time.reindex_like(ret_filled).fillna(0.0)

        # Portfolio returns (before & after slippage)
        gross_wave_ret = (weights_time * ret_filled).sum(axis=1)
        gross_wave_ret.name = "wave_return_gross"

        turnover = weights_time.diff().abs().sum(axis=1) * 0.5
        turnover = turnover.fillna(0.0)
        slippage_cost = turnover * self.slippage_bps

        raw_wave_ret = gross_wave_ret - slippage_cost
        raw_wave_ret.name = "wave_return_raw"

        exposure_series = self._compute_exposure_series(wave, mode, vix_series)

        wave_ret_series = raw_wave_ret * exposure_series
        wave_ret_series.name = "wave_return"

        wave_value = (1.0 + wave_ret_series).cumprod()
        wave_value.name = "wave_value"

        # Benchmark
        benchmark = self.get_benchmark(wave)
        if isinstance(benchmark, dict):
            bm_prices = self._get_price_matrix(list(benchmark.keys()), period=history_period)
            bm_rets = bm_prices.pct_change().iloc[1:, :].astype(float)

            w_bm = pd.Series(benchmark)
            w_bm = w_bm.reindex(bm_rets.columns).fillna(0.0)
            if w_bm.sum() <= 0:
                w_bm = pd.Series(1.0, index=bm_rets.columns)
            else:
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

        df = pd.concat([wave_ret_series, bm_ret, wave_value, bm_value], axis=1).dropna()
        if df.empty:
            raise ValueError(f"No overlapping Wave/benchmark history for {wave}")

        df["alpha_captured"] = df["wave_return"] - df["benchmark_return"]

        # Alpha windows
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

        # Return windows
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

        # Realized beta (≈60D)
        beta_realized = np.nan
        tail_n = min(60, len(df))
        if tail_n >= 20:
            x = df["benchmark_return"].tail(tail_n).values.flatten()
            y = df["wave_return"].tail(tail_n).values.flatten()
            if np.var(x) > 0:
                cov_xy = np.cov(x, y)[0, 1]
                beta_realized = float(cov_xy / np.var(x))

        exposure_final = float(exposure_series.iloc[-1])
        vix_last = float(vix_series.iloc[-1])
        regime_last = self._get_vol_regime(vix_last)

        current_weights = weights_time.iloc[-1].dropna()
        current_weights = current_weights[current_weights > 0.0]

        history_30d = df.tail(30).copy()

        turnover_daily_avg = float(turnover.mean())
        turnover_annual = turnover_daily_avg * 252.0
        slippage_daily_avg = float(slippage_cost.mean())
        slippage_annual_drag = slippage_daily_avg * 252.0

        tlh_signals = self._compute_tlh_signals(price_matrix, current_weights)
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
            "tlh_details": tlh_signals["tlh_details"],
            "uapv_unit_price": uapv_unit_price,
        }

        if log:
            self._log_performance_row(wave, result)

        return result

    # ---------------------------------------------------------
    # VARIANT: USE CUSTOM PRICE MATRIX (E.G. INTRADAY)
    # ---------------------------------------------------------
    def get_wave_performance_with_prices(
        self,
        wave: str,
        mode: str,
        price_matrix: pd.DataFrame,
        log: bool = False,
    ) -> Optional[dict]:
        """
        Variant of get_wave_performance that uses a caller-provided price_matrix
        instead of pulling from yfinance.

        price_matrix:
            • DataFrame indexed by datetime
            • Columns are tickers
            • Should contain all tickers for the Wave (and benchmark if desired)
            • Can be daily or intraday bars (strategy is time-step agnostic).

        NOTE:
            - You are responsible for ensuring the frequency & history are
              appropriate (e.g., at least ~60–252 bars).
            - For intraday, the alpha math is the same but over finer bars.
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

        # Subset the provided price_matrix to needed tickers
        price_matrix = price_matrix.copy()
        missing = [t for t in all_tickers if t not in price_matrix.columns]
        if len(missing) == len(all_tickers):
            raise ValueError(f"None of the Wave tickers are in the provided price matrix: {missing}")

        price_matrix = price_matrix.loc[:, price_matrix.columns.isin(all_tickers)]
        price_matrix = price_matrix.dropna(how="all")

        ret_matrix = price_matrix.pct_change().iloc[1:, :].astype(float)
        ret_matrix = ret_matrix.dropna(axis=1, how="all")
        if ret_matrix.empty:
            raise ValueError(f"No usable return history for Wave {wave} in provided prices")
        ret_filled = ret_matrix.fillna(0.0)

        # For now, reuse ^VIX (daily) and align to the custom index
        vix_series = self._get_vix_series(ret_matrix.index, period="1y")

        # Dynamic weights & performance identical to get_wave_performance
        weights_time = self._compute_dynamic_weights(wave, ret_matrix, vix_series, mode)
        weights_time = weights_time.reindex_like(ret_filled).fillna(0.0)

        gross_wave_ret = (weights_time * ret_filled).sum(axis=1)
        gross_wave_ret.name = "wave_return_gross"

        turnover = weights_time.diff().abs().sum(axis=1) * 0.5
        turnover = turnover.fillna(0.0)
        slippage_cost = turnover * self.slippage_bps

        raw_wave_ret = gross_wave_ret - slippage_cost
        raw_wave_ret.name = "wave_return_raw"

        exposure_series = self._compute_exposure_series(wave, mode, vix_series)
        wave_ret_series = raw_wave_ret * exposure_series
        wave_ret_series.name = "wave_return"

        wave_value = (1.0 + wave_ret_series).cumprod()
        wave_value.name = "wave_value"

        # Benchmark still pulled via get_benchmark / yfinance for now
        benchmark = self.get_benchmark(wave)
        if isinstance(benchmark, dict):
            bm_prices = self._get_price_matrix(list(benchmark.keys()), period="1y")
            bm_rets = bm_prices.pct_change().iloc[1:, :].astype(float)
            w_bm = pd.Series(benchmark).reindex(bm_rets.columns).fillna(0.0)
            if w_bm.sum() <= 0:
                w_bm = pd.Series(1.0, index=bm_rets.columns)
            else:
                w_bm = w_bm / w_bm.sum()
            bm_ret = bm_rets.mul(w_bm.values, axis=1).sum(axis=1)
            bm_ret.name = "benchmark_return"
            bm_value = (1.0 + bm_ret).cumprod()
            bm_value.name = "benchmark_value"
        else:
            bm_price = self._get_price_series(benchmark, period="1y")
            bm_ret = bm_price.pct_change().iloc[1:].astype(float)
            bm_ret.name = "benchmark_return"
            bm_value = (1.0 + bm_ret).cumprod()
            bm_value.name = "benchmark_value"

        df = pd.concat([wave_ret_series, bm_ret, wave_value, bm_value], axis=1).dropna()
        if df.empty:
            raise ValueError(f"No overlapping Wave/benchmark history for {wave} (custom prices)")

        df["alpha_captured"] = df["wave_return"] - df["benchmark_return"]

        # Alpha windows
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

        # Realized beta (~60 bars)
        beta_realized = np.nan
        tail_n = min(60, len(df))
        if tail_n >= 20:
            x = df["benchmark_return"].tail(tail_n).values.flatten()
            y = df["wave_return"].tail(tail_n).values.flatten()
            if np.var(x) > 0:
                cov_xy = np.cov(x, y)[0, 1]
                beta_realized = float(cov_xy / np.var(x))

        exposure_final = float(exposure_series.iloc[-1])
        vix_last = float(vix_series.iloc[-1])
        regime_last = self._get_vol_regime(vix_last)

        current_weights = weights_time.iloc[-1].dropna()
        current_weights = current_weights[current_weights > 0.0]

        history_30d = df.tail(30).copy()

        turnover_daily_avg = float(turnover.mean())
        turnover_annual = turnover_daily_avg * 252.0
        slippage_daily_avg = float(slippage_cost.mean())
        slippage_annual_drag = slippage_daily_avg * 252.0

        tlh_signals = self._compute_tlh_signals(price_matrix, current_weights)
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
            "tlh_details": tlh_signals["tlh_details"],
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
            # Logging must never break engine
            pass


# -------------------------------------------------------------
# SmartSafeSweepEngine — multi-Wave + SmartSafe allocator layer
# -------------------------------------------------------------
class SmartSafeSweepEngine:
    """
    SmartSafeSweepEngine — multi-Wave + SmartSafe allocator

    This is a light "household" layer on top of WavesEngine. It does NOT place
    trades; it just suggests how much to put in:
      • risk Waves (all non-SmartSafe waves)
      • SmartSafe Wave
    based on a simple risk level and current VIX regime.
    """

    def __init__(self, engine: WavesEngine, smartsafe_wave_name: str = "SmartSafe Wave"):
        self.engine = engine
        self.smartsafe_wave_name = smartsafe_wave_name

    def _split_waves(self) -> tuple[List[str], List[str]]:
        all_waves = self.engine.get_wave_names()
        smart = [w for w in all_waves if self.smartsafe_wave_name.lower() in w.lower()]
        risk = [w for w in all_waves if w not in smart]
        return risk, smart

    def recommend_allocation(self, mode: str, risk_level: str) -> dict:
        """
        Returns a dict of {Wave: allocation_weight} that sums to 1.0.

        risk_level: "Conservative", "Moderate", "Aggressive"
        """

        risk_waves, smart_waves = self._split_waves()
        if not smart_waves:
            # no SmartSafe found — allocate only across risk waves
            w = 1.0 / max(1, len(risk_waves))
            return {rw: w for rw in risk_waves}

        smart_wave = smart_waves[0]

        rl = risk_level.lower()
        if rl.startswith("con"):
            smart_alloc = 0.60
        elif rl.startswith("agg"):
            smart_alloc = 0.20
        else:
            smart_alloc = 0.40  # Moderate default

        # Simple VIX-based tweak: push more to SmartSafe if regime is elevated/extreme
        try:
            # Just inspect the first risk wave for VIX regime
            m0 = self.engine.get_wave_performance(risk_waves[0], mode=mode, log=False)
            regime = m0.get("vol_regime", "normal")
            if regime in ("elevated", "extreme"):
                smart_alloc = min(0.80, smart_alloc + 0.10)
        except Exception:
            pass

        risk_alloc_total = 1.0 - smart_alloc
        per_risk = risk_alloc_total / max(1, len(risk_waves))

        alloc = {rw: per_risk for rw in risk_waves}
        alloc[smart_wave] = smart_alloc
        return alloc

    def evaluate_portfolio(self, allocations: dict, mode: str) -> dict:
        """
        Given {Wave: allocation} weights (sum ≈ 1.0),
        compute blended Wave performance across Waves.
        """

        waves = list(allocations.keys())
        weights = np.array([allocations[w] for w in waves], dtype=float)
        if weights.sum() <= 0:
            return {}

        weights = weights / weights.sum()

        entries = []
        for i, w in enumerate(waves):
            try:
                m = self.engine.get_wave_performance(w, mode=mode, log=False)
                entries.append((w, m, weights[i]))
            except Exception:
                continue

        if not entries:
            return {}

        def _blend(key: str) -> Optional[float]:
            vals = []
            ws = []
            for wname, m, wgt in entries:
                v = m.get(key, None)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                vals.append(v)
                ws.append(wgt)
            if not vals or sum(ws) <= 0:
                return None
            ws = np.array(ws)
            ws = ws / ws.sum()
            return float(np.dot(ws, np.array(vals)))

        return {
            "allocations": {w: float(a) for w, a in allocations.items()},
            "alpha_30d_blended": _blend("alpha_30d"),
            "alpha_60d_blended": _blend("alpha_60d"),
            "alpha_1y_blended": _blend("alpha_1y"),
            "return_30d_wave_blended": _blend("return_30d_wave"),
            "return_60d_wave_blended": _blend("return_60d_wave"),
            "return_1y_wave_blended": _blend("return_1y_wave"),