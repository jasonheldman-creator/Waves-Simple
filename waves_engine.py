import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


class WavesEngine:
    """
    WavesEngine — Core engine for WAVES Intelligence™
    
    Manages wave performance tracking, dynamic weighting, VIX-aware exposure,
    and benchmark-relative alpha computation.
    """

    def __init__(self, list_path: str, weights_path: str, logs_root: str = "logs"):
        """
        Initialize WavesEngine with universe and wave weights.
        
        Args:
            list_path: Path to the universe CSV (list of stocks)
            weights_path: Path to wave weights CSV
            logs_root: Directory for logs
        """
        self.list_path = Path(list_path)
        self.weights_path = Path(weights_path)
        self.logs_root = Path(logs_root)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        
        # Load wave weights
        self.wave_weights_df = pd.read_csv(self.weights_path)
        
        # Load universe list if exists
        if self.list_path.exists():
            self.universe_df = pd.read_csv(self.list_path)
        else:
            self.universe_df = pd.DataFrame()

    def get_wave_names(self) -> List[str]:
        """Get list of all wave names."""
        if "wave" in self.wave_weights_df.columns:
            return sorted(self.wave_weights_df["wave"].unique().tolist())
        return []

    def get_wave_holdings(self, wave: str) -> Optional[pd.DataFrame]:
        """
        Get holdings for a specific wave.
        
        Args:
            wave: Name of the wave
            
        Returns:
            DataFrame with ticker, company, sector, and weight columns
        """
        if self.wave_weights_df.empty:
            return None
            
        wave_data = self.wave_weights_df[self.wave_weights_df["wave"] == wave].copy()
        if wave_data.empty:
            return None
            
        # Merge with universe data if available
        if not self.universe_df.empty and "ticker" in wave_data.columns:
            # Normalize ticker columns
            if "Ticker" in self.universe_df.columns:
                ticker_col = "Ticker"
            elif "ticker" in self.universe_df.columns:
                ticker_col = "ticker"
            else:
                ticker_col = None
                
            if ticker_col:
                universe_subset = self.universe_df[[ticker_col, "Company", "Sector"]].copy() if "Company" in self.universe_df.columns and "Sector" in self.universe_df.columns else self.universe_df
                wave_data = wave_data.merge(
                    universe_subset,
                    left_on="ticker",
                    right_on=ticker_col,
                    how="left"
                )
        
        # Ensure standard columns exist
        if "company" not in wave_data.columns:
            wave_data["company"] = wave_data.get("ticker", "")
        if "sector" not in wave_data.columns:
            wave_data["sector"] = "Unknown"
            
        return wave_data

    def _get_price_series(self, ticker: str, period: str = "1y") -> pd.Series:
        """
        Get price series for a ticker (mock implementation).
        
        Args:
            ticker: Stock ticker symbol
            period: Time period
            
        Returns:
            Series of prices indexed by date
        """
        # Mock price data for demo
        dates = pd.date_range(end=pd.Timestamp.today(), periods=252, freq="D")
        prices = 100 * (1 + np.random.randn(252).cumsum() * 0.01)
        return pd.Series(prices, index=dates, name=ticker)

    def _get_vol_regime(self, vix: float) -> str:
        """
        Determine volatility regime based on VIX level.
        
        Args:
            vix: VIX level
            
        Returns:
            Regime string: "low", "normal", "elevated", or "extreme"
        """
        if vix < 15:
            return "low"
        elif vix < 20:
            return "normal"
        elif vix < 30:
            return "elevated"
        else:
            return "extreme"

    def _compute_tlh_signals(self, price_matrix: pd.DataFrame, current_weights: pd.Series) -> Dict:
        """
        Compute Tax Loss Harvesting signals.
        
        Args:
            price_matrix: DataFrame of historical prices
            current_weights: Series of current position weights
            
        Returns:
            Dictionary with TLH candidates info
        """
        if price_matrix.empty or current_weights.empty:
            return {
                "tlh_candidate_count": 0,
                "tlh_candidate_weight": 0.0,
                "tlh_details": []
            }
        
        # Calculate high watermark over last 60 days
        lookback = min(60, len(price_matrix))
        recent_prices = price_matrix.tail(lookback)
        
        tlh_candidates = []
        for ticker in current_weights.index:
            if ticker not in recent_prices.columns:
                continue
                
            ticker_prices = recent_prices[ticker].dropna()
            if len(ticker_prices) < 2:
                continue
                
            high_60d = ticker_prices.max()
            current_price = ticker_prices.iloc[-1]
            
            if current_price < high_60d * 0.9:  # More than 10% below high
                drawdown = (current_price / high_60d) - 1.0
                weight = current_weights[ticker]
                tlh_candidates.append({
                    "ticker": ticker,
                    "drawdown": drawdown,
                    "weight": weight
                })
        
        total_tlh_weight = sum(c["weight"] for c in tlh_candidates)
        
        return {
            "tlh_candidate_count": len(tlh_candidates),
            "tlh_candidate_weight": total_tlh_weight,
            "tlh_details": tlh_candidates
        }

    def get_wave_performance(self, wave: str, mode: str = "standard", log: bool = False) -> Dict:
        """
        Calculate wave performance metrics.
        
        This is the continuation of the method that was truncated in the original file.
        The original file started with the benchmark processing code.
        
        Args:
            wave: Name of the wave
            mode: Performance calculation mode
            log: Whether to log the results
            
        Returns:
            Dictionary with performance metrics
        """
        # Get wave holdings and weights
        holdings = self.get_wave_holdings(wave)
        if holdings is None or holdings.empty:
            raise ValueError(f"No holdings found for wave: {wave}")
        
        # Get tickers for this wave
        if "ticker" in holdings.columns:
            tickers = holdings["ticker"].unique().tolist()
        else:
            raise ValueError(f"No ticker column in holdings for wave: {wave}")
        
        # Build price matrix for wave constituents
        price_data = {}
        for ticker in tickers:
            try:
                price_data[ticker] = self._get_price_series(ticker, period="1y")
            except Exception:
                continue
        
        if not price_data:
            raise ValueError(f"No price data available for wave: {wave}")
        
        price_matrix = pd.DataFrame(price_data)
        price_matrix = price_matrix.dropna(how="all")
        
        # Get wave weights over time
        wave_holdings = holdings[holdings["ticker"].isin(price_matrix.columns)].copy()
        weights = wave_holdings.set_index("ticker")["weight"]
        weights = weights / weights.sum()  # Normalize
        
        # Build weight matrix (same weights over time for simplicity)
        weights_time = pd.DataFrame(
            {ticker: weights.get(ticker, 0.0) for ticker in price_matrix.columns},
            index=price_matrix.index
        )
        
        # Calculate wave returns
        returns = price_matrix.pct_change().iloc[1:]
        wave_returns = (returns * weights_time.iloc[1:]).sum(axis=1)
        wave_returns.name = "wave_return"
        
        # Calculate wave value
        wave_value = (1.0 + wave_returns).cumprod()
        wave_value.name = "wave_value"
        
        # Mock VIX data
        vix_series = pd.Series(
            15 + 5 * np.random.randn(len(price_matrix)),
            index=price_matrix.index
        ).clip(10, 50)
        
        # VIX-based exposure adjustment
        exposure_series = vix_series.apply(lambda v: 1.0 if v < 20 else 0.8 if v < 30 else 0.6)
        
        # Mock turnover and slippage
        turnover = pd.Series(0.01, index=returns.index)  # 1% daily turnover
        slippage_cost = turnover * 0.0005  # 5 bps slippage
        
        # Determine benchmark (simplified - currently uses SPY)
        # Future enhancement: Load benchmark from config per wave
        # Supports both string ticker (e.g., "SPY") or dict of {ticker: weight}
        benchmark = "SPY"  # Default benchmark
        
        # Benchmark processing - handles both dict and string benchmarks
        if isinstance(benchmark, dict):
            bm_tickers = list(benchmark.keys())
            bm_data = {}
            for t in bm_tickers:
                try:
                    bm_data[t] = self._get_price_series(t, period="1y")
                except Exception:
                    continue
            if not bm_data:
                raise ValueError(f"No benchmark price data for {wave}")
            bm_prices = pd.DataFrame(bm_data)
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

        df = pd.concat([wave_returns, bm_ret, wave_value, bm_value], axis=1).dropna()
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

    def _split_waves(self) -> Tuple[List[str], List[str]]:
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
        }
