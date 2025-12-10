# waves_engine.py
"""
WAVES Intelligence™ Engine (Safe Mode)

- Loads wave_weights.csv
- Builds 10 Waves (including AI & SmartSafe)
- Downloads price history via yfinance
- Computes Wave & Benchmark returns
- Computes 60D and 1Y alpha
- Safe Mode: any missing/failed ticker is dropped,
  the Wave still runs as long as ≥1 ticker has data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import logging
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("waves_engine")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------
@dataclass
class WaveMetrics:
    wave: str
    ret_60d: float
    alpha_60d: float
    ret_1y: float
    alpha_1y: float
    n_holdings_used: int
    n_holdings_defined: int
    warnings: List[str]


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------
class WavesEngine:
    """
    Core engine responsible for:
    - loading weights
    - downloading prices
    - computing Wave & benchmark returns
    - returning metrics in a pandas DataFrame
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.weights_path = self.base_dir / "wave_weights.csv"

        if not self.weights_path.exists():
            raise FileNotFoundError(f"wave_weights.csv not found at {self.weights_path}")

        self.wave_weights = self._load_wave_weights()

        # Pre-compute universe of all tickers we need
        self.all_tickers = sorted(self.wave_weights["ticker"].unique())

        # Benchmark blends per Wave: {wave: {ticker: weight}}
        # Weights must sum to 1.
        self.benchmarks: Dict[str, Dict[str, float]] = self._build_benchmark_map()

        # Price history cache
        self.price_history: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_all_metrics(self) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Returns:
            metrics_df: DataFrame with one row per Wave.
            warnings:   dict[wave -> list of warning strings]
        """
        logger.info("Computing all Wave metrics (Safe Mode)...")

        self.price_history = self._download_price_history(self.all_tickers)

        metrics: List[WaveMetrics] = []
        warnings: Dict[str, List[str]] = {}

        for wave_name in sorted(self.wave_weights["wave"].unique()):
            wave_warn: List[str] = []
            try:
                wm = self._compute_metrics_for_wave(wave_name, wave_warn)
                if wm is not None:
                    metrics.append(wm)
            except Exception as exc:  # Safe mode: never kill entire engine
                msg = f"Wave '{wave_name}' failed entirely: {exc}"
                logger.exception(msg)
                wave_warn.append(msg)

            if wave_warn:
                warnings[wave_name] = wave_warn

        if not metrics:
            logger.warning("No Wave produced metrics – check benchmark/ticker configuration.")
            metrics_df = pd.DataFrame(
                columns=[
                    "Wave",
                    "Return 60D",
                    "Alpha 60D",
                    "Return 1Y",
                    "Alpha 1Y",
                    "# Holdings Used",
                    "# Holdings Defined",
                ]
            )
            return metrics_df, warnings

        metrics_df = pd.DataFrame(
            [
                {
                    "Wave": m.wave,
                    "Return 60D": m.ret_60d,
                    "Alpha 60D": m.alpha_60d,
                    "Return 1Y": m.ret_1y,
                    "Alpha 1Y": m.alpha_1y,
                    "# Holdings Used": m.n_holdings_used,
                    "# Holdings Defined": m.n_holdings_defined,
                }
                from waves_engine import WavesEngine, WaveMetrics

                compute_all_metrics(): Tuple[pd.DataFrame, Dict[str, List[str]]]
                """
                Returns:
                    metrics_df: DataFrame with one row per Wave.
                    warnings:   dict[wave -> list of warning strings]
                """
                logger.info("Computing all Wave metrics (Safe Mode)...")

                self.price_history = self._download_price_history(self.all_tickers)

                metrics: List[WaveMetrics] = []
                warnings: Dict[str, List[str]] = {}

                for wave_name in sorted(self.wave_weights["wave"].unique()):
                    wave_warn: List[str] = []
                    try:
                        wm = self._compute_metrics_for_wave(wave_name, wave_warn)
                        if wm is not None:
                            metrics.append(wm)
                    except Exception as exc:  # Safe mode: never kill entire engine
                        msg = f"Wave '{wave_name}' failed entirely: {exc}"
                        logger.exception(msg)
                        wave_warn.append(msg)

                    if wave_warn:
                        warnings[wave_name] = wave_warn

                if not metrics:
                    logger.warning("No Wave produced metrics – check benchmark/ticker configuration.")
                    metrics_df = pd.DataFrame(
                        columns=[
                            "Wave",
                            "Return 60D",
                            "Alpha 60D",
                            "Return 1Y",
                            "Alpha 1Y",
                            "# Holdings Used",
                            "# Holdings Defined",
                        ]
                    )
                    return metrics_df, warnings

                metrics_df = pd.DataFrame(
                    [
                        {
                            "Wave": m.wave,
                            "Return 60D": m.ret_60d,
                            "Alpha 60D": m.alpha_60d,
                            "Return 1Y": m.ret_1y,
                            "Alpha 1Y": m.alpha_1y,
                            "# Holdings Used": m.n_holdings_used,
                            "# Holdings Defined": m.n_holdings_defined,
                        }
                        for m in metrics
                    ]
                )

                metrics_df = metrics_df.sort_values("Wave").reset_index(drop=True)
                logger.info("Finished computing Wave metrics.")
                return metrics_df, warnings

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------
        def _load_wave_weights(self) -> pd.DataFrame:
            df = pd.read_csv(self.weights_path)

            required_cols = {"wave", "ticker", "weight"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"wave_weights.csv missing columns: {missing}")

            df["wave"] = df["wave"].astype(str).str.strip()
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
            df["weight"] = df["weight"].astype(float)

            # Drop rows with zero weight, just in case
            df = df[df["weight"] > 0].copy()

            return df

        def _download_price_history(self, tickers: List[str]) -> pd.DataFrame:
            logger.info("Downloading price history for %d tickers...", len(tickers))
            if not tickers:
                raise ValueError("No tickers provided to download_price_history")

            unique = sorted(set(tickers))

            data = yf.download(
                unique,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )

            # Handle single vs multi-ticker shapes
            if isinstance(data, pd.Series):
                prices = data.to_frame(name=unique[0])
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    # Use 'Adj Close' if present; otherwise 'Close'
                    level_1 = set(data.columns.get_level_values(0))
                    if "Adj Close" in level_1:
                        prices = data["Adj Close"].copy()
                    elif "Close" in level_1:
                        prices = data["Close"].copy()
                    else:
                        prices = data.copy()
                else:
                    prices = data.copy()

            prices = prices.sort_index()
            prices = prices.ffill().bfill()

            logger.info("Price history shape: %s", prices.shape)
            return prices

        def _build_benchmark_map(self) -> Dict[str, Dict[str, float]]:
            """
            Benchmark blends per Wave.

            These are based on the Google AI recommendations you provided.
            Adjust here if you ever want to change benchmark construction.
            """
            benchmarks: Dict[str, Dict[str, float]] = {
                # S&P Wave: pure S&P 500 exposure
                "S&P Wave": {"SPY": 1.0},

                # Core large-cap growth
                "Growth Wave": {
                    "QQQ": 0.40,
                    "IWF": 0.60,  # Russell 1000 Growth
                },

                # Small-Mid Cap Growth
                "Small-Mid Cap Growth Wave": {
                    "CSMD": 0.50,   # or IVOG/MDYG equivalent
                    "IWO": 0.30,    # small-cap growth tilt (fallback)
                    "SPY": 0.20,
                },

                # Clean Transit-Infrastructure Wave
                # Industrials + Consumer Discretionary + broad market
                "Clean Transit-Infrastructure Wave": {
                    "FIDU": 0.45,
                    "FDIS": 0.45,
                    "SPY": 0.10,
                },

                # Cloud & Enterprise Software Growth Wave
                "Cloud & Enterprise Software Growth Wave": {
                    "IGV": 0.60,
                    "WCLD": 0.20,  # CLOU also acceptable
                    "SPY": 0.20,
                },

                # Crypto Equity Wave (mid/large cap)
                "Crypto Equity Wave (mid/large cap)": {
                    "FBTC": 0.70,  # or IBIT
                    "DAPP": 0.30,
                },

                # Income Wave
                "Income Wave": {
                    "SCHD": 0.70,
                    "SPY": 0.30,
                },

                # Quantum Computing Wave – specialist index preferred; use QQQ fallback
                "Quantum Computing Wave": {
                    "QQQ": 0.60,
                    "IYW": 0.20,
                    "SPY": 0.20,
                },

                # AI Wave – diversified software & cloud benchmarks
                "AI Wave": {
                    "IGV": 0.50,
                    "VGT": 0.25,
                    "QQQ": 0.15,
                    "SPY": 0.10,
                },

                # SmartSafe Wave – ultra-short government & T-bill ladder
                "SmartSafe Wave": {
                    "SGOV": 0.70,
                    "BIL": 0.20,
                    "SHY": 0.10,
                },
            }

            # Normalize just in case any rounding makes them slightly off
            normed: Dict[str, Dict[str, float]] = {}
            for wave, comp in benchmarks.items():
                total = float(sum(comp.values()))
                if total <= 0:
                    continue
                normed[wave] = {t: w / total for t, w in comp.items()}

            return normed

        def _compute_metrics_for_wave(
            self, wave_name: str, warnings: List[str]
        ) -> Optional[WaveMetrics]:
            """
            Compute metrics for a single Wave in Safe Mode.
            Returns None if absolutely nothing can be computed.
            """
            if self.price_history is None:
                raise RuntimeError("Price history has not been downloaded yet.")

            # Slice weights for this wave
            wdf = self.wave_weights[self.wave_weights["wave"] == wave_name].copy()
            if wdf.empty:
                warnings.append(f"No weights found for Wave '{wave_name}'.")
                return None

            defined_tickers = wdf["ticker"].tolist()
            weights = wdf["weight"].values.astype(float)

            # Restrict to tickers that actually have price data
            available_cols = [t for t in defined_tickers if t in self.price_history.columns]

            if not available_cols:
                warnings.append(
                    f"No price data available for any tickers in Wave '{wave_name}'. "
                    f"Defined tickers: {defined_tickers}"
                )
                return None

            missing = sorted(set(defined_tickers) - set(available_cols))
            if missing:
                warnings.append(
                    f"Dropping {len(missing)} tickers with no price data in Wave "
                    f"'{wave_name}': {missing}"
                )

            # Align weights to available tickers
            mask = [t in available_cols for t in defined_tickers]
            used_weights = weights[mask].astype(float)
            used_tickers = [t for t in defined_tickers if t in available_cols]

            if used_weights.sum() <= 0:
                warnings.append(
                    f"Non-positive total weight after filtering for Wave '{wave_name}'."
                )
                return None

            used_weights = used_weights / used_weights.sum()

            # --- Wave returns ---
            price_slice = self.price_history[used_tickers]
            ret_slice = price_slice.pct_change().dropna(how="all")
            # rows that are fully NaN -> drop
            ret_slice = ret_slice.dropna(axis=0, how="any")

            if ret_slice.empty:
                warnings.append(f"No return data after pct_change for Wave '{wave_name}'.")
                return None

            wave_ret = (ret_slice * used_weights).sum(axis=1)

            # --- Benchmark returns ---
            bench_comp = self.benchmarks.get(wave_name)
            if not bench_comp:
                # Fallback: use SPY if available, otherwise Wave vs itself (alpha=0)
                if "SPY" in self.price_history.columns:
                    bench_comp = {"SPY": 1.0}
                    warnings.append(
                        f"No explicit benchmark blend defined for '{wave_name}'. "
                        f"Using SPY as fallback."
                    )
                else:
                    warnings.append(
                        f"No benchmark blend or SPY price for '{wave_name}'. "
                        f"Alpha will be zero (self-benchmark)."
                    )
                    bench_ret = wave_ret.copy()
                    return self._build_wave_metrics(
                        wave_name,
                        wave_ret,
                        bench_ret,
                        len(used_tickers),
                        len(defined_tickers),
                        warnings,
                    )

            bench_tickers = list(bench_comp.keys())
            bench_weights = np.array(list(bench_comp.values()), dtype=float)

            bench_available = [t for t in bench_tickers if t in self.price_history.columns]
            if not bench_available:
                warnings.append(
                    f"No benchmark tickers available with price data for '{wave_name}'. "
                    f"Using Wave itself as benchmark (alpha=0)."
                )
                bench_ret = wave_ret.copy()
            else:
                missing_bench = sorted(set(bench_tickers) - set(bench_available))
                if missing_bench:
                    warnings.append(
                        f"Dropping benchmark tickers with no data for '{wave_name}': "
                        f"{missing_bench}"
                    )

                mask_b = [t in bench_available for t in bench_tickers]
                bench_weights_used = bench_weights[mask_b]
                if bench_weights_used.sum() <= 0:
                    warnings.append(
                        f"Benchmark weights collapsed for '{wave_name}'. "
                        f"Using equal weights for available benchmark tickers."
                    )
                    bench_weights_used = np.ones(len(bench_available), dtype=float)

                bench_weights_used = bench_weights_used / bench_weights_used.sum()

                bench_prices = self.price_history[bench_available]
                bench_rets_raw = bench_prices.pct_change().dropna(how="any", axis=0)
                if bench_rets_raw.empty:
                    warnings.append(
                        f"No benchmark return data after pct_change for '{wave_name}'. "
                        f"Using Wave itself as benchmark."
                    )
                    bench_ret = wave_ret.copy()
                else:
                    bench_ret = (bench_rets_raw * bench_weights_used).sum(axis=1)

            # Align index
            both = pd.concat(
                [wave_ret.rename("wave"), bench_ret.rename("bench")],
                axis=1,
                join="inner",
            ).dropna()

            if both.empty:
                warnings.append(
                    f"No overlapping dates between Wave and benchmark for '{wave_name}'."
                )
                return None

            wave_ret = both["wave"]
            bench_ret = both["bench"]

            return self._build_wave_metrics(
                wave_name,
                wave_ret,
                bench_ret,
                len(used_tickers),
                len(defined_tickers),
                warnings,
            )

        # ------------------------------------------------------------------
        def _build_wave_metrics(
            self,
            wave_name: str,
            wave_ret: pd.Series,
            bench_ret: pd.Series,
            n_used: int,
            n_defined: int,
            warnings: List[str],
        ) -> WaveMetrics:
            def window_total(series: pd.Series, days: int) -> float:
                if series.empty:
                    return np.nan
                if len(series) <= days:
                    window = series
                else:
                    window = series.iloc[-days:]
                return float((1.0 + window).prod() - 1.0)

            ret_60 = window_total(wave_ret, 60)
            bench_60 = window_total(bench_ret, 60)
            ret_1y = window_total(wave_ret, 252)
            bench_1y = window_total(bench_ret, 252)

            alpha_60 = ret_60 - bench_60 if not np.isnan(ret_60) and not np.isnan(bench_60) else np.nan
            alpha_1y = ret_1y - bench_1y if not np.isnan(ret_1y) and not np.isnan(bench_1y) else np.nan

            return WaveMetrics(
                wave=wave_name,
                ret_60d=ret_60,
                alpha_60d=alpha_60,
                ret_1y=ret_1y,
                alpha_1y=alpha_1y,
                n_holdings_used=n_used,
                n_holdings_defined=n_defined,
                warnings=list(warnings),
            )