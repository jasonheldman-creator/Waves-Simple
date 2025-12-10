# waves_engine.py

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class WaveMetrics:
    wave: str
    intraday_alpha: float
    alpha_30d: float
    alpha_60d: float
    alpha_1y: float
    port_1y_return: float
    bench_1y_return: float


class WavesEngine:
    """
    Core WAVES Intelligence™ engine.

    - Reads wave_weights.csv
    - Downloads prices for all tickers
    - Builds benchmark blends per Wave
    - Computes alpha over multiple windows
    - Exposes diagnostics for the console UI
    """

    def __init__(
        self,
        weights_path: str = "wave_weights.csv",
        lookback_years: int = 5,
    ) -> None:
        self.weights_path = weights_path
        self.lookback_years = lookback_years

        self.wave_weights: Dict[str, Dict[str, float]] = {}
        self.waves: List[str] = []
        self.all_tickers: List[str] = []

        self.benchmark_specs: Dict[str, Dict[str, float]] = self._build_benchmark_specs()

        self.price_data: Optional[pd.DataFrame] = None
        self.bench_price_data: Optional[pd.DataFrame] = None

        self.metrics: Dict[str, WaveMetrics] = {}
        self.diagnostics: List[str] = []
        self.missing_tickers: List[str] = []

        self._load_weights()
        self._download_all_prices()
        self._compute_all_metrics()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        df = pd.read_csv(self.weights_path)

        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        expected = {"wave", "ticker", "weight"}
        if not expected.issubset(set(df.columns)):
            raise RuntimeError(
                f"wave_weights.csv must have columns {expected}, found {set(df.columns)}"
            )

        df["wave"] = df["wave"].astype(str).str.strip()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

        df = df.dropna(subset=["weight"])

        waves = sorted(df["wave"].unique().tolist())
        self.waves = waves

        wave_weights: Dict[str, Dict[str, float]] = {}
        all_tickers = set()

        for wave in waves:
            sub = df[df["wave"] == wave]
            # If duplicate tickers exist per wave, sum and renormalise
            grouped = sub.groupby("ticker")["weight"].sum()
            total = grouped.sum()
            if total <= 0:
                self.diagnostics.append(
                    f"[WARN] Wave '{wave}' has non-positive total weight ({total}); skipping."
                )
                continue
            normed = grouped / total
            wave_weights[wave] = normed.to_dict()
            all_tickers.update(normed.index.tolist())

        if not wave_weights:
            raise RuntimeError("No usable waves found in wave_weights.csv.")

        self.wave_weights = wave_weights
        self.all_tickers = sorted(all_tickers)

    def _build_benchmark_specs(self) -> Dict[str, Dict[str, float]]:
        """
        Benchmark blends per Wave using the ETF mixes we discussed.
        We normalise weights to 1.0 internally.
        """
        specs: Dict[str, Dict[str, float]] = {
            # 1) Broad market
            "S&P Wave": {"SPY": 1.0},
            # 2) Large-cap growth / tech
            #    (You asked for 50% QQQ and 60% IWF; we keep ratios but renormalise.)
            "Growth Wave": {"QQQ": 0.5, "IWF": 0.6},
            # 3) Small-Mid cap growth
            #    Reasonable small/mid growth proxies.
            "Small-Mid Cap Growth Wave": {"VTWG": 0.6, "IWO": 0.4},
            "Small-Mid Cap Growth Wave": {"VTWG": 0.6, "IWO": 0.4},  # handle both spellings

            # 4) Clean Transit-Infrastructure
            #    Industrials + Consumer Discretionary + SPY (auto & industrial blend).
            "Clean Transit-Infrastructure Wave": {"FIDU": 0.45, "FDIS": 0.45, "SPY": 0.10},
            "Clean Transit-Infrastructure Wave": {"FIDU": 0.45, "FDIS": 0.45, "SPY": 0.10},

            # 5) Cloud & Enterprise Software Growth
            #    IGV 60%, WCLD/CLOU 20%, SPY 20%
            "Cloud & Enterprise Software Growth Wave": {
                "IGV": 0.6,
                "WCLD": 0.2,
                "SPY": 0.2,
            },

            # 6) Crypto Equity Wave (mid/large cap)
            #    70% spot BTC proxy, 30% digital assets equity index
            "Crypto Equity Wave (mid/large cap)": {"FBTC": 0.7, "DAPP": 0.3},

            # 7) Income Wave
            "Income Wave": {"SCHD": 1.0},

            # 8) Quantum Computing Wave
            #    Use a growthy tech blend as proxy
            "Quantum Computing Wave": {"QQQ": 0.5, "IGV": 0.25, "IYW": 0.25},

            # 9) AI Wave
            #    IGV 50%, VGT 25%, QQQ 15%, SPY 10%
            "AI Wave": {"IGV": 0.5, "VGT": 0.25, "QQQ": 0.15, "SPY": 0.10},

            # 10) SmartSafe Wave
            #     Same as holdings: 70% SGOV, 20% BIL, 10% SHY
            "SmartSafe Wave": {"SGOV": 0.7, "BIL": 0.2, "SHY": 0.1},
        }

        # Normalise all weights
        normalised: Dict[str, Dict[str, float]] = {}
        for wave, comp in specs.items():
            total = float(sum(comp.values()))
            if total <= 0:
                continue
            normalised[wave] = {t: w / total for t, w in comp.items()}
        return normalised

    def _download_prices(self, tickers: List[str]) -> pd.DataFrame:
        if not tickers:
            raise RuntimeError("No tickers provided for price download.")

        end = dt.date.today()
        start = end - dt.timedelta(days=365 * self.lookback_years)

        data = yf.download(
            tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )

        # yfinance returns different shapes for 1 vs many tickers
        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            # Single ticker case
            prices = data["Adj Close"].to_frame()
            prices.columns = tickers
        else:
            # Multi-ticker: columns are MultiIndex (ticker, field)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data.xs("Adj Close", axis=1, level=1)
            else:
                prices = data.copy()

        # Drop entirely empty columns
        all_nan = prices.isna().all()
        bad_cols = all_nan[all_nan].index.tolist()
        if bad_cols:
            self.diagnostics.append(
                f"[WARN] Dropping {len(bad_cols)} tickers with no price data: {bad_cols}"
            )
            self.missing_tickers.extend([str(c) for c in bad_cols])
            prices = prices.loc[:, ~all_nan]

        if prices.empty:
            raise RuntimeError(
                f"No price data found for any of tickers: {tickers}"
            )

        prices = prices.sort_index()
        return prices

    def _download_all_prices(self) -> None:
        # 1) Portfolio holdings
        self.price_data = self._download_prices(self.all_tickers)

        # 2) Benchmark tickers
        bench_tickers = set()
        for comp in self.benchmark_specs.values():
            bench_tickers.update(comp.keys())
        self.bench_price_data = self._download_prices(sorted(bench_tickers))

    # ------------------------------------------------------------------
    # Return helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_returns(price_df: pd.DataFrame) -> pd.DataFrame:
        rets = price_df.pct_change().dropna(how="all")
        return rets

    def _wave_return_series(self, wave: str) -> pd.Series:
        assert self.price_data is not None
        weights = self.wave_weights[wave]
        tickers = [t for t in weights.keys() if t in self.price_data.columns]

        if not tickers:
            raise RuntimeError(f"No valid tickers for wave '{wave}' after cleaning.")

        sub_prices = self.price_data[tickers]
        sub_rets = self._to_returns(sub_prices)

        # Re-normalize weights over tickers that actually have data
        w_vec = np.array([weights[t] for t in tickers], dtype=float)
        w_vec = w_vec / w_vec.sum()

        port_rets = sub_rets.dot(w_vec)
        port_rets.name = wave
        return port_rets

    def _benchmark_return_series(self, wave: str) -> pd.Series:
        assert self.bench_price_data is not None
        spec = None

        # Try exact, then some fallbacks for special characters
        if wave in self.benchmark_specs:
            spec = self.benchmark_specs[wave]
        elif wave.replace("–", "-") in self.benchmark_specs:
            spec = self.benchmark_specs[wave.replace("–", "-")]
        elif wave.replace("-", "-") in self.benchmark_specs:  # hyphen variations
            spec = self.benchmark_specs[wave.replace("-", "-")]

        if spec is None:
            # Default to SPY if no spec
            self.diagnostics.append(
                f"[INFO] No specific benchmark blend for '{wave}', defaulting to SPY."
            )
            spec = {"SPY": 1.0}

        tickers = [t for t in spec.keys() if t in self.bench_price_data.columns]
        if not tickers:
            raise RuntimeError(
                f"No valid benchmark tickers for wave '{wave}' (spec={spec})."
            )

        sub_prices = self.bench_price_data[tickers]
        sub_rets = self._to_returns(sub_prices)

        w_vec = np.array([spec[t] for t in tickers], dtype=float)
        w_vec = w_vec / w_vec.sum()

        bench_rets = sub_rets.dot(w_vec)
        bench_rets.name = f"{wave} Benchmark"
        return bench_rets

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _window_alpha(
        port: pd.Series,
        bench: pd.Series,
        n_days: int,
    ) -> Tuple[float, float, float]:
        """
        Returns:
            (alpha, port_return, bench_return) over last n_days.
        """
        if port.empty or bench.empty:
            return np.nan, np.nan, np.nan

        # Align
        joined = pd.concat([port, bench], axis=1, join="inner").dropna()
        if joined.empty:
            return np.nan, np.nan, np.nan

        if n_days > 0 and len(joined) > n_days:
            window = joined.iloc[-n_days:]
        else:
            window = joined

        p = window.iloc[:, 0]
        b = window.iloc[:, 1]

        def cum_ret(r: pd.Series) -> float:
            return float((1.0 + r).prod() - 1.0)

        p_ret = cum_ret(p)
        b_ret = cum_ret(b)
        alpha = p_ret - b_ret
        return alpha, p_ret, b_ret

    def _compute_all_metrics(self) -> None:
        self.metrics = {}

        for wave in self.waves:
            if wave not in self.wave_weights:
                continue

            try:
                port_rets = self._wave_return_series(wave)
                bench_rets = self._benchmark_return_series(wave)

                # Align
                joined = pd.concat([port_rets, bench_rets], axis=1, join="inner").dropna()
                if joined.empty:
                    raise RuntimeError(f"No overlapping dates for '{wave}' vs benchmark.")

                p = joined.iloc[:, 0]
                b = joined.iloc[:, 1]

                # Intraday alpha = last day's excess return
                intraday_alpha = float(p.iloc[-1] - b.iloc[-1])

                alpha_30d, _, _ = self._window_alpha(p, b, n_days=30)
                alpha_60d, _, _ = self._window_alpha(p, b, n_days=60)
                alpha_1y, p_1y, b_1y = self._window_alpha(p, b, n_days=252)

                self.metrics[wave] = WaveMetrics(
                    wave=wave,
                    intraday_alpha=intraday_alpha,
                    alpha_30d=alpha_30d,
                    alpha_60d=alpha_60d,
                    alpha_1y=alpha_1y,
                    port_1y_return=p_1y,
                    bench_1y_return=b_1y,
                )

            except Exception as e:
                self.diagnostics.append(
                    f"[ERROR] Failed to compute metrics for '{wave}': {e}"
                )

        if not self.metrics:
            raise RuntimeError(
                "Engine failed to compute metrics for all Waves. "
                "Check diagnostics and benchmark specs."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_metrics_df(self) -> pd.DataFrame:
        records = []
        for w, m in self.metrics.items():
            records.append(
                {
                    "Wave": m.wave,
                    "Intraday Alpha": m.intraday_alpha,
                    "Alpha 30D": m.alpha_30d,
                    "Alpha 60D": m.alpha_60d,
                    "Alpha 1Y": m.alpha_1y,
                    "Wave 1Y Return": m.port_1y_return,
                    "Benchmark 1Y Return": m.bench_1y_return,
                }
            )
        df = pd.DataFrame.from_records(records)
        df = df.set_index("Wave").sort_index()
        return df

    def wave_series_pair(self, wave: str) -> Tuple[pd.Series, pd.Series]:
        """
        For Wave Explorer: returns (portfolio_return_series, benchmark_return_series)
        aligned to the same date index.
        """
        port = self._wave_return_series(wave)
        bench = self._benchmark_return_series(wave)
        joined = pd.concat([port, bench], axis=1, join="inner").dropna()
        return joined.iloc[:, 0], joined.iloc[:, 1]