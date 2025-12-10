# waves_engine.py

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta


DATA_ROOT = Path(".")


@dataclass
class WaveMetrics:
    wave: str
    mode: str
    intraday_return: float
    intraday_alpha: float
    r30_return: float
    r30_alpha: float
    r60_return: float
    r60_alpha: float
    r1y_return: float
    r1y_alpha: float
    beta: float
    benchmark_name: str


class WavesEngine:
    """
    WAVES Intelligence™ Engine
    - Loads list.csv (universe) and wave_weights.csv
    - Builds Waves from weights
    - Fetches daily prices with yfinance
    - Computes Wave vs benchmark returns + alpha (intraday, 30D, 60D, 1Y)
    """

    def __init__(
        self,
        list_path: Union[str, Path] = DATA_ROOT / "list.csv",
        weights_path: Union[str, Path] = DATA_ROOT / "wave_weights.csv",
        lookback_days: int = 365,
        mode: str = "standard",
    ):
        self.list_path = Path(list_path)
        self.weights_path = Path(weights_path)
        self.lookback_days = lookback_days
        self.mode = mode  # "standard", "alpha_minus_beta", "private_logic"
        self.universe_df = self._load_universe()
        self.weights_df = self._load_weights()
        self.waves = sorted(self.weights_df["wave"].unique())

        # price history shared across all waves/benchmarks
        self.price_history: Optional[pd.DataFrame] = None
        self.metrics: List[WaveMetrics] = []

    # ------------------------------
    # Loading & setup
    # ------------------------------

    def _load_universe(self) -> pd.DataFrame:
        if not self.list_path.exists():
            raise FileNotFoundError(f"list.csv not found at {self.list_path}")

        df = pd.read_csv(self.list_path)

        # Normalize column names
        cols = {c.lower().strip(): c for c in df.columns}
        if "ticker" not in cols:
            raise ValueError("list.csv must have a 'Ticker' column")
        if "ticker" != cols["ticker"]:
            df.rename(columns={cols["ticker"]: "Ticker"}, inplace=True)

        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df = df.drop_duplicates(subset=["Ticker"])
        return df

    def _load_weights(self) -> pd.DataFrame:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"wave_weights.csv not found at {self.weights_path}")

        df = pd.read_csv(self.weights_path)

        # Normalize column names to wave, ticker, weight
        rename_map = {}
        for c in df.columns:
            lc = c.lower().strip()
            if lc in ("wave", "name"):
                rename_map[c] = "wave"
            elif lc in ("ticker", "symbol"):
                rename_map[c] = "ticker"
            elif lc in ("weight", "w"):
                rename_map[c] = "weight"
        df = df.rename(columns=rename_map)

        required = {"wave", "ticker", "weight"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"wave_weights.csv must have columns {required}, has {set(df.columns)}"
            )

        df["wave"] = df["wave"].astype(str).str.strip()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

        # Merge with universe for sanity (optional metadata)
        df = df.merge(
            self.universe_df[["Ticker"]],
            left_on="ticker",
            right_on="Ticker",
            how="left",
            indicator=True,
        )
        # Keep only tickers present in list.csv
        df = df[df["_merge"] != "left_only"].drop(columns=["Ticker", "_merge"])

        # Deduplicate and normalize per wave
        df = (
            df.groupby(["wave", "ticker"], as_index=False)["weight"]
            .sum()
            .reset_index(drop=True)
        )

        def _normalize(group: pd.DataFrame) -> pd.DataFrame:
            w_sum = group["weight"].sum()
            if w_sum <= 0:
                group["weight"] = 0.0
            else:
                group["weight"] = group["weight"] / w_sum
            return group

        df = df.groupby("wave", group_keys=False).apply(_normalize)
        return df

    # ------------------------------
    # Benchmark mapping
    # ------------------------------

    def get_benchmark(
        self, wave: str
    ) -> Union[str, Dict[str, float]]:  # ticker or blend
        """
        Returns either a single ETF ticker or a dict of {ticker: weight}.
        All weights MUST sum to 1.0 when dict.
        """
        w = wave.strip()

        # --- Crypto Equity Wave (mid/large cap) ---
        # 70% FBTC (spot BTC proxy), 30% DAPP (digital assets equity)
        if w in ("Crypto Equity Wave (mid/large cap)", "Crypto Equity Wave", "Crypto Income Wave"):
            return {"FBTC": 0.70, "DAPP": 0.30}

        # --- Future Power & Energy Wave ---
        # 65% XLE (Energy), 25% XLU (Utilities), 10% SPY (broad)
        if w in ("Future Power & Energy Wave", "Future Power & Energy"):
            return {"XLE": 0.65, "XLU": 0.25, "SPY": 0.10}

        # --- Clean Transit-Infrastructure Wave ---
        # 45% FIDU (Industrials), 45% FDIS (Consumer Disc), 10% SPY
        if w == "Clean Transit-Infrastructure Wave":
            return {"FIDU": 0.45, "FDIS": 0.45, "SPY": 0.10}

        # --- Cloud & Enterprise Software Growth (formerly Small Cap Growth) ---
        # 60% IGV, 20% WCLD, 20% SPY
        if w in (
            "Cloud & Enterprise Software Growth Wave",
            "Cloud Computing & Enterprise Software Growth Fund",
            "Small Cap Growth Wave",
        ):
            return {"IGV": 0.60, "WCLD": 0.20, "SPY": 0.20}

        # --- S&P Wave ---
        if w == "S&P Wave":
            return "SPY"

        # --- Growth Wave (large-cap growth) ---
        # 50% QQQ, 50% IWF
        if w == "Growth Wave":
            return {"QQQ": 0.50, "IWF": 0.50}

        # --- Small-Mid Cap Growth Wave ---
        # 50% VTWG (small-cap growth), 50% VO (mid-cap)
        if w in ("Small-Mid Cap Growth Wave", "Small to Mid Cap Growth Wave"):
            return {"VTWG": 0.50, "VO": 0.50}

        # --- Income Wave ---
        # SCHD as core proxy
        if w == "Income Wave":
            return "SCHD"

        # --- AI Wave ---
        # 40% QQQ, 60% SOXX
        if w == "AI Wave":
            return {"QQQ": 0.40, "SOXX": 0.60}

        # --- Quantum Computing Wave ---
        # 70% IYW (tech), 30% SOXX (semi)
        if w == "Quantum Computing Wave":
            return {"IYW": 0.70, "SOXX": 0.30}

        # Fallback: broad S&P
        return "SPY"

    def _benchmark_label(self, bench: Union[str, Dict[str, float]]) -> str:
        if isinstance(bench, str):
            return bench
        parts = [f"{t}:{w:.0%}" for t, w in bench.items()]
        return " + ".join(parts)

    # ------------------------------
    # Price history & returns
    # ------------------------------

    def _all_needed_tickers(self) -> List[str]:
        tickers = set(self.weights_df["ticker"].unique().tolist())

        for w in self.waves:
            b = self.get_benchmark(w)
            if isinstance(b, str):
                tickers.add(b)
            else:
                tickers.update(b.keys())
        return sorted(tickers)

    def load_price_history(self) -> pd.DataFrame:
        if self.price_history is not None:
            return self.price_history

        tickers = self._all_needed_tickers()
        if not tickers:
            raise ValueError("No tickers found to fetch.")

        data = yf.download(
            tickers,
            period=f"{self.lookback_days}d",
            auto_adjust=True,
            progress=False,
        )

        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            prices = data["Adj Close"].copy()
        else:
            prices = data.copy()

        # Normalize: if single ticker, yfinance returns Series
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        # Clean index & columns
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.dropna(how="all")
        prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

        self.price_history = prices
        return prices

    def _compute_beta(
        self, wave_ret: pd.Series, bench_ret: pd.Series
    ) -> float:
        wave_ret = wave_ret.dropna()
        bench_ret = bench_ret.dropna()
        aligned = pd.concat([wave_ret, bench_ret], axis=1, join="inner").dropna()
        if len(aligned) < 10:
            return 1.0
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var = np.var(aligned.iloc[:, 1])
        if var <= 0:
            return 1.0
        beta = cov / var
        # Keep beta in a reasonable band
        return float(np.clip(beta, 0.2, 2.0))

    def _period_return(self, series: pd.Series, days: int) -> float:
        if series.empty:
            return np.nan
        series = series.dropna()
        if len(series) < 2:
            return np.nan
        if days <= 1:
            # Intraday: last vs previous
            if len(series) < 2:
                return np.nan
            return float(series.iloc[-1] / series.iloc[-2] - 1.0)
        # rolling period in trading days
        if len(series) <= days:
            start = series.iloc[0]
        else:
            start = series.iloc[-(days + 1)]
        end = series.iloc[-1]
        if start <= 0:
            return np.nan
        return float(end / start - 1.0)

    def _build_wave_index(
        self, wave: str, prices: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, float, str]:
        """
        Returns:
            wave_idx, bench_idx, beta, bench_label
        """
        w_rows = self.weights_df[self.weights_df["wave"] == wave].copy()
        if w_rows.empty:
            raise ValueError(f"No weights found for Wave '{wave}'")

        tickers = [t for t in w_rows["ticker"].tolist() if t in prices.columns]
        if not tickers:
            raise ValueError(f"No price data for any tickers in Wave '{wave}'")

        sub = prices[tickers].copy()
        # Ensure no duplicate columns
        sub = sub.loc[:, ~sub.columns.duplicated(keep="first")]

        weights = (
            w_rows.set_index("ticker")["weight"]
            .reindex(sub.columns)
            .fillna(0.0)
        )
        if weights.sum() <= 0:
            weights[:] = 1.0 / len(weights)
        else:
            weights = weights / weights.sum()

        wave_ret = sub.pct_change().dropna()
        port_ret = (wave_ret * weights.values).sum(axis=1)

        wave_idx = (1.0 + port_ret).cumprod()

        bench = self.get_benchmark(wave)
        bench_label = self._benchmark_label(bench)

        if isinstance(bench, str):
            if bench not in prices.columns:
                raise ValueError(f"No price data for benchmark {bench} for Wave '{wave}'")
            bench_prices = prices[[bench]].copy()
            bench_ret = bench_prices.pct_change().dropna().iloc[:, 0]
        else:
            b_tickers = [t for t in bench.keys() if t in prices.columns]
            if not b_tickers:
                raise ValueError(
                    f"No price data for any benchmark tickers in Wave '{wave}'"
                )
            b_sub = prices[b_tickers].copy()
            b_sub = b_sub.loc[:, ~b_sub.columns.duplicated(keep="first")]
            b_weights = (
                pd.Series(bench)
                .reindex(b_sub.columns)
                .fillna(0.0)
            )
            if b_weights.sum() <= 0:
                b_weights[:] = 1.0 / len(b_weights)
            else:
                b_weights = b_weights / b_weights.sum()
            b_ret = b_sub.pct_change().dropna()
            bench_ret = (b_ret * b_weights.values).sum(axis=1)

        bench_idx = (1.0 + bench_ret).cumprod()
        beta = self._compute_beta(port_ret, bench_ret)

        # align indices
        aligned = pd.concat([wave_idx, bench_idx], axis=1, join="inner").dropna()
        aligned.columns = ["wave", "bench"]
        return aligned["wave"], aligned["bench"], beta, bench_label

    # ------------------------------
    # Public API
    # ------------------------------

    def compute_all_metrics(self) -> List[WaveMetrics]:
        prices = self.load_price_history()
        results: List[WaveMetrics] = []

        for w in self.waves:
            try:
                wave_idx, bench_idx, beta, bench_label = self._build_wave_index(w, prices)

                # Period returns
                intraday_wave = self._period_return(wave_idx, 1)
                intraday_bench = self._period_return(bench_idx, 1)

                r30_wave = self._period_return(wave_idx, 30)
                r30_bench = self._period_return(bench_idx, 30)

                r60_wave = self._period_return(wave_idx, 60)
                r60_bench = self._period_return(bench_idx, 60)

                r1y_wave = self._period_return(wave_idx, 252)
                r1y_bench = self._period_return(bench_idx, 252)

                # Alpha = wave - beta * benchmark
                intraday_alpha = intraday_wave - beta * intraday_bench
                r30_alpha = r30_wave - beta * r30_bench
                r60_alpha = r60_wave - beta * r60_bench
                r1y_alpha = r1y_wave - beta * r1y_bench

                metrics = WaveMetrics(
                    wave=w,
                    mode=self.mode,
                    intraday_return=intraday_wave,
                    intraday_alpha=intraday_alpha,
                    r30_return=r30_wave,
                    r30_alpha=r30_alpha,
                    r60_return=r60_wave,
                    r60_alpha=r60_alpha,
                    r1y_return=r1y_wave,
                    r1y_alpha=r1y_alpha,
                    beta=beta,
                    benchmark_name=bench_label,
                )
                results.append(metrics)
            except Exception as e:
                # On failure, store NaNs but keep Wave visible
                metrics = WaveMetrics(
                    wave=w,
                    mode=self.mode,
                    intraday_return=np.nan,
                    intraday_alpha=np.nan,
                    r30_return=np.nan,
                    r30_alpha=np.nan,
                    r60_return=np.nan,
                    r60_alpha=np.nan,
                    r1y_return=np.nan,
                    r1y_alpha=np.nan,
                    beta=np.nan,
                    benchmark_name=f"ERROR: {e}",
                )
                results.append(metrics)

        self.metrics = results
        return results

    def metrics_dataframe(self) -> pd.DataFrame:
        if not self.metrics:
            self.compute_all_metrics()

        rows = []
        for m in self.metrics:
            rows.append(
                {
                    "Wave": m.wave,
                    "Mode": m.mode,
                    "Benchmark": m.benchmark_name,
                    "Beta (≈)": m.beta,
                    "Intraday Return (%)": m.intraday_return * 100 if pd.notna(m.intraday_return) else np.nan,
                    "Intraday Alpha (%)": m.intraday_alpha * 100 if pd.notna(m.intraday_alpha) else np.nan,
                    "30D Return (%)": m.r30_return * 100 if pd.notna(m.r30_return) else np.nan,
                    "30D Alpha (%)": m.r30_alpha * 100 if pd.notna(m.r30_alpha) else np.nan,
                    "60D Return (%)": m.r60_return * 100 if pd.notna(m.r60_return) else np.nan,
                    "60D Alpha (%)": m.r60_alpha * 100 if pd.notna(m.r60_alpha) else np.nan,
                    "1Y Return (%)": m.r1y_return * 100 if pd.notna(m.r1y_return) else np.nan,
                    "1Y Alpha (%)": m.r1y_alpha * 100 if pd.notna(m.r1y_alpha) else np.nan,
                }
            )
        df = pd.DataFrame(rows)
        df = df.sort_values("Wave").reset_index(drop=True)
        return df

    def wave_history(self, wave: str) -> Optional[pd.DataFrame]:
        if self.price_history is None:
            self.load_price_history()
        prices = self.price_history
        try:
            wave_idx, bench_idx, beta, bench_label = self._build_wave_index(wave, prices)
        except Exception:
            return None
        df = pd.DataFrame(
            {"Wave Index": wave_idx, "Benchmark Index": bench_idx}
        )
        return df