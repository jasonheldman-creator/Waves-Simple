# waves_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class WaveMetrics:
    name: str
    ret_60d: float
    ret_1y: float
    alpha_1y: float


class WavesEngine:
    """
    Minimal, robust engine:
      - Reads wave_weights.csv (wave, ticker, weight)
      - Downloads prices for all tickers + benchmark ETFs
      - Computes 60D & 1Y returns and 1Y alpha vs benchmark blend
    """

    def __init__(self, weights_path: str = "wave_weights.csv") -> None:
        self.weights_path = weights_path
        self.wave_weights = self._load_wave_weights()
        self.benchmarks = self._build_benchmark_map()

    # ---------- setup ----------

    def _load_wave_weights(self) -> pd.DataFrame:
        df = pd.read_csv(self.weights_path)
        # normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        # expect columns: wave, ticker, weight
        if not {"wave", "ticker", "weight"}.issubset(set(df.columns)):
            raise RuntimeError(
                "wave_weights.csv must have columns: wave,ticker,weight"
            )
        # clean whitespace
        df["wave"] = df["wave"].astype(str).str.strip()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["weight"] = df["weight"].astype(float)
        return df

    def _build_benchmark_map(self) -> Dict[str, Dict[str, float]]:
        """
        Benchmarks based on your latest instructions.

        Note: weights here are un-normalized; we normalize them in code.
        """
        raw = {
            # Core equities
            "S&P Wave": {"SPY": 1.0},

            # Growth Wave: you suggested 50% QQQ, 60% IWF -> normalized to 45/55
            "Growth Wave": {"QQQ": 0.45, "IWF": 0.55},

            # Small-Mid Cap Growth Wave (your cloud/infrastructure mid-cap basket)
            "Small-Mid Cap Growth Wave": {"IGV": 0.60, "WCLD": 0.20, "SPY": 0.20},

            # Clean Transit-Infrastructure Wave (industrials + autos blend)
            "Clean Transit-Infrastructure Wave": {
                "FIDU": 0.45,
                "FDIS": 0.45,
                "SPY": 0.10,
            },

            # Cloud & Enterprise Software Growth Wave
            "Cloud & Enterprise Software Growth Wave": {
                "IGV": 0.60,
                "WCLD": 0.20,
                "SPY": 0.20,
            },

            # Crypto Equity Wave (mid/large cap): 70% spot BTC proxy, 30% DAPP
            "Crypto Equity Wave (mid/large cap)": {
                "FBTC": 0.70,  # Fidelity spot BTC ETF
                "DAPP": 0.30,
            },

            # Income Wave – simple SCHD proxy
            "Income Wave": {"SCHD": 1.0},

            # Quantum Computing Wave – concentrated growth & robotics
            "Quantum Computing Wave": {"QQQ": 0.50, "BOTZ": 0.50},

            # AI Wave – reusing your IGV/VGT/QQQ/SPY tech blend
            "AI Wave": {"IGV": 0.50, "VGT": 0.25, "QQQ": 0.15, "SPY": 0.10},

            # SmartSafe Wave – short-term Treasuries blend
            "SmartSafe Wave": {"SGOV": 0.70, "BIL": 0.20, "SHY": 0.10},
        }

        # normalize benchmark weights to 1.0
        normed: Dict[str, Dict[str, float]] = {}
        for wave, comp in raw.items():
            total = float(sum(comp.values()))
            if total <= 0:
                continue
            normed[wave] = {t: w / total for t, w in comp.items()}
        return normed

    # ---------- helpers ----------

    @staticmethod
    def _yf_symbol(ticker: str) -> str:
        """
        Convert CSV ticker into yfinance symbol. The big gotcha was BRK.B.
        We simply swap '.' for '-' which works for most cases (e.g. BRK-B).
        """
        return ticker.replace(".", "-")

    def _collect_all_tickers(self) -> Set[str]:
        tickers: Set[str] = set(self.wave_weights["ticker"].unique())
        for comp in self.benchmarks.values():
            tickers.update(comp.keys())
        return tickers

    def _download_price_data(self) -> pd.DataFrame:
        tickers = sorted(self._collect_all_tickers())
        if not tickers:
            raise RuntimeError("No tickers found in wave_weights / benchmarks.")

        yf_map = {t: self._yf_symbol(t) for t in tickers}
        yf_symbols = sorted(set(yf_map.values()))

        data = yf.download(
            yf_symbols,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

        if data.empty:
            raise RuntimeError("Price download returned empty data.")

        # yfinance returns columns ['Adj Close', ...] or just ['Adj Close']
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data["Adj Close"].copy()
        else:
            # single column case
            adj_close = data.copy()

        # Rebuild to original tickers
        out = pd.DataFrame(index=adj_close.index)
        for orig, yf_sym in yf_map.items():
            if yf_sym in adj_close.columns:
                out[orig] = adj_close[yf_sym]
        # Drop columns that never downloaded
        out = out.dropna(axis=1, how="all")
        if out.empty:
            raise RuntimeError("No usable price data for any ticker.")
        return out

    @staticmethod
    def _pct_returns(price_df: pd.DataFrame) -> pd.DataFrame:
        rets = price_df.pct_change().dropna(how="all")
        return rets

    # ---------- core computations ----------

    def _wave_portfolio_returns(
        self, returns: pd.DataFrame, wave: str
    ) -> pd.Series:
        rows = self.wave_weights[self.wave_weights["wave"] == wave]
        if rows.empty:
            raise RuntimeError(f"No holdings found for wave '{wave}'")

        tickers = [t for t in rows["ticker"] if t in returns.columns]
        if not tickers:
            raise RuntimeError(f"No return data for any holdings in '{wave}'")

        w = rows.set_index("ticker")["weight"].reindex(tickers).astype(float)
        total = float(w.sum())
        if total <= 0:
            raise RuntimeError(f"Invalid weights for wave '{wave}'")
        w = w / total

        sub = returns[tickers]
        # portfolio daily returns
        port = sub.mul(w.values, axis=1).sum(axis=1)
        port.name = wave
        return port

    def _benchmark_returns(
        self, returns: pd.DataFrame, wave: str
    ) -> pd.Series | None:
        comp = self.benchmarks.get(wave)
        if not comp:
            return None

        tickers = [t for t in comp.keys() if t in returns.columns]
        if not tickers:
            # no benchmark data available
            return None

        w = pd.Series(comp).reindex(tickers).astype(float)
        total = float(w.sum())
        if total <= 0:
            return None
        w = w / total

        sub = returns[tickers]
        bench = sub.mul(w.values, axis=1).sum(axis=1)
        bench.name = f"{wave} (benchmark)"
        return bench

    @staticmethod
    def _period_return(series: pd.Series, days: int) -> float:
        if series is None or series.empty:
            return np.nan
        # last `days` observations
        last = series.iloc[-days:]
        if last.empty:
            return np.nan
        return float((1.0 + last).prod() - 1.0)

    def _compute_wave_metrics(
        self, all_returns: pd.DataFrame, wave: str
    ) -> WaveMetrics | None:
        try:
            port = self._wave_portfolio_returns(all_returns, wave)
        except Exception:
            return None

        bench = self._benchmark_returns(all_returns, wave)

        # 60D and 1Y portfolio returns
        r_60d = self._period_return(port, 60)
        r_1y = self._period_return(port, 252)

        if bench is not None:
            # align benchmark series to portfolio index
            bench = bench.reindex(port.index).dropna()
            alpha_series = port.loc[bench.index] - bench
            alpha_1y = self._period_return(alpha_series, 252)
        else:
            alpha_1y = np.nan

        return WaveMetrics(name=wave, ret_60d=r_60d, ret_1y=r_1y, alpha_1y=alpha_1y)

    # ---------- public API ----------

    def compute_all_metrics(self) -> Dict[str, WaveMetrics]:
        prices = self._download_price_data()
        returns = self._pct_returns(prices)

        waves = sorted(self.wave_weights["wave"].unique())
        metrics: Dict[str, WaveMetrics] = {}

        for wave in waves:
            m = self._compute_wave_metrics(returns, wave)
            if m is not None:
                metrics[wave] = m

        return metrics