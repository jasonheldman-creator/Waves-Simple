"""
waves_engine.py — Clean, Non-Circular Version
WAVES Intelligence™ Core Engine
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


class WavesEngine:

    # ---------------------------------------------------------------
    # INIT
    # ---------------------------------------------------------------
    def __init__(self, list_path="list.csv", weights_path="wave_weights.csv"):
        self.list_path = Path(list_path)
        self.weights_path = Path(weights_path)

        self.universe = self._load_list()
        self.weights = self._load_weights()

    # ---------------------------------------------------------------
    # Load list.csv
    # ---------------------------------------------------------------
    def _load_list(self):
        df = pd.read_csv(self.list_path)

        required_cols = {"Ticker"}
        if not required_cols.issubset(df.columns):
            raise ValueError("list.csv must include a 'Ticker' column")

        df = df.rename(columns=str.lower)
        return df

    # ---------------------------------------------------------------
    # Load wave_weights.csv
    # ---------------------------------------------------------------
    def _load_weights(self):
        df = pd.read_csv(self.weights_path)

        required = {"wave", "ticker", "weight"}
        if not required.issubset(df.columns):
            raise ValueError("wave_weights.csv must include wave,ticker,weight columns")

        df["ticker"] = df["ticker"].str.upper()
        return df

    # ---------------------------------------------------------------
    # WAVE NAMES
    # ---------------------------------------------------------------
    def get_wave_names(self):
        return sorted(self.weights["wave"].unique())

    # ---------------------------------------------------------------
    # BENCHMARK (default SPY)
    # ---------------------------------------------------------------
    def get_benchmark(self, wave):
        return "SPY"

    # ---------------------------------------------------------------
    # HOLDINGS + TOP 10
    # ---------------------------------------------------------------
    def get_wave_holdings(self, wave):
        df = self.weights[self.weights["wave"] == wave].copy()
        return df

    def get_top_holdings(self, wave, n=10):
        df = self.get_wave_holdings(wave)
        df = df.sort_values("weight", ascending=False).head(n)
        return df

    # ---------------------------------------------------------------
    # PRICE HISTORY
    # ---------------------------------------------------------------
    def _fetch(self, tickers, period="1y"):
        data = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(tickers, str):
            return data["Close"]
        return data["Close"]

    # ---------------------------------------------------------------
    # WAVE PERFORMANCE
    # ---------------------------------------------------------------
    def get_wave_performance(self, wave, mode="standard", days=30, log=False):
        holdings = self.get_wave_holdings(wave)

        if holdings.empty:
            return None

        weights = holdings.set_index("ticker")["weight"]
        tickers = list(weights.index)

        hist = self._fetch(tickers, period="1y")
        if hist is None or hist.empty:
            return None

        wave_series = hist.pct_change().mul(weights, axis=1).sum(axis=1)
        wave_curve = (1 + wave_series).cumprod()

        benchmark = self.get_benchmark(wave)
        bm_series = self._fetch(benchmark, period="1y").pct_change()
        bm_curve = (1 + bm_series).cumprod()

        # Align
        df = pd.DataFrame({
            "wave_return": wave_series,
            "benchmark_return": bm_series,
            "wave_value": wave_curve,
            "benchmark_value": bm_curve,
        }).dropna()

        # DAILY ALPHA CAPTURE
        df["alpha_captured"] = df["wave_return"] - df["benchmark_return"]

        # WINDOWS
        intraday_alpha = df["alpha_captured"].iloc[-1]
        alpha_30 = df["alpha_captured"].tail(30).sum()
        alpha_60 = df["alpha_captured"].tail(60).sum()
        alpha_1y = df["alpha_captured"].sum()

        # RETURN WINDOWS
        def ret(series, n):
            if len(series) < n:
                return None
            return (series[-1] / series[-n]) - 1

        ret_30 = ret(df["wave_value"], 30)
        ret_30_bm = ret(df["benchmark_value"], 30)
        ret_60 = ret(df["wave_value"], 60)
        ret_60_bm = ret(df["benchmark_value"], 60)
        ret_1y = ret(df["wave_value"], 250)
        ret_1y_bm = ret(df["benchmark_value"], 250)

        # REALIZED BETA (60d)
        beta = (
            np.cov(df["wave_return"].tail(60), df["benchmark_return"].tail(60))[0][1] /
            np.var(df["benchmark_return"].tail(60))
        )

        # EXPOSURE — simple placeholder
        exposure = 1.0
        if mode == "alpha-minus-beta":
            exposure = max(0.3, 1 - beta)
        elif mode == "private_logic":
            exposure = 1.1

        return {
            "benchmark": benchmark,
            "beta_realized": beta,
            "exposure_final": exposure,

            "intraday_alpha_captured": intraday_alpha,
            "alpha_30d": alpha_30,
            "alpha_60d": alpha_60,
            "alpha_1y": alpha_1y,

            "return_30d_wave": ret_30,
            "return_30d_benchmark": ret_30_bm,
            "return_60d_wave": ret_60,
            "return_60d_benchmark": ret_60_bm,
            "return_1y_wave": ret_1y,
            "return_1y_benchmark": ret_1y_bm,

            "history_30d": df.tail(30),
        }