# waves_engine.py
# WAVES Intelligence™ – Engine v2.1
# 10-Wave lineup with AI Wave + SmartSafe Wave
# Daily alpha capture vs blended ETF benchmarks

from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


# ---------- CONFIG ----------

LIST_CSV_PATH = "list.csv"
WAVE_WEIGHTS_CSV_PATH = "wave_weights.csv"

SMARTSAFE_WAVE = "SmartSafe Wave"

RISK_WAVES: List[str] = [
    "S&P Wave",
    "Growth Wave",
    "Cloud & Enterprise Software Growth Wave",
    "Crypto Equity Wave (mid/large cap)",
    "Clean Transit-Infrastructure Wave",
    "Income Wave",
    "Quantum Computing Wave",
    "Small-Mid Cap Growth Wave",
    "AI & Machine Intelligence Wave",
]

ALL_WAVES: List[str] = RISK_WAVES + [SMARTSAFE_WAVE]

# Benchmark blends (weights MUST sum to 1.0)
BENCHMARK_MAP: Dict[str, List[Tuple[str, float]]] = {
    # Broad Core
    "S&P Wave": [("SPY", 1.0)],

    # Large-cap growth core
    # (User requested 50/50 QQQ/IWF; we normalize to exactly 1.0)
    "Growth Wave": [("QQQ", 0.5), ("IWF", 0.5)],

    # Software / Cloud
    "Cloud & Enterprise Software Growth Wave": [
        ("IGV", 0.60),
        ("WCLD", 0.20),  # or CLOU – using WCLD here
        ("SPY", 0.20),
    ],

    # Crypto blend
    "Crypto Equity Wave (mid/large cap)": [
        ("FBTC", 0.70),  # spot BTC proxy
        ("DAPP", 0.30),  # digital asset equities
    ],

    # Clean Transit & Infrastructure (industrials/auto)
    "Clean Transit-Infrastructure Wave": [
        ("FIDU", 0.45),
        ("FDIS", 0.45),
        ("SPY", 0.10),
    ],

    # Income
    "Income Wave": [("SCHD", 1.0)],

    # Quantum Computing (proxy blend tech + semis)
    "Quantum Computing Wave": [
        ("QQQ", 0.60),
        ("SOXX", 0.40),
    ],

    # Small/Mid Growth
    "Small-Mid Cap Growth Wave": [
        ("CSMD", 0.40),  # Russell 2500 / similar
        ("IVOG", 0.40),  # or IVOO / MDYG – mid growth proxy
        ("SPY", 0.20),
    ],

    # NEW: AI Wave
    "AI & Machine Intelligence Wave": [
        ("IGV", 0.40),
        ("VGT", 0.25),
        ("QQQ", 0.20),
        ("SOXX", 0.15),
    ],

    # SmartSafe – treated separately; here for context only
    SMARTSAFE_WAVE: [
        ("SGOV", 0.34),
        ("BIL", 0.33),
        ("SHV", 0.33),
    ],
}

# How many trading days for each window
WINDOW_30D = 30
WINDOW_60D = 60
WINDOW_1Y = 252  # ~12 months trading days


@dataclass
class WaveMetrics:
    wave_name: str
    alpha_30d: Optional[float]
    alpha_60d: Optional[float]
    alpha_1y: Optional[float]
    wave_1y: Optional[float]
    benchmark_1y: Optional[float]


class WavesEngine:
    """
    Core portfolio + benchmark engine for WAVES Intelligence™
    """

    def __init__(
        self,
        list_csv: str = LIST_CSV_PATH,
        wave_weights_csv: str = WAVE_WEIGHTS_CSV_PATH,
        years_history: int = 3,
    ) -> None:
        self.list_csv = list_csv
        self.wave_weights_csv = wave_weights_csv
        self.years_history = years_history

        self.universe_df: Optional[pd.DataFrame] = None
        self.wave_weights: Dict[str, pd.Series] = {}
        self.price_history: Optional[pd.DataFrame] = None
        self.metrics_by_wave: Dict[str, WaveMetrics] = {}
        self.wave_nav: Dict[str, pd.Series] = {}
        self.benchmark_nav: Dict[str, Optional[pd.Series]] = {}

    # ---------- LOADING ----------

    def load_universe(self) -> pd.DataFrame:
        df = pd.read_csv(self.list_csv)

        # Normalize ticker column
        if "Ticker" in df.columns:
            df = df.rename(columns={"Ticker": "ticker"})
        elif "ticker" not in df.columns:
            raise ValueError(
                "list.csv must have a 'Ticker' or 'ticker' column."
            )

        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        self.universe_df = df
        return df

    def load_wave_weights(self) -> Dict[str, pd.Series]:
        df = pd.read_csv(self.wave_weights_csv)

        expected_cols = {"wave", "ticker", "weight"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(
                f"wave_weights.csv must have columns {expected_cols}, "
                f"found {set(df.columns)}"
            )

        df["wave"] = df["wave"].astype(str).str.strip()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["weight"] = df["weight"].astype(float)

        weights: Dict[str, pd.Series] = {}
        for wave, grp in df.groupby("wave"):
            w = grp.groupby("ticker")["weight"].sum()
            total = w.sum()
            if total <= 0:
                continue
            w = w / total
            weights[wave] = w

        # Only keep Waves we know about (risk waves), SmartSafe handled separately
        self.wave_weights = {
            w: s for w, s in weights.items() if w in RISK_WAVES
        }
        return self.wave_weights

    def _all_required_tickers(self) -> List[str]:
        tickers = set()

        # Wave constituents
        for s in self.wave_weights.values():
            tickers.update(list(s.index))

        # Benchmarks
        for comps in BENCHMARK_MAP.values():
            for sym, _w in comps:
                tickers.add(sym)

        # VIX for regime gating
        tickers.add("^VIX")

        return sorted(tickers)

    def load_price_history(self) -> pd.DataFrame:
        tickers = self._all_required_tickers()
        if not tickers:
            raise RuntimeError("No tickers discovered to download.")

        end = datetime.utcnow().date()
        start = end - timedelta(days=365 * self.years_history + 15)

        data = yf.download(
            tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
        )

        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            px = data["Adj Close"].copy()
        else:
            px = data.copy()

        px = px.sort_index()
        self.price_history = px
        return px

    # ---------- CALCULATIONS ----------

    @staticmethod
    def _to_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().fillna(0.0)

    def _compute_portfolio_nav(
        self, weights: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        if self.price_history is None:
            raise RuntimeError("Price history not loaded")

        # Filter to available columns
        cols = [t for t in weights.index if t in self.price_history.columns]
        if not cols:
            raise RuntimeError("None of the tickers exist in price history.")

        w = weights.loc[cols]
        w = w / w.sum()

        sub_px = self.price_history[cols]
        rets = self._to_returns(sub_px)

        port_ret = (rets * w.values).sum(axis=1)

        nav = (1.0 + port_ret).cumprod()
        return nav, port_ret

    def _compute_benchmark_nav(
        self, wave_name: str
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        comps = BENCHMARK_MAP.get(wave_name)
        if not comps:
            return None, None

        if self.price_history is None:
            raise RuntimeError("Price history not loaded")

        symbols = [sym for sym, _w in comps]
        weights = np.array([w for _sym, w in comps], dtype=float)

        cols = [s for s in symbols if s in self.price_history.columns]
        if not cols:
            return None, None

        # Re-align weights
        mask = [s in cols for s in symbols]
        weights = weights[mask]
        weights = weights / weights.sum()

        sub_px = self.price_history[cols]
        rets = self._to_returns(sub_px)

        bm_ret = (rets * weights).sum(axis=1)
        bm_nav = (1.0 + bm_ret).cumprod()
        return bm_nav, bm_ret

    def _vix_scale(self) -> float:
        if (
            self.price_history is None
            or "^VIX" not in self.price_history.columns
        ):
            return 1.0

        vix_series = self.price_history["^VIX"].dropna()
        if vix_series.empty:
            return 1.0

        vix = float(vix_series.iloc[-1])

        if vix < 18:
            return 1.25
        elif vix <= 28:
            return 1.0
        else:
            return 0.6

    @staticmethod
    def _window_total_return(nav: pd.Series, window_days: int) -> Optional[float]:
        if nav is None or nav.empty:
            return None
        if len(nav) < 2:
            return None

        window_days = min(window_days, len(nav) - 1)
        if window_days <= 0:
            return None

        end = nav.iloc[-1]
        start = nav.iloc[-(window_days + 1)]
        tr = (end / start) - 1.0
        return float(tr * 100.0)

    def _alpha_over_window(
        self,
        wave_nav: Optional[pd.Series],
        bm_nav: Optional[pd.Series],
        window_days: int,
    ) -> Optional[float]:
        if wave_nav is None or bm_nav is None:
            return None
        if wave_nav.empty or bm_nav.empty:
            return None

        # Align windows by index
        df = pd.DataFrame({"wave": wave_nav, "bm": bm_nav}).dropna()
        if len(df) < 2:
            return None

        window_days = min(window_days, len(df) - 1)
        if window_days <= 0:
            return None

        sub = df.iloc[-(window_days + 1) :]
        w_tr = (sub["wave"].iloc[-1] / sub["wave"].iloc[0]) - 1.0
        b_tr = (sub["bm"].iloc[-1] / sub["bm"].iloc[0]) - 1.0
        alpha = (w_tr - b_tr) * 100.0
        return float(alpha)

    def compute_all_metrics(self) -> Dict[str, WaveMetrics]:
        if self.price_history is None:
            raise RuntimeError("Price history not loaded")
        if not self.wave_weights:
            raise RuntimeError("Wave weights not loaded")

        vix_scale = self._vix_scale()

        metrics: Dict[str, WaveMetrics] = {}
        self.wave_nav = {}
        self.benchmark_nav = {}

        # ---------- Risk Waves ----------
        for wave in RISK_WAVES:
            if wave not in self.wave_weights:
                # allow Waves with no equity holdings (e.g., future upgrades)
                continue

            raw_nav, raw_rets = self._compute_portfolio_nav(
                self.wave_weights[wave]
            )

            # Apply VIX scaling to returns, then rebuild NAV
            scaled_rets = raw_rets * vix_scale
            scaled_nav = (1.0 + scaled_rets).cumprod()

            bm_nav, _bm_rets = self._compute_benchmark_nav(wave)

            self.wave_nav[wave] = scaled_nav
            self.benchmark_nav[wave] = bm_nav

            alpha_30 = self._alpha_over_window(
                scaled_nav, bm_nav, WINDOW_30D
            )
            alpha_60 = self._alpha_over_window(
                scaled_nav, bm_nav, WINDOW_60D
            )
            alpha_1y = self._alpha_over_window(
                scaled_nav, bm_nav, WINDOW_1Y
            )
            wave_1y = self._window_total_return(scaled_nav, WINDOW_1Y)
            bm_1y = (
                self._window_total_return(bm_nav, WINDOW_1Y)
                if bm_nav is not None
                else None
            )

            metrics[wave] = WaveMetrics(
                wave_name=wave,
                alpha_30d=alpha_30,
                alpha_60d=alpha_60,
                alpha_1y=alpha_1y,
                wave_1y=wave_1y,
                benchmark_1y=bm_1y,
            )

        # ---------- SmartSafe Wave ----------
        self._compute_smartsafe_metrics(metrics)

        self.metrics_by_wave = metrics
        return metrics

    def _compute_smartsafe_metrics(
        self, metrics: Dict[str, WaveMetrics]
    ) -> None:
        """SmartSafe yield-style metrics; not alpha-ranked."""

        comps = BENCHMARK_MAP.get(SMARTSAFE_WAVE)
        if not comps or self.price_history is None:
            return

        symbols = [sym for sym, _w in comps]
        weights = np.array([w for _sym, w in comps], dtype=float)

        cols = [s for s in symbols if s in self.price_history.columns]
        if not cols:
            return

        mask = [s in cols for s in symbols]
        weights = weights[mask]
        weights = weights / weights.sum()

        sub_px = self.price_history[cols]
        rets = self._to_returns(sub_px)
        port_ret = (rets * weights).sum(axis=1)
        nav = (1.0 + port_ret).cumprod()

        self.wave_nav[SMARTSAFE_WAVE] = nav
        self.benchmark_nav[SMARTSAFE_WAVE] = None

        wave_1y = self._window_total_return(nav, WINDOW_1Y)

        metrics[SMARTSAFE_WAVE] = WaveMetrics(
            wave_name=SMARTSAFE_WAVE,
            alpha_30d=None,
            alpha_60d=None,
            alpha_1y=None,
            wave_1y=wave_1y,
            benchmark_1y=None,
        )


# Convenience helper used by app.py
def build_engine() -> WavesEngine:
    engine = WavesEngine()
    engine.load_universe()
    engine.load_wave_weights()
    engine.load_price_history()
    engine.compute_all_metrics()
    return engine


if __name__ == "__main__":
    # Simple sanity run if you execute this file directly
    eng = build_engine()
    print("Loaded Waves:")
    for w, m in eng.metrics_by_wave.items():
        print(
            f"{w:40s}  "
            f"α30={m.alpha_30d:.2f if m.alpha_30d is not None else float('nan')}"
        )