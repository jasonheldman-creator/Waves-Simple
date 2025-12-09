"""
waves_engine.py — Institutional Hybrid Engine (B3 + H3)

- Per-wave strategy recipes (S&P, AI, Infinity, etc.)
- VIX ladder -> SmartSafe allocation
- Target beta per wave
- MarketStack first, yfinance fallback, synthetic last
- Writes daily logs to:
    logs/positions/<Wave>_positions_YYYYMMDD.csv
    logs/performance/<Wave>_performance_daily.csv

This is designed to be SAFE:
- If external data fails, it falls back gracefully.
- If logs are missing, the console can still use sandbox/demo logic.
"""

from __future__ import annotations

import os
import math
import glob
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Engine version tracking
ENGINE_VERSION = "1.5.0"

def _get_engine_last_updated() -> str:
    """Get the last modification time of this file (in local time)."""
    try:
        import inspect
        engine_file = inspect.getfile(inspect.currentframe())
        if os.path.exists(engine_file):
            mtime = os.path.getmtime(engine_file)
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return "Unknown"

ENGINE_LAST_UPDATED = _get_engine_last_updated()

# Optional deps: requests + yfinance
try:
    import requests
except Exception:
    requests = None

try:
    import yfinance as yf
except Exception:
    yf = None

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

ROOT_DIR = Path(".")
LOGS_DIR = ROOT_DIR / "logs"
LOG_POSITIONS_DIR = LOGS_DIR / "positions"
LOG_PERFORMANCE_DIR = LOGS_DIR / "performance"

for d in (LOGS_DIR, LOG_POSITIONS_DIR, LOG_PERFORMANCE_DIR):
    d.mkdir(parents=True, exist_ok=True)

WAVE_WEIGHTS_PATH = ROOT_DIR / "wave_weights.csv"

VIX_TICKER = "^VIX"

LOOKBACK_DAYS_DEFAULT = 365  # 1-year baseline; extend if you like

MODES = ["Standard", "AlphaMinusBeta", "PrivateLogic"]

# ---------------------------------------------------------------------
# Wave recipes (institutional spec)
# ---------------------------------------------------------------------


@dataclass
class WaveRecipe:
    name: str
    category: str
    benchmark: str
    target_beta: float
    risk_band: str
    use_smartsafe: bool = True
    mode_scalars: Dict[str, float] = None

    def scalar_for_mode(self, mode: str) -> float:
        base = self.mode_scalars or {
            "Standard": 1.00,
            "AlphaMinusBeta": 0.80,
            "PrivateLogic": 1.10,
        }
        return float(base.get(mode, 1.0))


# 10-wave lineup (9 equity + SmartSafe)
WAVE_RECIPES: Dict[str, WaveRecipe] = {
    "S&P 500 Wave": WaveRecipe(
        name="S&P 500 Wave",
        category="Core Equity",
        benchmark="SPY",
        target_beta=0.90,
        risk_band="Core",
        use_smartsafe=True,
        mode_scalars={
            "Standard": 1.00,
            "AlphaMinusBeta": 0.80,
            "PrivateLogic": 1.05,
        },
    ),
    "Growth Wave": WaveRecipe(
        name="Growth Wave",
        category="Growth Equity",
        benchmark="SPYG",
        target_beta=1.05,
        risk_band="Core",
    ),
    "Small Cap Growth Wave": WaveRecipe(
        name="Small Cap Growth Wave",
        category="Small Cap Growth",
        benchmark="IWO",
        target_beta=1.20,
        risk_band="Growth",
    ),
    "Future Power & Energy Wave": WaveRecipe(
        name="Future Power & Energy Wave",
        category="Thematic Equity",
        benchmark="XLE",
        target_beta=1.10,
        risk_band="Growth",
    ),
    "Quantum Computing Wave": WaveRecipe(
        name="Quantum Computing Wave",
        category="Thematic Equity",
        benchmark="QQQ",
        target_beta=1.30,
        risk_band="Thematic",
    ),
    "Clean Transit-Infrastructure Wave": WaveRecipe(
        name="Clean Transit-Infrastructure Wave",
        category="Thematic Equity",
        benchmark="SPY",
        target_beta=1.15,
        risk_band="Thematic",
    ),
    "AI Wave": WaveRecipe(
        name="AI Wave",
        category="Thematic Equity",
        benchmark="QQQ",
        target_beta=1.25,
        risk_band="Growth",
        mode_scalars={
            "Standard": 1.10,
            "AlphaMinusBeta": 0.85,
            "PrivateLogic": 1.30,
        },
    ),
    "Infinity Wave": WaveRecipe(
        name="Infinity Wave",
        category="Flagship Multi-Theme",
        benchmark="ACWI",
        target_beta=1.10,
        risk_band="Flagship",
        mode_scalars={
            "Standard": 1.05,
            "AlphaMinusBeta": 0.85,
            "PrivateLogic": 1.20,
        },
    ),
    "International Developed Wave": WaveRecipe(
        name="International Developed Wave",
        category="Global Equity",
        benchmark="EFA",
        target_beta=1.00,
        risk_band="Core",
    ),
    "SmartSafe Wave": WaveRecipe(
        name="SmartSafe Wave",
        category="SmartSafe / Cash",
        benchmark="BIL",
        target_beta=0.05,
        risk_band="SmartSafe",
        use_smartsafe=False,
        mode_scalars={
            "Standard": 0.05,
            "AlphaMinusBeta": 0.05,
            "PrivateLogic": 0.05,
        },
    ),
}

SMARTSAFE_ANNUAL_YIELD = 0.05  # 5% cash-like yield


def list_wave_names(include_smartsafe: bool = True) -> List[str]:
    names = list(WAVE_RECIPES.keys())
    if not include_smartsafe:
        names = [n for n in names if n != "SmartSafe Wave"]
    return names


# ---------------------------------------------------------------------
# Wave weights loader
# ---------------------------------------------------------------------


def load_wave_weights(path: Path = WAVE_WEIGHTS_PATH) -> pd.DataFrame:
    """
    Expected columns:
        Wave, Ticker, Weight
    """
    if not path.exists():
        raise FileNotFoundError(f"wave_weights.csv not found at {path}")

    df = pd.read_csv(path)
    required = {"Wave", "Ticker", "Weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv missing required columns: {missing}")

    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Weight"] = df["Weight"].astype(float)

    # Normalize weights within each wave
    df["Weight"] = df.groupby("Wave")["Weight"].transform(
        lambda w: w / w.sum() if w.sum() != 0 else w
    )
    return df


def get_strategy_recipe(wave_name: str) -> Dict[str, object]:
    """
    Used by the console sidebar to show snapshot info.
    """
    recipe = WAVE_RECIPES.get(wave_name)
    if recipe is None:
        return {}
    return {
        "wave": recipe.name,
        "category": recipe.category,
        "benchmark": recipe.benchmark,
        "target_beta": recipe.target_beta,
        "risk_band": recipe.risk_band,
        "use_smartsafe": recipe.use_smartsafe,
    }


# ---------------------------------------------------------------------
# MarketStack + yfinance hybrid price fetcher
# ---------------------------------------------------------------------


def _fetch_marketstack_prices(
    tickers: List[str], start: datetime, end: datetime
) -> Optional[pd.DataFrame]:
    """
    Fetch daily adjusted close prices from MarketStack.
    Requires MARKETSTACK_API_KEY in environment.
    """
    api_key = os.getenv("MARKETSTACK_API_KEY")
    if not api_key or requests is None:
        return None

    # MarketStack free tier is limited; we keep this simple.
    # We'll fetch each ticker individually for robustness.
    frames = []
    for ticker in tickers:
        try:
            params = {
                "access_key": api_key,
                "symbols": ticker,
                "date_from": start.strftime("%Y-%m-%d"),
                "date_to": end.strftime("%Y-%m-%d"),
                "limit": 1000,
            }
            resp = requests.get(
                "http://api.marketstack.com/v1/eod", params=params, timeout=10
            )
            if resp.status_code != 200:
                continue
            data = resp.json().get("data", [])
            if not data:
                continue
            df = pd.DataFrame(data)
            if "date" not in df.columns or "adj_close" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.set_index("date").sort_index()
            df = df.loc[(df.index >= start) & (df.index <= end)]
            frames.append(df["adj_close"].rename(ticker))
        except Exception:
            continue

    if not frames:
        return None

    out = pd.concat(frames, axis=1)
    return out


def _fetch_yfinance_prices(
    tickers: List[str], start: datetime, end: datetime
) -> Optional[pd.DataFrame]:
    if yf is None or not tickers:
        return None
    try:
        data = yf.download(
            tickers=tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if data.empty:
            return None

        # Normalize to simple wide frame: index=date, columns=tickers
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])

        close.index = close.index.tz_localize(None)
        close = close.sort_index()
        return close
    except Exception:
        return None


def _synthetic_prices(
    tickers: List[str], start: datetime, end: datetime
) -> pd.DataFrame:
    """
    Final fallback: generate smooth synthetic random walk prices.
    """
    dates = pd.date_range(start=start, end=end, freq="B")
    rng = np.random.default_rng(42)
    data = {}
    for t in tickers:
        steps = rng.normal(loc=0.0003, scale=0.01, size=len(dates))
        series = 100 * (1 + pd.Series(steps, index=dates)).cumprod()
        data[t] = series
    return pd.DataFrame(data, index=dates)


def fetch_price_history(
    tickers: List[str], start: datetime, end: datetime
) -> pd.DataFrame:
    """
    H3 hybrid:
        1) Try MarketStack (if API key present)
        2) Fallback to yfinance
        3) Fallback to synthetic
    """
    tickers = sorted(list({t.upper() for t in tickers if t}))
    if not tickers:
        return pd.DataFrame()

    # Try MarketStack
    ms = _fetch_marketstack_prices(tickers, start, end)
    if ms is not None and not ms.empty:
        px = ms.copy()
        px = px.reindex(pd.date_range(start=start, end=end, freq="B")).ffill().bfill()
        return px

    # Fallback: yfinance
    yf_px = _fetch_yfinance_prices(tickers, start, end)
    if yf_px is not None and not yf_px.empty:
        px = yf_px.copy()
        px = px.reindex(pd.date_range(start=start, end=end, freq="B")).ffill().bfill()
        return px

    # Fallback: synthetic
    return _synthetic_prices(tickers, start, end)


# ---------------------------------------------------------------------
# VIX ladder & SmartSafe allocation
# ---------------------------------------------------------------------


def fetch_vix_history(days: int = 365) -> Optional[pd.Series]:
    """
    Return daily VIX close; fallback to synthetic if needed.
    """
    if yf is None:
        # Synthetic "calm" VIX around 18
        dates = pd.date_range(
            end=datetime.utcnow().date(), periods=days, freq="B"
        )
        vals = np.full(len(dates), 18.0)
        return pd.Series(vals, index=dates, name="VIX")

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 5)
    try:
        data = yf.download(VIX_TICKER, start=start, end=end, progress=False)
        if data.empty:
            raise RuntimeError("Empty VIX data")
        s = data["Adj Close"]
        s.index = s.index.tz_localize(None)
        s = s.sort_index()
        s.name = "VIX"
        return s
    except Exception:
        dates = pd.date_range(
            end=datetime.utcnow().date(), periods=days, freq="B"
        )
        vals = np.full(len(dates), 18.0)
        return pd.Series(vals, index=dates, name="VIX")


def smartsafe_weight_from_vix(vix: float) -> float:
    """
    Institutional-ish VIX ladder -> SmartSafe allocation.
    """
    if vix < 16:
        return 0.05
    elif vix < 20:
        return 0.10
    elif vix < 25:
        return 0.20
    elif vix < 30:
        return 0.35
    elif vix < 40:
        return 0.55
    else:
        return 0.75


def risk_regime_from_vix(vix: float) -> str:
    if vix < 16:
        return "Calm"
    elif vix < 20:
        return "Normal"
    elif vix < 25:
        return "Elevated"
    elif vix < 30:
        return "Stressed"
    elif vix < 40:
        return "Crisis"
    else:
        return "Panic"


# ---------------------------------------------------------------------
# Per-wave performance construction
# ---------------------------------------------------------------------


def _today_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d")


def build_wave_performance_series(
    wave_name: str,
    weights_df: Optional[pd.DataFrame] = None,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
) -> pd.DataFrame:
    """
    Build the STANDARD-mode daily performance series for a given wave.

    Columns:
        date, wave, nav, return_1d,
        bench_nav, bench_return_1d,
        return_30d, return_60d, return_252d,
        bench_return_30d, bench_return_60d, bench_return_252d,
        alpha_1d, alpha_30d, alpha_60d, alpha_1y,
        smartsafe_weight, smartsafe_yield_annual,
        vix, risk_regime, regime
    """
    recipe = WAVE_RECIPES.get(wave_name)
    if recipe is None:
        return pd.DataFrame()

    if weights_df is None:
        weights_df = load_wave_weights()

    w = weights_df[weights_df["Wave"] == wave_name].copy()
    if w.empty and wave_name != "SmartSafe Wave":
        # No explicit positions; still provide a synthetic path
        tickers = [recipe.benchmark]
        weights = np.array([1.0])
    elif wave_name == "SmartSafe Wave":
        # SmartSafe: treat like a synthetic cash sleeve
        tickers = []
        weights = np.array([])
    else:
        tickers = sorted(w["Ticker"].unique().tolist())
        weights = (
            w.groupby("Ticker")["Weight"].sum().reindex(tickers).fillna(0.0).values
        )
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()

    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days + 10)

    dates = pd.date_range(start=start, end=end, freq="B")

    # Fetch VIX series
    vix_series = fetch_vix_history(lookback_days + 30)
    vix_series = vix_series.reindex(dates, method="ffill").fillna(18.0)

    if wave_name == "SmartSafe Wave":
        # Synthetic SmartSafe path: steady 5% annual yield
        r_daily = (1 + SMARTSAFE_ANNUAL_YIELD) ** (1 / 252.0) - 1
        nav = 100 * (1 + r_daily) ** np.arange(len(dates))
        nav_series = pd.Series(nav, index=dates, name="nav")
        ret = nav_series.pct_change().fillna(0.0)
        # Benchmark = BIL-like; assume same as SmartSafe for simplicity
        bench_nav = nav_series.copy()
        bench_ret = ret.copy()
    else:
        # Fetch portfolio & benchmark prices
        # Portfolio prices:
        price_df = fetch_price_history(tickers, dates[0], dates[-1])
        price_df = price_df.reindex(dates).ffill().bfill()

        rel = price_df / price_df.iloc[0]
        nav_series = (rel * weights).sum(axis=1) * 100.0
        ret = nav_series.pct_change().fillna(0.0)

        # Benchmark prices:
        bench_px = fetch_price_history([recipe.benchmark], dates[0], dates[-1])
        if bench_px.empty:
            bench_nav = nav_series.copy()
            bench_ret = ret.copy() * 0.0
        else:
            s = bench_px[recipe.benchmark].reindex(dates).ffill().bfill()
            bench_nav = s / s.iloc[0] * 100.0
            bench_ret = bench_nav.pct_change().fillna(0.0)

    df = pd.DataFrame(
        {
            "date": dates,
            "wave": wave_name,
            "nav": nav_series.values,
            "return_1d": ret.values,
            "bench_nav": bench_nav.values,
            "bench_return_1d": bench_ret.values,
            "vix": vix_series.values,
        }
    )

    # Horizon returns
    df["return_30d"] = np.nan
    df["return_60d"] = np.nan
    df["return_252d"] = np.nan
    df["bench_return_30d"] = np.nan
    df["bench_return_60d"] = np.nan
    df["bench_return_252d"] = np.nan

    for i in range(len(df)):
        if i >= 21:
            df.loc[df.index[i], "return_30d"] = (
                df["nav"].iloc[i] / df["nav"].iloc[i - 21] - 1.0
            )
            df.loc[df.index[i], "bench_return_30d"] = (
                df["bench_nav"].iloc[i] / df["bench_nav"].iloc[i - 21] - 1.0
            )
        if i >= 42:
            df.loc[df.index[i], "return_60d"] = (
                df["nav"].iloc[i] / df["nav"].iloc[i - 42] - 1.0
            )
            df.loc[df.index[i], "bench_return_60d"] = (
                df["bench_nav"].iloc[i] / df["bench_nav"].iloc[i - 42] - 1.0
            )
        if i >= 252:
            df.loc[df.index[i], "return_252d"] = (
                df["nav"].iloc[i] / df["nav"].iloc[i - 252] - 1.0
            )
            df.loc[df.index[i], "bench_return_252d"] = (
                df["bench_nav"].iloc[i] / df["bench_nav"].iloc[i - 252] - 1.0
            )

    # Alpha columns
    df["alpha_1d"] = df["return_1d"] - df["bench_return_1d"]
    df["alpha_30d"] = df["return_30d"] - df["bench_return_30d"]
    df["alpha_60d"] = df["return_60d"] - df["bench_return_60d"]
    df["alpha_1y"] = df["return_252d"] - df["bench_return_252d"]

    # SmartSafe allocation & regime
    ss_weights = []
    regimes = []
    for v in df["vix"]:
        w_ss = smartsafe_weight_from_vix(float(v)) if recipe.use_smartsafe else 0.0
        ss_weights.append(w_ss)
        regimes.append(risk_regime_from_vix(float(v)))
    df["smartsafe_weight"] = ss_weights
    df["smartsafe_yield_annual"] = SMARTSAFE_ANNUAL_YIELD
    df["risk_regime"] = regimes
    df["regime"] = "LIVE"  # vs SANDBOX if we ever simulate

    # nav_risk (for compatibility)
    df["nav_risk"] = df["nav"]

    # Trim to lookback window
    cutoff = datetime.utcnow().date() - timedelta(days=lookback_days)
    df = df[df["date"].dt.date >= cutoff]

    return df.reset_index(drop=True)


def write_wave_logs(
    wave_name: str,
    weights_df: Optional[pd.DataFrame] = None,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Build performance & positions logs for a Wave and write them to disk.

    Returns: (positions_path, performance_path)
    """
    if weights_df is None:
        weights_df = load_wave_weights()

    # Positions snapshot (today)
    today_tag = _today_tag()
    pos = weights_df[weights_df["Wave"] == wave_name].copy()
    if pos.empty and wave_name == "SmartSafe Wave":
        # Synthetic SmartSafe position
        pos = pd.DataFrame(
            {
                "Wave": [wave_name],
                "Ticker": ["BIL"],
                "Weight": [1.0],
            }
        )
    pos.insert(0, "Date", datetime.utcnow().strftime("%Y-%m-%d"))
    pos_path = LOG_POSITIONS_DIR / f"{wave_name.replace(' ', '_')}_positions_{today_tag}.csv"
    pos.to_csv(pos_path, index=False)

    # Performance series
    perf_df = build_wave_performance_series(
        wave_name, weights_df=weights_df, lookback_days=lookback_days
    )
    if perf_df.empty:
        perf_path = None
    else:
        perf_path = LOG_PERFORMANCE_DIR / f"{wave_name.replace(' ', '_')}_performance_daily.csv"
        perf_df.to_csv(perf_path, index=False)

    return pos_path, perf_path


def run_engine_for_all_waves(lookback_days: int = LOOKBACK_DAYS_DEFAULT) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Rebuild logs for all Waves; returns a map of file paths as strings.
    """
    out: Dict[str, Dict[str, Optional[str]]] = {}
    try:
        weights_df = load_wave_weights()
    except Exception:
        # If weights file missing, we still allow SmartSafe to be built.
        weights_df = pd.DataFrame(columns=["Wave", "Ticker", "Weight"])

    for wave in list_wave_names(include_smartsafe=True):
        pos_path, perf_path = write_wave_logs(
            wave, weights_df=weights_df, lookback_days=lookback_days
        )
        out[wave] = {
            "positions": str(pos_path) if pos_path else None,
            "performance": str(perf_path) if perf_path else None,
        }
    return out


# ---------------------------------------------------------------------
# Performance loaders & alpha capture (for console use)
# ---------------------------------------------------------------------


def _latest_performance_file(wave_name: str) -> Optional[Path]:
    safe = wave_name.replace(" ", "_")
    pattern = str(LOG_PERFORMANCE_DIR / f"{safe}_performance_daily*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return Path(files[-1])


def load_wave_performance_history(wave_name: str) -> pd.DataFrame:
    """
    Load performance CSV that run_engine_for_all_waves wrote.
    """
    path = _latest_performance_file(wave_name)
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def compute_alpha_capture_for_wave_from_logs(
    wave_name: str, mode: str = "Standard"
) -> Dict[str, float]:
    """
    Read the logged performance series for a wave and compute alpha metrics,
    applying mode-specific scaling (equity exposure).
    """
    if mode not in MODES:
        mode = "Standard"

    recipe = WAVE_RECIPES.get(wave_name)
    if recipe is None:
        return {}

    df = load_wave_performance_history(wave_name)
    if df.empty or "return_1d" not in df.columns:
        return {}

    # Mode scaling: we treat logged returns as "Standard" equity returns.
    scalar = recipe.scalar_for_mode(mode)

    # SmartSafe effect is already embedded at the Standard level;
    # here we only tweak equity exposure for AMB/PL.
    wave_ret = df["return_1d"].astype(float) * scalar
    bench_ret = df.get("bench_return_1d", pd.Series(0.0, index=df.index)).astype(float)

    def horizon(total_days: int) -> Tuple[float, float]:
        if len(df) < total_days:
            return float("nan"), float("nan")
        r_w = (1 + wave_ret.iloc[-total_days:]).prod() - 1.0
        r_b = (1 + bench_ret.iloc[-total_days:]).prod() - 1.0
        return float(r_w), float(r_b)

    r1, b1 = horizon(1)
    r30, b30 = horizon(21)
    r60, b60 = horizon(42)
    r252, b252 = horizon(252)

    # Information ratio (1-year)
    if len(df) > 60:
        excess = wave_ret - bench_ret
        mu = excess.mean()
        sigma = excess.std(ddof=1)
        ir = mu / sigma * math.sqrt(252) if sigma > 0 else float("nan")
    else:
        ir = float("nan")

    # Latest SmartSafe & VIX info for UI
    ss = float(df["smartsafe_weight"].iloc[-1]) if "smartsafe_weight" in df.columns else 0.0
    vix = float(df["vix"].iloc[-1]) if "vix" in df.columns else float("nan")
    regime = str(df["risk_regime"].iloc[-1]) if "risk_regime" in df.columns else ""

    return {
        "Wave": wave_name,
        "Category": recipe.category,
        "Benchmark": recipe.benchmark,
        "Mode": mode,
        "Return_1d": r1,
        "Alpha_1d": r1 - b1,
        "Return_30d": r30,
        "Alpha_30d": r30 - b30,
        "Return_60d": r60,
        "Alpha_60d": r60 - b60,
        "Return_1y": r252,
        "Alpha_1y": r252 - b252,
        "Equity_Exposure_Scalar": scalar,
        "SmartSafe_Alloc": ss,
        "VIX": vix,
        "Risk_Regime": regime,
        "Alpha_IR": ir,
    }


def compute_alpha_capture_matrix(mode: str = "Standard") -> pd.DataFrame:
    """
    Engine-driven alpha matrix for all waves in a given mode.
    """
    rows = []
    for wave in list_wave_names(include_smartsafe=True):
        row = compute_alpha_capture_for_wave_from_logs(wave, mode=mode)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_sort"] = df["Wave"].apply(lambda w: 999 if "SmartSafe" in w else 0)
    df = df.sort_values(["_sort", "Wave"]).drop(columns=["_sort"]).reset_index(drop=True)
    return df


def list_waves_with_data() -> pd.DataFrame:
    """
    System status helper: tells you which waves have performance logs.
    """
    rows = []
    for wave in list_wave_names(include_smartsafe=True):
        df = load_wave_performance_history(wave)
        if df.empty:
            rows.append(
                {
                    "Wave": wave,
                    "Has_Performance": False,
                    "First_Date": None,
                    "Last_Date": None,
                    "Row_Count": 0,
                }
            )
        else:
            rows.append(
                {
                    "Wave": wave,
                    "Has_Performance": True,
                    "First_Date": df["date"].min(),
                    "Last_Date": df["date"].max(),
                    "Row_Count": len(df),
                }
            )
    return pd.DataFrame(rows)


def get_engine_status() -> Dict[str, str]:
    """
    Lightweight engine status summary for the console.
    """
    status: Dict[str, str] = {}
    status["engine_version"] = ENGINE_VERSION
    status["engine_updated"] = ENGINE_LAST_UPDATED
    status["positions_dir_exists"] = str(LOG_POSITIONS_DIR.exists())
    status["performance_dir_exists"] = str(LOG_PERFORMANCE_DIR.exists())
    status["positions_csv_count"] = str(len(glob.glob(str(LOG_POSITIONS_DIR / "*.csv"))))
    status["performance_csv_count"] = str(
        len(glob.glob(str(LOG_PERFORMANCE_DIR / "*.csv")))
    )
    # VIX
    vix_hist = fetch_vix_history(30)
    if vix_hist is None or vix_hist.empty:
        status["vix_status"] = "UNAVAILABLE"
        status["vix_latest"] = "N/A"
    else:
        status["vix_status"] = "OK"
        status["vix_latest"] = f"{float(vix_hist.iloc[-1]):.2f}"
    return status

def get_engine_version_info() -> Dict[str, str]:
    """
    Get engine version information.
    """
    return {
        "version": ENGINE_VERSION,
        "last_updated": ENGINE_LAST_UPDATED,
    }


__all__ = [
    "ENGINE_VERSION",
    "ENGINE_LAST_UPDATED",
    "WaveRecipe",
    "WAVE_RECIPES",
    "list_wave_names",
    "load_wave_weights",
    "get_strategy_recipe",
    "fetch_price_history",
    "fetch_vix_history",
    "smartsafe_weight_from_vix",
    "risk_regime_from_vix",
    "build_wave_performance_series",
    "write_wave_logs",
    "run_engine_for_all_waves",
    "load_wave_performance_history",
    "compute_alpha_capture_for_wave_from_logs",
    "compute_alpha_capture_matrix",
    "list_waves_with_data",
    "get_engine_status",
    "get_engine_version_info",
]
