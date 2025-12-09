"""
waves_engine.py — Engine v2 (Institutional Mode)

Core ideas:
- Per-wave strategy recipes (benchmark, target beta, category)
- VIX ladder -> SmartSafe allocation
- Mode-specific exposure rules:
    * Standard
    * Alpha-Minus-Beta
    * Private Logic™
- Alpha capture metrics on 1d / 30d / 60d / 1y horizons

This module is intentionally self-contained and conservative:
- It NEVER crashes the UI: all I/O is wrapped with safe fallbacks.
- If data is missing, it returns empty DataFrames or NaNs instead of raising.
- All outputs are pandas DataFrames ready for Streamlit display.

Directories expected by the console:
- logs/performance/<Wave>_performance_daily.csv
- logs/positions/<Wave>_positions_YYYYMMDD.csv  (not strictly required here)

You can safely drop this file into your repo as waves_engine.py.
"""

from __future__ import annotations

import os
import glob
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # Console will still work from existing logs


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

ROOT_DIR = Path(".")
LOG_POSITIONS_DIR = ROOT_DIR / "logs" / "positions"
LOG_PERFORMANCE_DIR = ROOT_DIR / "logs" / "performance"

VIX_TICKER = "^VIX"

MODES = ["Standard", "AlphaMinusBeta", "PrivateLogic"]

# -----------------------------------------------------------------------------
# Wave recipes (institutional spec)
# -----------------------------------------------------------------------------


@dataclass
class WaveRecipe:
    name: str
    category: str
    benchmark: str        # ETF / index proxy (e.g., SPY, QQQ, EFA)
    target_beta: float    # Target equity beta vs benchmark
    risk_band: str        # "Core", "Growth", "Thematic", "Flagship", "SmartSafe"
    use_smartsafe: bool = True
    # Mode-specific multipliers applied to target_beta (pre-SmartSafe)
    mode_scalars: Dict[str, float] = None

    def scalar_for_mode(self, mode: str) -> float:
        if self.mode_scalars is None:
            # Reasonable defaults
            base = {
                "Standard": 1.00,
                "AlphaMinusBeta": 0.80,
                "PrivateLogic": 1.10,
            }
        else:
            base = self.mode_scalars
        return float(base.get(mode, 1.0))


# IMPORTANT:
# These are *strategic* definitions. They do NOT change historical prices,
# but they change how we interpret exposure, alpha and SmartSafe usage.
WAVE_RECIPES: Dict[str, WaveRecipe] = {
    "AI Wave": WaveRecipe(
        name="AI Wave",
        category="Thematic Equity",
        benchmark="QQQ",
        target_beta=1.25,
        risk_band="Growth",
        use_smartsafe=True,
        mode_scalars={
            "Standard": 1.10,
            "AlphaMinusBeta": 0.85,
            "PrivateLogic": 1.30,
        },
    ),
    "Clean Transit-Infrastructure Wave": WaveRecipe(
        name="Clean Transit-Infrastructure Wave",
        category="Thematic Equity",
        benchmark="SPY",
        target_beta=1.15,
        risk_band="Growth",
        use_smartsafe=True,
    ),
    "Emerging Markets Wave": WaveRecipe(
        name="Emerging Markets Wave",
        category="Global Equity",
        benchmark="EEM",
        target_beta=1.05,
        risk_band="Core",
        use_smartsafe=True,
    ),
    "Future Power & Energy Wave": WaveRecipe(
        name="Future Power & Energy Wave",
        category="Thematic Equity",
        benchmark="XLE",
        target_beta=1.10,
        risk_band="Growth",
        use_smartsafe=True,
    ),
    "Growth Wave": WaveRecipe(
        name="Growth Wave",
        category="Growth Equity",
        benchmark="SPYG",
        target_beta=1.05,
        risk_band="Core",
        use_smartsafe=True,
    ),
    "Infinity Wave": WaveRecipe(
        name="Infinity Wave",
        category="Flagship Multi-Theme",
        benchmark="ACWI",
        target_beta=1.10,
        risk_band="Flagship",
        use_smartsafe=True,
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
        use_smartsafe=True,
    ),
    "Quantum Computing Wave": WaveRecipe(
        name="Quantum Computing Wave",
        category="Thematic Equity",
        benchmark="QQQ",
        target_beta=1.30,
        risk_band="Thematic",
        use_smartsafe=True,
    ),
    "S&P 500 Wave": WaveRecipe(
        name="S&P 500 Wave",
        category="Core Equity",
        benchmark="SPY",
        target_beta=0.90,   # disciplined beta < 1.0
        risk_band="Core",
        use_smartsafe=True,
        mode_scalars={
            "Standard": 1.00,
            "AlphaMinusBeta": 0.80,
            "PrivateLogic": 1.05,
        },
    ),
    "Small Cap Growth Wave": WaveRecipe(
        name="Small Cap Growth Wave",
        category="Small Cap Growth",
        benchmark="IWO",
        target_beta=1.20,
        risk_band="Growth",
        use_smartsafe=True,
    ),
    "SmartSafe Wave": WaveRecipe(
        name="SmartSafe Wave",
        category="SmartSafe / Cash",
        benchmark="BIL",    # T-Bill proxy
        target_beta=0.05,
        risk_band="SmartSafe",
        use_smartsafe=False,  # SmartSafe is the destination, not user
        mode_scalars={
            "Standard": 0.05,
            "AlphaMinusBeta": 0.05,
            "PrivateLogic": 0.05,
        },
    ),
}


def list_wave_names(include_smartsafe: bool = True) -> List[str]:
    names = list(WAVE_RECIPES.keys())
    if not include_smartsafe:
        names = [n for n in names if n != "SmartSafe Wave"]
    return names


# -----------------------------------------------------------------------------
# VIX ladder & SmartSafe allocation
# -----------------------------------------------------------------------------


def fetch_vix_history(days: int = 365) -> Optional[pd.Series]:
    """Fetch VIX daily close series; returns None if yfinance is unavailable."""
    if yf is None:
        return None
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 5)
    try:
        data = yf.download(VIX_TICKER, start=start, end=end, progress=False)
        if data.empty:
            return None
        s = data["Adj Close"].copy()
        s.index = s.index.tz_localize(None)
        s.name = "VIX"
        return s
    except Exception:
        return None


def latest_vix_level(vix_series: Optional[pd.Series] = None) -> float:
    if vix_series is None:
        vix_series = fetch_vix_history(60)
    if vix_series is None or vix_series.empty:
        # Fallback approximate "calm" regime
        return 18.0
    return float(vix_series.iloc[-1])


def smartsafe_allocation_from_vix(vix: float) -> float:
    """
    VIX ladder -> SmartSafe allocation (0..1).
    Institutional-ish mapping:
        VIX < 16   -> 5%
        16-20      -> 10%
        20-25      -> 20%
        25-30      -> 35%
        30-40      -> 55%
        > 40       -> 75%
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


def effective_equity_exposure(recipe: WaveRecipe, mode: str, vix_level: Optional[float] = None) -> Tuple[float, float]:
    """
    Compute effective equity exposure and SmartSafe allocation (0..1) for a wave
    under a given mode and VIX level.

    Returns: (equity_exposure, smartsafe_allocation)
    """
    if vix_level is None:
        vix_level = latest_vix_level()

    scalar = recipe.scalar_for_mode(mode)
    base_beta = recipe.target_beta * scalar

    ss_alloc = smartsafe_allocation_from_vix(vix_level) if recipe.use_smartsafe else 0.0
    ss_alloc = max(0.0, min(1.0, ss_alloc))

    # Equity exposure scaled down by SmartSafe slice
    equity_exposure = base_beta * (1.0 - ss_alloc)

    return float(equity_exposure), float(ss_alloc)


# -----------------------------------------------------------------------------
# Performance history loaders
# -----------------------------------------------------------------------------


def _latest_matching_file(pattern: str) -> Optional[Path]:
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return Path(files[-1])


def load_wave_performance_history(wave_name: str) -> pd.DataFrame:
    """
    Load daily performance history for a wave from logs/performance.

    Expected CSV columns (flexible, but typical):
    - date
    - return (daily pct or decimal)
    - benchmark_return (optional)
    """
    safe_name = wave_name.replace(" ", "_").replace("&", "and")
    pattern = str(LOG_PERFORMANCE_DIR / f"{safe_name}_performance_daily*.csv")
    latest = _latest_matching_file(pattern)
    if latest is None or not latest.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(latest)
    except Exception:
        return pd.DataFrame()

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Expect at least a date and a return column
    if "date" not in df.columns:
        # try index or other date-like columns
        for alt in ["as_of", "timestamp"]:
            if alt in df.columns:
                df["date"] = df[alt]
                break
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # normalize return column
    if "return" not in df.columns:
        for alt in ["daily_return", "wave_return", "portfolio_return"]:
            if alt in df.columns:
                df["return"] = df[alt]
                break

    if "return" not in df.columns:
        return pd.DataFrame()

    # Ensure returns are decimals, not percents
    if df["return"].abs().mean() > 0.5:  # crude check
        df["return"] = df["return"] / 100.0

    # Keep relevant columns
    keep_cols = ["date", "return"]
    if "benchmark_return" in df.columns:
        keep_cols.append("benchmark_return")
        if df["benchmark_return"].abs().mean() > 0.5:
            df["benchmark_return"] = df["benchmark_return"] / 100.0

    df = df[keep_cols].dropna(subset=["date", "return"]).sort_values("date")

    return df.reset_index(drop=True)


def fetch_benchmark_returns(ticker: str, dates: pd.Series) -> pd.Series:
    """
    Fetch benchmark total returns aligned to given dates.
    If yfinance is unavailable, returns zeros.
    """
    if yf is None or dates.empty:
        return pd.Series(0.0, index=dates, name="benchmark_return")

    start = dates.min() - pd.Timedelta(days=5)
    end = dates.max() + pd.Timedelta(days=2)

    try:
        px = yf.download(ticker, start=start, end=end, progress=False)["Adj Close"]
        px = px.tz_localize(None)
        px = px.reindex(sorted(px.index)).ffill()
        rets = px.pct_change().fillna(0.0)
        rets.name = "benchmark_return"
        # align to our dates
        aligned = rets.reindex(dates, method="ffill").fillna(0.0)
        aligned.index = dates
        return aligned
    except Exception:
        return pd.Series(0.0, index=dates, name="benchmark_return")


# -----------------------------------------------------------------------------
# Return / alpha utilities
# -----------------------------------------------------------------------------


def _horizon_total_return(returns: pd.Series, days: int) -> float:
    if returns.empty:
        return float("nan")
    # last N trading days
    r = returns.iloc[-days:].copy()
    if r.empty:
        return float("nan")
    total = (1.0 + r).prod() - 1.0
    return float(total)


def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def compute_horizon_metrics(
    wave_returns: pd.Series,
    benchmark_returns: pd.Series,
    horizons: List[int] = [1, 30, 60, 252],
) -> Dict[str, float]:
    """
    Compute horizon total returns and alpha for given horizons (in trading days).
    Keys:
        Return_1d, Return_30d, Return_60d, Return_1y
        Alpha_1d, Alpha_30d, Alpha_60d, Alpha_1y
    """
    wave_returns = _ensure_series(wave_returns).fillna(0.0)
    benchmark_returns = _ensure_series(benchmark_returns).fillna(0.0)
    # align
    df = pd.DataFrame({"wave": wave_returns, "bench": benchmark_returns}).fillna(0.0)

    out: Dict[str, float] = {}

    for d in horizons:
        label = "1d" if d == 1 else ("1y" if d >= 250 else f"{d}d")
        w_total = _horizon_total_return(df["wave"], d)
        b_total = _horizon_total_return(df["bench"], d)
        alpha = w_total - b_total
        out[f"Return_{label}"] = w_total
        out[f"Alpha_{label}"] = alpha

    return out


def compute_alpha_capture_for_wave(
    wave_name: str,
    mode: str = "Standard",
    vix_series: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Core engine function:
    - loads wave performance history
    - fetches / derives benchmark returns
    - applies mode + SmartSafe exposure adjustments
    - computes horizon total returns and alpha
    """
    if mode not in MODES:
        mode = "Standard"

    recipe = WAVE_RECIPES.get(wave_name)
    if recipe is None:
        return {}

    perf = load_wave_performance_history(wave_name)
    if perf.empty:
        return {}

    perf = perf.copy().sort_values("date")
    perf["return"] = perf["return"].astype(float).fillna(0.0)

    # Get benchmark series: either from file or via yfinance
    if "benchmark_return" in perf.columns:
        bench = perf["benchmark_return"].astype(float).fillna(0.0)
    else:
        bench = fetch_benchmark_returns(recipe.benchmark, perf["date"])

    # Exposure + SmartSafe adjustment
    equity_exp, ss_alloc = effective_equity_exposure(
        recipe, mode, latest_vix_level(vix_series)
    )

    # Interpret recorded returns as *pre*-mode scaling equity returns.
    # We re-scale them by equity exposure; SmartSafe assumed ~0 return.
    adjusted_wave_returns = perf["return"] * equity_exp

    metrics = compute_horizon_metrics(
        wave_returns=adjusted_wave_returns,
        benchmark_returns=bench,
        horizons=[1, 30, 60, 252],
    )

    # Optional: simple information ratio over 1y
    df = pd.DataFrame({"wave": adjusted_wave_returns, "bench": bench})
    df["excess"] = df["wave"] - df["bench"]
    if len(df) > 20:
        mean_excess = df["excess"].mean()
        std_excess = df["excess"].std(ddof=1)
        ir = mean_excess / std_excess * math.sqrt(252) if std_excess > 0 else float("nan")
    else:
        ir = float("nan")

    metrics.update(
        {
            "Wave": wave_name,
            "Category": recipe.category,
            "Benchmark": recipe.benchmark,
            "Mode": mode,
            "Target_Beta": recipe.target_beta,
            "Equity_Exposure": equity_exp,
            "SmartSafe_Alloc": ss_alloc,
            "Alpha_IR": ir,
        }
    )
    return metrics


# -----------------------------------------------------------------------------
# Public API for the console
# -----------------------------------------------------------------------------


def compute_alpha_capture_matrix(mode: str = "Standard") -> pd.DataFrame:
    """
    Compute alpha capture for all waves in a given mode.
    Output columns (for each wave):
        Wave, Category, Benchmark,
        Return_1d, Return_30d, Return_60d, Return_1y,
        Alpha_1d, Alpha_30d, Alpha_60d, Alpha_1y,
        Equity_Exposure, SmartSafe_Alloc, Alpha_IR
    """
    rows: List[Dict[str, float]] = []
    vix_hist = fetch_vix_history(90)

    for wave in list_wave_names(include_smartsafe=True):
        row = compute_alpha_capture_for_wave(wave, mode=mode, vix_series=vix_hist)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort: put SmartSafe last
    df["_sort"] = df["Wave"].apply(lambda w: 999 if "SmartSafe" in w else 0)
    df = df.sort_values(["_sort", "Wave"]).drop(columns=["_sort"]).reset_index(drop=True)

    # Pretty percentages for UI can be done in app.py; here we keep decimals.
    return df


def get_engine_status() -> Dict[str, str]:
    """
    Lightweight status summary for the System Status tab.
    """
    status = {}

    # Check directories
    status["positions_dir_exists"] = str(LOG_POSITIONS_DIR.exists())
    status["performance_dir_exists"] = str(LOG_PERFORMANCE_DIR.exists())

    # Count CSVs
    status["positions_csv_count"] = str(len(glob.glob(str(LOG_POSITIONS_DIR / "*.csv"))))
    status["performance_csv_count"] = str(len(glob.glob(str(LOG_PERFORMANCE_DIR / "*.csv"))))

    # VIX
    vix_hist = fetch_vix_history(30)
    if vix_hist is None or vix_hist.empty:
        status["vix_status"] = "UNAVAILABLE"
        status["vix_latest"] = "N/A"
    else:
        status["vix_status"] = "OK"
        status["vix_latest"] = f"{latest_vix_level(vix_hist):.2f}"

    return status


def list_waves_with_data() -> pd.DataFrame:
    """
    Helper for System Status — shows which waves actually have performance logs.
    """
    rows = []
    for wave in list_wave_names(include_smartsafe=True):
        perf = load_wave_performance_history(wave)
        if perf.empty:
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
                    "First_Date": perf["date"].min(),
                    "Last_Date": perf["date"].max(),
                    "Row_Count": len(perf),
                }
            )
    return pd.DataFrame(rows)


# You can add a "run_engine_once" here later to generate fresh logs from
# list.csv + wave_weights.csv if you want the app to trigger a full rebalance.

__all__ = [
    "WaveRecipe",
    "WAVE_RECIPES",
    "list_wave_names",
    "fetch_vix_history",
    "latest_vix_level",
    "smartsafe_allocation_from_vix",
    "effective_equity_exposure",
    "load_wave_performance_history",
    "fetch_benchmark_returns",
    "compute_horizon_metrics",
    "compute_alpha_capture_for_wave",
    "compute_alpha_capture_matrix",
    "get_engine_status",
    "list_waves_with_data",
]