"""
waves_engine.py — Vector1 Stub Engine

This file:
- Defines per-wave strategy recipes
- Implements a VIX ladder / risk-off overlay
- Generates SANDBOX performance & positions logs for all Waves
- Provides load_wave_weights() for compatibility with the app
"""

import os
from datetime import datetime, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ----------------- Paths -----------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
LOGS_PERFORMANCE_DIR = os.path.join(LOGS_DIR, "performance")
LOGS_POSITIONS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_MARKET_DIR = os.path.join(LOGS_DIR, "market")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)
os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
os.makedirs(LOGS_MARKET_DIR, exist_ok=True)

# ----------------- Wave lineup -----------------

EQUITY_WAVES: List[str] = [
    "S&P 500 Wave",
    "Growth Wave",
    "Small Cap Growth Wave",
    "Small to Mid Cap Growth Wave",
    "Future Power & Energy Wave",
    "Quantum Computing Wave",
    "Clean Transit-Infrastructure Wave",
    "AI Wave",
    "Infinity Wave",
    "International Developed Wave",
    "Emerging Markets Wave",
]

# ----------------- Strategy recipes -----------------

STRATEGY_RECIPES: Dict[str, Dict[str, float]] = {
    "S&P 500 Wave": {
        "style": "Core Equity",
        "universe": "S&P 500",
        "target_beta": 1.00,
        "alpha_mu_annual": 0.01,      # +1% alpha / yr
        "alpha_sigma_annual": 0.04,   # mild tracking error
        "turnover_annual_max": 0.30,
        "max_drawdown_target": 0.20,
        "notes": "Low-turnover core with mild tilts.",
    },
    "Growth Wave": {
        "style": "US Growth",
        "universe": "Large/Mid Growth",
        "target_beta": 1.10,
        "alpha_mu_annual": 0.03,
        "alpha_sigma_annual": 0.10,
        "turnover_annual_max": 0.80,
        "max_drawdown_target": 0.30,
        "notes": "High growth focus with volatility-aware sizing.",
    },
    "Small Cap Growth Wave": {
        "style": "Small Cap Growth",
        "universe": "US Small Growth",
        "target_beta": 1.20,
        "alpha_mu_annual": 0.04,
        "alpha_sigma_annual": 0.14,
        "turnover_annual_max": 1.20,
        "max_drawdown_target": 0.35,
        "notes": "Higher-octane small-cap alpha with strict brakes.",
    },
    "Small to Mid Cap Growth Wave": {
        "style": "SMID Growth",
        "universe": "US Small+Mid",
        "target_beta": 1.10,
        "alpha_mu_annual": 0.03,
        "alpha_sigma_annual": 0.11,
        "turnover_annual_max": 0.90,
        "max_drawdown_target": 0.32,
        "notes": "Blends small and mid cap for smoother growth ride.",
    },
    "Future Power & Energy Wave": {
        "style": "Thematic Energy",
        "universe": "Energy + Renewables",
        "target_beta": 1.15,
        "alpha_mu_annual": 0.04,
        "alpha_sigma_annual": 0.12,
        "turnover_annual_max": 1.00,
        "max_drawdown_target": 0.35,
        "notes": "Energy transition, infra, and power grid plays.",
    },
    "Quantum Computing Wave": {
        "style": "Deep-Tech / Quantum",
        "universe": "AI + Semi + Quantum",
        "target_beta": 1.25,
        "alpha_mu_annual": 0.05,
        "alpha_sigma_annual": 0.18,
        "turnover_annual_max": 1.50,
        "max_drawdown_target": 0.40,
        "notes": "Concentrated deep-tech themes with risk brakes.",
    },
    "Clean Transit-Infrastructure Wave": {
        "style": "Clean Transit & Infra",
        "universe": "EVs, rail, infra",
        "target_beta": 1.10,
        "alpha_mu_annual": 0.04,
        "alpha_sigma_annual": 0.13,
        "turnover_annual_max": 1.00,
        "max_drawdown_target": 0.35,
        "notes": "Mobility + infrastructure with quality bias.",
    },
    "AI Wave": {
        "style": "AI Flagship",
        "universe": "AI leaders + stack",
        "target_beta": 1.20,
        "alpha_mu_annual": 0.05,
        "alpha_sigma_annual": 0.16,
        "turnover_annual_max": 1.40,
        "max_drawdown_target": 0.38,
        "notes": "AI infrastructure and applications, concentrated.",
    },
    "Infinity Wave": {
        "style": "Multi-Theme Flagship",
        "universe": "Cross-asset AI/Tech/Growth",
        "target_beta": 1.05,
        "alpha_mu_annual": 0.04,
        "alpha_sigma_annual": 0.10,
        "turnover_annual_max": 1.20,
        "max_drawdown_target": 0.30,
        "notes": "Adaptive multi-theme flagship (Tesla Roadster Wave).",
    },
    "International Developed Wave": {
        "style": "Intl Developed",
        "universe": "DM ex-US",
        "target_beta": 0.95,
        "alpha_mu_annual": 0.02,
        "alpha_sigma_annual": 0.08,
        "turnover_annual_max": 0.80,
        "max_drawdown_target": 0.28,
        "notes": "DM ex-US with currency and macro overlays.",
    },
    "Emerging Markets Wave": {
        "style": "Emerging Markets",
        "universe": "EM Equity",
        "target_beta": 1.10,
        "alpha_mu_annual": 0.03,
        "alpha_sigma_annual": 0.14,
        "turnover_annual_max": 1.00,
        "max_drawdown_target": 0.35,
        "notes": "EM growth with downside controls and FX awareness.",
    },
}


def get_strategy_recipe(wave_name: str) -> Dict[str, float]:
    """Return strategy config for this wave with safe defaults."""
    default = {
        "style": "Generic Equity",
        "universe": "Global",
        "target_beta": 1.0,
        "alpha_mu_annual": 0.03,
        "alpha_sigma_annual": 0.10,
        "turnover_annual_max": 1.0,
        "max_drawdown_target": 0.30,
        "notes": "",
    }
    cfg = STRATEGY_RECIPES.get(wave_name, {})
    merged = default.copy()
    merged.update(cfg)
    return merged


# ----------------- VIX ladder -----------------

def _simulate_vix_path(dates: pd.DatetimeIndex) -> np.ndarray:
    """Fallback VIX simulation if no real history is present."""
    n = len(dates)
    vix = []
    level = 18.0
    for _ in range(n):
        level = max(10.0, min(50.0, level + np.random.normal(0, 1)))
        vix.append(level)
    return np.array(vix)


def load_or_generate_vix(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Try to load VIX from logs/market/vix_history.csv, otherwise simulate.
    If we simulate, we also write it to the CSV for consistency.
    """
    path = os.path.join(LOGS_MARKET_DIR, "vix_history.csv")
    n = len(dates)

    if os.path.exists(path):
        try:
            vdf = pd.read_csv(path)
            if "date" in vdf.columns and "vix" in vdf.columns:
                vdf["date"] = pd.to_datetime(vdf["date"]).dt.normalize()
                idx = pd.Index(dates.normalize(), name="date")
                merged = (
                    pd.DataFrame({"date": idx})
                    .merge(vdf[["date", "vix"]], on="date", how="left")
                    .ffill()
                    .bfill()
                )
                if merged["vix"].notna().any():
                    return merged["vix"].to_numpy()
        except Exception:
            pass

    # If not found or invalid, simulate and persist
    vix = _simulate_vix_path(dates)
    vdf_new = pd.DataFrame({"date": dates.normalize(), "vix": vix})
    vdf_new.to_csv(path, index=False)
    return vix


def apply_vix_ladder_to_returns(
    bench_mu: float,
    bench_sigma: float,
    alpha_mu: float,
    alpha_sigma: float,
    dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate daily returns for benchmark and alpha component
    scaled by a simple VIX ladder:
      < 18  -> Risk-On   (slightly higher alpha, normal beta)
      18-25 -> Neutral
      > 25  -> Risk-Off  (lower exposure, lower alpha, lower vol)
    """
    n = len(dates)
    vix = load_or_generate_vix(dates)

    risk_mult_alpha = np.ones(n)
    risk_mult_beta = np.ones(n)
    regime: List[str] = []

    for i, v in enumerate(vix):
        if v < 18:
            risk_mult_alpha[i] = 1.10
            risk_mult_beta[i] = 1.00
            regime.append("Risk-On")
        elif v <= 25:
            risk_mult_alpha[i] = 1.00
            risk_mult_beta[i] = 1.00
            regime.append("Neutral")
        else:
            risk_mult_alpha[i] = 0.70
            risk_mult_beta[i] = 0.80
            regime.append("Risk-Off")

    bench_ret = np.random.normal(
        bench_mu * risk_mult_beta,
        bench_sigma * np.abs(risk_mult_beta),
        size=n,
    )
    alpha_noise = np.random.normal(
        alpha_mu * risk_mult_alpha,
        alpha_sigma * np.abs(risk_mult_alpha),
        size=n,
    )
    wave_ret = bench_ret + alpha_noise

    return bench_ret, wave_ret, vix, regime


# ----------------- Performance simulation -----------------

def simulate_wave_path_with_recipe(
    wave_name: str,
    days: int = 260,
) -> pd.DataFrame:
    """
    Generate a SANDBOX performance path for one Wave using:
      - its strategy recipe
      - VIX ladder overlay
    """
    end_date: date = datetime.today().date()
    dates = pd.bdate_range(end=end_date, periods=days)
    n = len(dates)

    cfg = get_strategy_recipe(wave_name)

    # Benchmark process (generic equity index)
    bench_mu_annual = 0.08        # 8%/year
    bench_sigma_annual = 0.15     # 15%/year

    bench_mu = bench_mu_annual / 252.0
    bench_sigma = bench_sigma_annual / np.sqrt(252.0)

    alpha_mu = cfg["alpha_mu_annual"] / 252.0
    alpha_sigma = cfg["alpha_sigma_annual"] / np.sqrt(252.0)

    bench_ret, wave_ret, vix, regime = apply_vix_ladder_to_returns(
        bench_mu, bench_sigma, alpha_mu, alpha_sigma, dates
    )

    bench_nav = 100 * (1 + bench_ret).cumprod()
    wave_nav = 100 * (1 + wave_ret).cumprod()

    df = pd.DataFrame(
        {
            "date": dates,
            "nav": wave_nav,
            "return_1d": wave_ret,
            "bench_nav": bench_nav,
            "bench_return_1d": bench_ret,
            "vix": vix,
            "risk_regime": regime,
        }
    )

    # Horizon returns / alpha
    df["return_30d"] = np.nan
    df["return_60d"] = np.nan
    df["return_252d"] = np.nan
    df["bench_return_30d"] = np.nan
    df["bench_return_60d"] = np.nan
    df["bench_return_252d"] = np.nan

    for i in range(n):
        if i >= 21:
            df.loc[df.index[i], "return_30d"] = wave_nav[i] / wave_nav[i - 21] - 1.0
            df.loc[df.index[i], "bench_return_30d"] = (
                bench_nav[i] / bench_nav[i - 21] - 1.0
            )
        if i >= 42:
            df.loc[df.index[i], "return_60d"] = wave_nav[i] / wave_nav[i - 42] - 1.0
            df.loc[df.index[i], "bench_return_60d"] = (
                bench_nav[i] / bench_nav[i - 42] - 1.0
            )
        if i >= 252:
            df.loc[df.index[i], "return_252d"] = wave_nav[i] / wave_nav[i - 252] - 1.0
            df.loc[df.index[i], "bench_return_252d"] = (
                bench_nav[i] / bench_nav[i - 252] - 1.0
            )

    df["alpha_1d"] = df["return_1d"] - df["bench_return_1d"]
    df["alpha_30d"] = df["return_30d"] - df["bench_return_30d"]
    df["alpha_60d"] = df["return_60d"] - df["bench_return_60d"]
    df["alpha_1y"] = df["return_252d"] - df["bench_return_252d"]

    df["wave"] = wave_name
    df["regime"] = "SANDBOX"
    return df


# ----------------- Positions simulation -----------------

def _sample_tickers_for_wave(wave: str) -> List[str]:
    """Simple universe definition for each wave."""
    if wave == "Clean Transit-Infrastructure Wave":
        return ["TSLA", "NIO", "RIVN", "CHPT", "BLNK", "F", "GM", "CAT", "DE", "UNP"]
    elif wave == "Quantum Computing Wave":
        return ["NVDA", "AMD", "IBM", "QCOM", "AVGO", "TSM", "MSFT", "GOOGL"]
    elif wave == "S&P 500 Wave":
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "JPM", "JNJ", "XOM", "PG"]
    elif wave == "AI Wave":
        return ["NVDA", "MSFT", "GOOGL", "META", "AVGO", "CRM", "SNOW", "ADBE"]
    elif wave == "Future Power & Energy Wave":
        return ["NEE", "ENPH", "FSLR", "XOM", "CVX", "PLUG", "SEDG"]
    elif wave == "Infinity Wave":
        return ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "AVGO"]
    elif wave == "International Developed Wave":
        return ["NOVO-B.CO", "NESN.SW", "ASML", "SONY", "BP", "BHP", "RIO"]
    elif wave == "Emerging Markets Wave":
        return ["TSM", "BABA", "PDD", "INFY", "VALE", "PBR", "MELI"]
    else:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]


def simulate_positions_for_wave(wave: str) -> pd.DataFrame:
    """
    Generate a static positions snapshot for a Wave.
    The app will show the top-10 from this.
    """
    tickers = _sample_tickers_for_wave(wave)
    n = len(tickers)
    raw = np.abs(np.random.rand(n))
    weights = raw / raw.sum()

    df = pd.DataFrame(
        {
            "wave": [wave] * n,
            "ticker": tickers,
            "name": tickers,
            "weight": weights,
        }
    )
    return df


# ----------------- Log writers -----------------

def write_performance_log(df: pd.DataFrame, wave_name: str) -> str:
    prefix = wave_name.replace(" ", "_")
    path = os.path.join(
        LOGS_PERFORMANCE_DIR, f"{prefix}_performance_daily.csv"
    )
    df.to_csv(path, index=False)
    return path


def write_positions_log(df: pd.DataFrame, wave_name: str) -> str:
    today_str = datetime.today().strftime("%Y%m%d")
    prefix = wave_name.replace(" ", "_")
    path = os.path.join(
        LOGS_POSITIONS_DIR, f"{prefix}_positions_{today_str}.csv"
    )
    df.to_csv(path, index=False)
    return path


# ----------------- Public API -----------------

def load_wave_weights() -> pd.DataFrame:
    """
    For compatibility with the app.
    If wave_weights.csv exists, load it.
    Otherwise, synthesize a minimal one listing the Waves and equal weights.
    """
    candidate = os.path.join(ROOT_DIR, "wave_weights.csv")
    if os.path.exists(candidate):
        try:
            df = pd.read_csv(candidate)
            if not df.empty:
                return df
        except Exception:
            pass

    # Fallback synthetic weights
    df = pd.DataFrame(
        {
            "wave": EQUITY_WAVES,
            "weight": [1.0 / len(EQUITY_WAVES)] * len(EQUITY_WAVES),
        }
    )
    return df


def run_wave_update(wave_name: str, days: int = 260) -> Tuple[str, str]:
    """
    Generate SANDBOX performance + positions logs for a single Wave.
    Returns (performance_log_path, positions_log_path).
    """
    perf_df = simulate_wave_path_with_recipe(wave_name, days=days)
    pos_df = simulate_positions_for_wave(wave_name)

    perf_path = write_performance_log(perf_df, wave_name)
    pos_path = write_positions_log(pos_df, wave_name)
    return perf_path, pos_path


def run_all_waves(days: int = 260) -> None:
    """Generate logs for all EQUITY_WAVES."""
    for w in EQUITY_WAVES:
        perf_path, pos_path = run_wave_update(w, days=days)
        print(f"[waves_engine] Updated {w}")
        print(f"  Performance: {perf_path}")
        print(f"  Positions:   {pos_path}")


# ----------------- Script entry -----------------

if __name__ == "__main__":
    print("Running WAVES stub engine (Vector1 SANDBOX)...")
    run_all_waves(days=260)
    print("Done.")