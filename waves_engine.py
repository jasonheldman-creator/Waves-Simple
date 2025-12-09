"""
waves_engine.py — Real-History Engine (Vector1)

- Loads wave_weights.csv (+ list.csv for names)
- Maps each Wave to a benchmark ETF (SPY, QQQ, etc.)
- Fetches real price history via yfinance
- Computes daily, 30d, 60d, 1y returns and alpha vs benchmark
- Writes logs:
    logs/performance/<Wave>_performance_daily.csv
    logs/positions/<Wave>_positions_YYYYMMDD.csv
- Provides:
    load_wave_weights()
    get_strategy_recipe()
    run_all_waves()
"""

import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------- Optional yfinance import (real prices) ----------

try:
    import yfinance as yf  # type: ignore

    HAS_YF = True
except Exception:
    HAS_YF = False

# ---------- Paths ----------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
LOGS_PERFORMANCE_DIR = os.path.join(LOGS_DIR, "performance")
LOGS_POSITIONS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_MARKET_DIR = os.path.join(LOGS_DIR, "market")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)
os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
os.makedirs(LOGS_MARKET_DIR, exist_ok=True)

# ---------- Wave lineup & benchmarks ----------

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

# Mapping from Wave name to benchmark ticker
WAVE_BENCHMARKS: Dict[str, str] = {
    "S&P 500 Wave": "SPY",
    "Growth Wave": "QQQ",
    "Small Cap Growth Wave": "IWM",
    "Small to Mid Cap Growth Wave": "IJH",
    "Future Power & Energy Wave": "XLE",
    "Quantum Computing Wave": "QQQ",
    "Clean Transit-Infrastructure Wave": "IDEV",  # or ITA/IFRA/XTN etc.
    "AI Wave": "QQQ",  # proxy AI-heavy index
    "Infinity Wave": "ACWI",
    "International Developed Wave": "EFA",
    "Emerging Markets Wave": "EEM",
}

# ---------- Strategy recipes (used by UI; not critical to math) ----------

STRATEGY_RECIPES: Dict[str, Dict[str, float]] = {
    "S&P 500 Wave": {
        "style": "Core Equity",
        "universe": "S&P 500",
        "target_beta": 1.00,
        "alpha_mu_annual": 0.01,
        "alpha_sigma_annual": 0.04,
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
    """Used by the app sidebar for the 'Strategy Snapshot' section."""
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


# ---------- Wave weights & list.csv ----------


def load_wave_weights() -> pd.DataFrame:
    """
    Load wave_weights.csv (required for real engine).
    Expected columns: wave, ticker, weight (case-insensitive).
    """
    path = os.path.join(ROOT_DIR, "wave_weights.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"wave_weights.csv not found at {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("wave_weights.csv is empty")

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    wave_col = cols.get("wave")
    ticker_col = cols.get("ticker")
    weight_col = cols.get("weight")

    if wave_col is None or ticker_col is None or weight_col is None:
        raise ValueError(
            "wave_weights.csv must contain columns 'wave', 'ticker', 'weight'"
        )

    df = df[[wave_col, ticker_col, weight_col]].rename(
        columns={wave_col: "wave", ticker_col: "ticker", weight_col: "weight"}
    )
    df["wave"] = df["wave"].astype(str)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["weight"] = df["weight"].astype(float)

    # Keep only known equity waves (if other Waves exist in the file)
    df = df[df["wave"].isin(EQUITY_WAVES)]

    # Normalize weights within each wave
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )

    return df


def load_master_list() -> Optional[pd.DataFrame]:
    """
    Optional: list.csv (for security names/sectors).
    Expected columns: ticker, name, sector (flexible).
    """
    candidate = os.path.join(ROOT_DIR, "list.csv")
    if not os.path.exists(candidate):
        return None
    try:
        df = pd.read_csv(candidate)
        if df.empty:
            return None
        cols = {c.lower(): c for c in df.columns}
        ticker_col = cols.get("ticker")
        if ticker_col is None:
            return None
        df = df.rename(columns={ticker_col: "ticker"})
        df["ticker"] = df["ticker"].astype(str).str.upper()
        return df
    except Exception:
        return None


# ---------- Price history (via yfinance) ----------


def get_price_history(
    tickers: List[str],
    years: int = 3,
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers over the last `years` years.
    """
    if not HAS_YF:
        raise RuntimeError("yfinance not installed in this environment.")

    tickers = sorted(set(tickers))
    if not tickers:
        raise ValueError("No tickers provided for get_price_history()")

    end = datetime.today().date()
    start = end - timedelta(days=365 * years + 30)

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )

    # Normalize into a DataFrame: index = date, columns = tickers
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
    else:
        px = data

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    # Ensure columns match tickers order
    px = px[tickers].copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    return px


# ---------- NAV & alpha calculations ----------


def compute_wave_nav(
    price_panel: pd.DataFrame,
    weights_df: pd.DataFrame,
    wave_name: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Given a price panel (date x ticker) and this wave's weights,
    compute Wave NAV and benchmark NAV, plus returns & alpha.
    """
    this_wave = weights_df[weights_df["wave"] == wave_name].copy()
    if this_wave.empty:
        raise ValueError(f"No weights found for wave '{wave_name}'")

    tickers = this_wave["ticker"].tolist()
    w = this_wave["weight"].values

    # Ensure all tickers exist in price_panel
    missing = [t for t in tickers if t not in price_panel.columns]
    if missing:
        raise ValueError(f"Missing price history for tickers: {missing}")

    prices = price_panel[tickers].copy()
    # Wave "price" as weighted sum of component prices
    wave_price = (prices * w).sum(axis=1)

    # Benchmark mapping
    bench_ticker = WAVE_BENCHMARKS.get(wave_name, "SPY")
    if bench_ticker not in price_panel.columns:
        raise ValueError(f"Missing price history for benchmark {bench_ticker}")

    bench_price = price_panel[bench_ticker].copy()

    # Align indexes (drop dates where either side is NaN)
    df = pd.DataFrame({"wave_px": wave_price, "bench_px": bench_price})
    df = df.dropna().copy()
    df.index.name = "date"

    # Turn into NAV (100 starting index)
    df["nav"] = df["wave_px"] / df["wave_px"].iloc[0] * 100.0
    df["bench_nav"] = df["bench_px"] / df["bench_px"].iloc[0] * 100.0

    # Daily returns
    df["return_1d"] = df["nav"].pct_change()
    df["bench_return_1d"] = df["bench_nav"].pct_change()

    # Horizon returns (21 ≈ 30d, 42 ≈ 60d, 252 ≈ 1y)
    horizons = [(21, "30d"), (42, "60d"), (252, "252d")]

    for h, label in horizons:
        df[f"return_{label}"] = df["nav"] / df["nav"].shift(h) - 1.0
        df[f"bench_return_{label}"] = (
            df["bench_nav"] / df["bench_nav"].shift(h) - 1.0
        )

    # Alphas
    df["alpha_1d"] = df["return_1d"] - df["bench_return_1d"]
    df["alpha_30d"] = df["return_30d"] - df["bench_return_30d"]
    df["alpha_60d"] = df["return_60d"] - df["bench_return_60d"]
    df["alpha_1y"] = df["return_252d"] - df["bench_return_252d"]

    df["wave"] = wave_name
    df["benchmark_ticker"] = bench_ticker
    df["regime"] = "LIVE_PRICES" if HAS_YF else "SANDBOX"

    return df.reset_index(), bench_ticker


# ---------- Positions snapshot ----------


def build_positions_snapshot(
    weights_df: pd.DataFrame, master_list: Optional[pd.DataFrame], wave_name: str
) -> pd.DataFrame:
    this_wave = weights_df[weights_df["wave"] == wave_name].copy()
    if this_wave.empty:
        raise ValueError(f"No weights found for wave '{wave_name}'")

    this_wave["wave"] = wave_name

    if master_list is not None:
        merged = this_wave.merge(
            master_list, on="ticker", how="left", suffixes=("", "_list")
        )
        # Prefer 'name' column from list.csv if it exists
        cols = {c.lower(): c for c in merged.columns}
        name_col = cols.get("name")
        if name_col:
            merged = merged.rename(columns={name_col: "name"})
        else:
            merged["name"] = merged["ticker"]
        # Keep a clean subset
        out = merged[["wave", "ticker", "name", "weight"]].copy()
    else:
        out = this_wave.copy()
        out["name"] = out["ticker"]
        out = out[["wave", "ticker", "name", "weight"]]

    return out.sort_values("weight", ascending=False)


# ---------- Log writers ----------


def write_performance_log(df: pd.DataFrame, wave_name: str) -> str:
    prefix = wave_name.replace(" ", "_")
    path = os.path.join(LOGS_PERFORMANCE_DIR, f"{prefix}_performance_daily.csv")
    df.to_csv(path, index=False)
    return path


def write_positions_log(df: pd.DataFrame, wave_name: str) -> str:
    today_str = datetime.today().strftime("%Y%m%d")
    prefix = wave_name.replace(" ", "_")
    path = os.path.join(LOGS_POSITIONS_DIR, f"{prefix}_positions_{today_str}.csv")
    df.to_csv(path, index=False)
    return path


# ---------- Public engine API ----------


def run_wave_update(
    wave_name: str,
    price_panel: Optional[pd.DataFrame] = None,
    years: int = 3,
) -> Tuple[str, str]:
    """
    Generate performance + positions logs for a single Wave.

    If price_panel is provided, it must contain all component tickers
    and the appropriate benchmark ticker. Otherwise, we fetch prices.
    """
    weights_df = load_wave_weights()
    master_list = load_master_list()

    this_wave = weights_df[weights_df["wave"] == wave_name]
    if this_wave.empty:
        raise ValueError(f"Wave '{wave_name}' not present in wave_weights.csv")

    all_wave_tickers = weights_df["ticker"].unique().tolist()
    bench_ticker = WAVE_BENCHMARKS.get(wave_name, "SPY")

    if price_panel is None:
        tickers_needed = list(set(all_wave_tickers + [bench_ticker]))
        price_panel = get_price_history(tickers_needed, years=years)

    perf_df, _ = compute_wave_nav(price_panel, weights_df, wave_name)
    pos_df = build_positions_snapshot(weights_df, master_list, wave_name)

    perf_path = write_performance_log(perf_df, wave_name)
    pos_path = write_positions_log(pos_df, wave_name)
    return perf_path, pos_path


def run_all_waves(years: int = 3) -> None:
    """
    Main entrypoint: recompute performance logs for all EQUITY_WAVES.

    - Reads wave_weights.csv (and list.csv if available)
    - Downloads price history for all tickers + all benchmarks
    - Writes performance & positions logs for each wave
    """
    if not HAS_YF:
        raise RuntimeError(
            "yfinance is not available. Install it in this environment "
            "to use real price history."
        )

    weights_df = load_wave_weights()
    master_list = load_master_list()

    all_wave_tickers = weights_df["ticker"].unique().tolist()
    bench_tickers = list(set(WAVE_BENCHMARKS.values()))
    all_tickers = sorted(set(all_wave_tickers + bench_tickers))

    print(f"[waves_engine] Fetching prices for {len(all_tickers)} tickers...")
    price_panel = get_price_history(all_tickers, years=years)
    print(f"[waves_engine] Price panel shape: {price_panel.shape}")

    for wave_name in EQUITY_WAVES:
        if wave_name not in weights_df["wave"].unique():
            print(f"[waves_engine] Skipping {wave_name} (no weights).")
            continue
        try:
            perf_df, bench = compute_wave_nav(price_panel, weights_df, wave_name)
            pos_df = build_positions_snapshot(weights_df, master_list, wave_name)
            perf_path = write_performance_log(perf_df, wave_name)
            pos_path = write_positions_log(pos_df, wave_name)
            print(f"[waves_engine] Updated {wave_name} (benchmark: {bench})")
            print(f"  Performance: {perf_path}")
            print(f"  Positions:   {pos_path}")
        except Exception as exc:
            print(f"[waves_engine] ERROR on {wave_name}: {exc}")


# ---------- Script entrypoint ----------

if __name__ == "__main__":
    print("Running WAVES Intelligence™ real-history engine (Vector1)...")
    if not HAS_YF:
        print(
            "ERROR: yfinance not installed. Install it in this environment "
            "to fetch real prices."
        )
    else:
        run_all_waves(years=3)
    print("Done.")