# waves_engine.py
"""
WAVES Intelligence™ Engine — Real History + SmartSafe™ (Vector 1.6)

Key features
------------
- Loads wave_weights.csv (+ optional list.csv for names/sectors)
- 11 equity Waves with benchmark mappings
- Fetches real price history via yfinance (if available)
- Computes:
    • risk-only NAV (nav_risk)
    • SmartSafe-adjusted NAV (nav)
    • benchmark NAV (bench_nav)
    • returns & alpha horizons: 30d / 60d / 1y
- SmartSafe™ overlay:
    • gentle dynamic allocation based on volatility & drawdown
    • SmartSafe earns a stable yield (cash-like)
    • nav (client experience) includes SmartSafe
    • alpha_* is computed from nav_risk vs benchmark, so
      SmartSafe doesn’t crush the measured alpha
- Writes logs:
    logs/performance/<Wave>_performance_daily.csv
    logs/positions/<Wave>_positions_YYYYMMDD.csv

Public API
----------
- load_wave_weights()
- get_strategy_recipe(wave_name)
- run_wave_update(wave_name, price_panel=None, years=3)
- run_all_waves(years=3)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional yfinance import
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

WAVE_BENCHMARKS: Dict[str, str] = {
    "S&P 500 Wave": "SPY",
    "Growth Wave": "QQQ",
    "Small Cap Growth Wave": "IWM",
    "Small to Mid Cap Growth Wave": "IJH",
    "Future Power & Energy Wave": "XLE",
    "Quantum Computing Wave": "QQQ",
    "Clean Transit-Infrastructure Wave": "IDEV",  # proxy
    "AI Wave": "QQQ",
    "Infinity Wave": "ACWI",
    "International Developed Wave": "EFA",
    "Emerging Markets Wave": "EEM",
}

# ---------- Strategy recipes (for UI snapshot) ----------

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
        "notes": "High growth with volatility-aware sizing.",
    },
    "Small Cap Growth Wave": {
        "style": "Small Cap Growth",
        "universe": "US Small Growth",
        "target_beta": 1.20,
        "alpha_mu_annual": 0.04,
        "alpha_sigma_annual": 0.14,
        "turnover_annual_max": 1.20,
        "max_drawdown_target": 0.35,
        "notes": "Higher-octane small-cap growth with strict brakes.",
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


# ---------- Data loaders ----------


def load_wave_weights() -> pd.DataFrame:
    """
    Load wave_weights.csv. Required columns (case-insensitive):
    - wave
    - ticker
    - weight
    """
    path = os.path.join(ROOT_DIR, "wave_weights.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"wave_weights.csv not found at {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("wave_weights.csv is empty")

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

    # Only keep our lineup
    df = df[df["wave"].isin(EQUITY_WAVES)]

    # Normalize weights within Wave
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )

    return df


def load_master_list() -> Optional[pd.DataFrame]:
    """
    Optional list.csv with columns like ticker, name, sector.
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


def get_price_history(tickers: List[str], years: int = 3) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers over the last `years` years.
    """
    if not HAS_YF:
        raise RuntimeError("yfinance not installed; cannot fetch prices.")

    tickers = sorted(set(tickers))
    if not tickers:
        raise ValueError("No tickers provided for get_price_history().")

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

    if isinstance(data.columns, pd.MultiIndex):
        px = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
    else:
        px = data

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    px = px[tickers].copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    return px


# ---------- SmartSafe overlay (gentler) ----------


def apply_smartsafe_overlay(
    df: pd.DataFrame,
    smartsafe_yield_annual: float = 0.04,
    vol_window: int = 21,
    vol_low: float = 0.18,   # calmer threshold
    vol_high: float = 0.30,  # high-vol threshold
    dd_low: float = 0.05,
    dd_high: float = 0.12,
    max_smartsafe: float = 0.35,  # cap allocation to SmartSafe
    step: float = 0.05,
) -> pd.DataFrame:
    """
    SmartSafe overlay:
      - Uses realized volatility & drawdown on risk NAV (nav_risk).
      - Moves some weight into a stable-yield SmartSafe sleeve when
        things are rough; moves back out when calm.
      - nav_risk is NOT changed, nav is the blended client NAV.
    """
    out = df.copy().sort_index()

    if "nav_risk" not in out.columns:
        raise ValueError("apply_smartsafe_overlay expects 'nav_risk' in df")

    risk_nav = out["nav_risk"].values.astype(float)
    n = len(out)
    if n == 0:
        return out

    risk_ret = pd.Series(risk_nav).pct_change()
    vol_roll = risk_ret.rolling(vol_window).std() * np.sqrt(252.0)
    running_max = pd.Series(risk_nav).cummax()
    dd = risk_nav / running_max - 1.0

    smartsafe_weight = np.zeros(n)
    sm_nav = np.zeros(n)
    sm_nav[0] = risk_nav[0]

    y_daily = (1.0 + smartsafe_yield_annual) ** (1.0 / 252.0) - 1.0
    w = 0.0  # initial SmartSafe allocation

    for i in range(1, n):
        sm_nav[i] = sm_nav[i - 1] * (1.0 + y_daily)

        v = vol_roll.iloc[i]
        d = dd.iloc[i]

        v_val = float(v) if pd.notnull(v) else float("nan")
        d_val = float(d) if pd.notnull(d) else float("nan")

        if not np.isnan(v_val) and not np.isnan(d_val):
            if v_val > vol_high or d_val < -dd_high:
                # risk-off → add SmartSafe
                w = min(max_smartsafe, w + step)
            elif v_val < vol_low and d_val > -dd_low:
                # calm + shallow drawdown → reduce SmartSafe
                w = max(0.0, w - step)

        smartsafe_weight[i] = w

    smartsafe_weight[0] = smartsafe_weight[1] if n > 1 else w

    combined_nav = smartsafe_weight * sm_nav + (1.0 - smartsafe_weight) * risk_nav

    out["smartsafe_nav"] = sm_nav
    out["smartsafe_weight"] = smartsafe_weight
    out["smartsafe_yield_annual"] = smartsafe_yield_annual
    out["nav"] = combined_nav

    return out


# ---------- NAV & alpha (alpha from risk-only) ----------


def compute_wave_nav(
    price_panel: pd.DataFrame,
    weights_df: pd.DataFrame,
    wave_name: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Compute NAV and alpha for a single Wave:

      • nav_risk      : risk-only NAV from holdings
      • nav           : SmartSafe-adjusted NAV (client experience)
      • bench_nav     : benchmark NAV
      • return_*      : returns on nav (client experience)
      • alpha_*       : alpha from nav_risk vs benchmark
    """
    this_wave = weights_df[weights_df["wave"] == wave_name].copy()
    if this_wave.empty:
        raise ValueError(f"No weights found for wave '{wave_name}'")

    tickers = this_wave["ticker"].tolist()
    w = this_wave["weight"].values

    missing = [t for t in tickers if t not in price_panel.columns]
    if missing:
        raise ValueError(f"Missing price history for tickers: {missing}")

    prices = price_panel[tickers].copy()
    wave_px = (prices * w).sum(axis=1)

    bench_ticker = WAVE_BENCHMARKS.get(wave_name, "SPY")
    if bench_ticker not in price_panel.columns:
        raise ValueError(f"Missing price history for benchmark {bench_ticker}")
    bench_price = price_panel[bench_ticker].copy()

    df = pd.DataFrame({"wave_px": wave_px, "bench_px": bench_price}).dropna()
    df.index.name = "date"

    # Risk-only and benchmark NAV
    df["nav_risk"] = df["wave_px"] / df["wave_px"].iloc[0] * 100.0
    df["bench_nav"] = df["bench_px"] / df["bench_px"].iloc[0] * 100.0

    # SmartSafe overlay -> nav (client experience)
    df = apply_smartsafe_overlay(df)

    # Client-experience returns (SmartSafe-adjusted)
    df["return_1d"] = df["nav"].pct_change()
    df["bench_return_1d"] = df["bench_nav"].pct_change()

    horizons = [(21, "30d"), (42, "60d"), (252, "252d")]
    for h, label in horizons:
        df[f"return_{label}"] = df["nav"] / df["nav"].shift(h) - 1.0
        df[f"bench_return_{label}"] = df["bench_nav"] / df["bench_nav"].shift(h) - 1.0

    # Risk-sleeve returns (for alpha)
    df["risk_return_1d"] = df["nav_risk"].pct_change()
    df["risk_return_30d"] = df["nav_risk"] / df["nav_risk"].shift(21) - 1.0
    df["risk_return_60d"] = df["nav_risk"] / df["nav_risk"].shift(42) - 1.0
    df["risk_return_252d"] = df["nav_risk"] / df["nav_risk"].shift(252) - 1.0

    # Alpha from risk-only vs benchmark
    df["alpha_1d"] = df["risk_return_1d"] - df["bench_return_1d"]
    df["alpha_30d"] = df["risk_return_30d"] - df["bench_return_30d"]
    df["alpha_60d"] = df["risk_return_60d"] - df["bench_return_60d"]
    df["alpha_1y"] = df["risk_return_252d"] - df["bench_return_252d"]

    df["wave"] = wave_name
    df["benchmark_ticker"] = bench_ticker
    df["regime"] = "LIVE_SMARTSAFE" if HAS_YF else "SANDBOX_SMARTSAFE"

    return df.reset_index(), bench_ticker


# ---------- Positions snapshot & logging ----------


def build_positions_snapshot(
    weights_df: pd.DataFrame,
    master_list: Optional[pd.DataFrame],
    wave_name: str,
) -> pd.DataFrame:
    this_wave = weights_df[weights_df["wave"] == wave_name].copy()
    if this_wave.empty:
        raise ValueError(f"No weights found for wave '{wave_name}'")

    this_wave["wave"] = wave_name

    if master_list is not None:
        merged = this_wave.merge(
            master_list, on="ticker", how="left", suffixes=("", "_list")
        )
        cols = {c.lower(): c for c in merged.columns}
        name_col = cols.get("name")
        if name_col:
            merged = merged.rename(columns={name_col: "name"})
        else:
            merged["name"] = merged["ticker"]
        out = merged[["wave", "ticker", "name", "weight"]].copy()
    else:
        out = this_wave.copy()
        out["name"] = out["ticker"]
        out = out[["wave", "ticker", "name", "weight"]]

    return out.sort_values("weight", ascending=False)


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
    Recompute performance logs for all EQUITY_WAVES.
    """
    if not HAS_YF:
        raise RuntimeError(
            "yfinance is not available. Install it to fetch real prices."
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


# ---------- Script entry ----------

if __name__ == "__main__":
    print("Running WAVES Intelligence™ engine with SmartSafe™ (Vector 1.6)...")
    if not HAS_YF:
        print("ERROR: yfinance not installed. Install it to fetch prices.")
    else:
        run_all_waves(years=3)
    print("Done.")