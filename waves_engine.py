"""
waves_engine.py — WAVES Intelligence™ Engine
Stage 7: Adaptive Alpha + Engineered Concentration + SmartSafe 2.0 + SmartSafe 3.0 (LIVE)

Includes:
- Auto-discovered Waves from wave_weights.csv
- Custom blended benchmarks (multi-ETF where specified)
- 3 modes:
    * standard  -> mild momentum tilt + mild concentration, base beta target ~0.90
    * amb       -> very light momentum tilt + beta-leaning defensive + SmartSafe 2.0
    * pl        -> strong momentum tilt + stronger concentration, beta target slightly >1
- SmartSafe 2.0:
    * VIX-based sweep into BIL, trimming highest-vol names first
- SmartSafe 3.0 (LIVE overlay):
    * Uses VIX, 60D return/alpha, max drawdown, and beta drift
    * Classifies regime: Normal / Caution / Stress / Panic
    * Applies extra sweep (0–30%) from highest-vol names into BIL on top of 2.0
- Performance & risk metrics:
    * Intraday return (post 3.0 overlay)
    * 30D, 60D, 1Y, and Since-Inception returns
    * 30D, 60D, 1Y, and Since-Inception alpha
    * 1Y volatility (annualized)
    * Max drawdown (since inception)
    * 1Y information ratio (daily excess Sharpe-style)
    * 1Y hit rate (pct of days with positive excess return)
    * 1Y beta vs benchmark
    * Beta target (per mode & Wave type)
    * Beta drift (actual - target)
    * SmartSafe 2.0 sweep fraction (from VIX ladder)
    * SmartSafe 3.0 regime + extra sweep fraction (LIVE overlay)

Important:
- No Waves are ever equal-weighted; we always respect wave_weights.csv
  and only normalize inside each Wave.
- Historical returns are still based on the pre-3.0 portfolio history.
  SmartSafe 3.0 is a *live* overlay for current exposures going forward.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# Paths / Constants
# ---------------------------------------------------------------------

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR
WEIGHTS_CSV = DATA_DIR / "wave_weights.csv"
UNIVERSE_CSV = DATA_DIR / "list.csv"

LOGS_DIR = BASE_DIR / "logs"
POSITIONS_LOG_DIR = LOGS_DIR / "positions"
PERF_LOG_DIR = LOGS_DIR / "performance"

for d in [LOGS_DIR, POSITIONS_LOG_DIR, PERF_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------

BENCHMARK_MAP: Dict[str, str] = {
    "S&P 500 Wave": "SPY",
    "S&P Wave": "SPY",
    "AI Wave": "QQQ",  # overridden by blended spec
    "AI & Innovation Wave": "QQQ",
    "Quantum Computing Wave": "QQQ",  # overridden
    "Future Power & Energy Wave": "XLE",  # overridden
    "Clean Transit-Infrastructure Wave": "IYT",  # overridden
    "Crypto Income Wave": "BITO",  # overridden
    "Crypto Equity Wave": "BITO",  # overridden
    "Income Wave": "SCHD",
    "Small Cap Growth Wave": "IWM",  # overridden
    "Small to Mid Cap Growth Wave": "VO",
    "Cloud & Software Wave": "IGV",  # overridden
    "SmartSafe Money Market Wave": "BIL",
}

# Custom blended benchmark specs (locked)
BLENDED_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "AI Wave": {"SMH": 0.50, "AIQ": 0.50},
    "Clean Transit-Infrastructure Wave": {"SPY": 0.40, "QQQ": 0.40, "IWM": 0.20},
    "Cloud & Software Wave": {"QQQ": 0.50, "WCLD": 0.40, "HACK": 0.30},
    # Crypto: 50% WGMI, 30% BLOK, 20% BITQ
    "Crypto Income Wave": {"WGMI": 0.50, "BLOK": 0.30, "BITQ": 0.20},
    "Crypto Equity Wave": {"WGMI": 0.50, "BLOK": 0.30, "BITQ": 0.20},
    "Future Power & Energy Wave": {"QQQ": 0.40, "BUG": 0.30, "WCLD": 0.30},
    "Growth Wave": {"QQQ": 0.40, "BUG": 0.30, "WCLD": 0.30},
    "Quantum Computing Wave": {"QQQ": 0.50, "SOXX": 0.25, "ARKK": 0.25},
    "Small Cap Growth Wave": {"ARKK": 0.40, "IPAY": 0.30, "XLY": 0.30},
}

# VIX ladder (SmartSafe 2.0)
VIX_LEVELS = [
    (40.0, 0.80),
    (30.0, 0.50),
    (25.0, 0.30),
    (20.0, 0.15),
]

DAILY_RETURN_CLIP = 0.20  # +/-20% clamp
TRADING_DAYS_1Y = 252

# Mode tokens
MODE_STANDARD = "standard"
MODE_AMB = "amb"
MODE_PRIVATE_LOGIC = "pl"


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _standardize_column(df: pd.DataFrame, candidates: List[str], target: str) -> pd.DataFrame:
    df = _normalize_headers(df)
    norm_to_original = {str(col).strip().lower(): col for col in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_to_original:
            original = norm_to_original[key]
            if original != target:
                df = df.rename(columns={original: target})
            break
    return df


def _load_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        return pd.DataFrame(columns=["ticker"])
    df = pd.read_csv(UNIVERSE_CSV)
    df = _standardize_column(df, ["Ticker", "ticker", "Symbol", "symbol"], "ticker")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def _load_wave_weights() -> pd.DataFrame:
    if not WEIGHTS_CSV.exists():
        raise FileNotFoundError(f"wave_weights.csv not found at {WEIGHTS_CSV}")

    df = pd.read_csv(WEIGHTS_CSV)
    df = _normalize_headers(df)

    df = _standardize_column(df, ["Wave", "wave", "Portfolio", "portfolio", "Name"], "wave")
    df = _standardize_column(df, ["Ticker", "ticker", "Symbol", "symbol"], "ticker")
    df = _standardize_column(df, ["Weight", "weight", "Wgt", "wgt"], "weight")

    required_cols = ["wave", "ticker", "weight"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in wave_weights.csv: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    grouped = df.groupby(["wave", "ticker"], as_index=False)["weight"].sum()
    grouped["weight"] = grouped.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )
    return grouped


def get_available_waves() -> List[str]:
    weights = _load_wave_weights()
    if "wave" not in weights.columns:
        raise ValueError(
            f"'wave' column not found after normalization. Columns: {list(weights.columns)}"
        )
    return sorted(weights["wave"].unique().tolist())


def _get_single_benchmark_ticker(wave_name: str) -> str:
    if wave_name in BENCHMARK_MAP:
        return BENCHMARK_MAP[wave_name]

    name_lower = wave_name.lower()
    if "crypto" in name_lower or "digital" in name_lower:
        return "BITO"
    if "ai" in name_lower or "tech" in name_lower or "cloud" in name_lower:
        return "QQQ"
    if "small" in name_lower:
        return "IWM"
    if "income" in name_lower:
        return "SCHD"
    if "smartsafe" in name_lower:
        return "BIL"
    if "s&p" in name_lower:
        return "SPY"
    return "SPY"


def _get_benchmark_for_wave(wave_name: str) -> Tuple[str, Dict[str, float]]:
    if wave_name in BLENDED_BENCHMARKS:
        raw = BLENDED_BENCHMARKS[wave_name]
        total = float(sum(raw.values()))
        if total <= 0:
            t = _get_single_benchmark_ticker(wave_name)
            return t, {t: 1.0}
        spec = {t: w / total for t, w in raw.items()}
        parts = [f"{int(round(w * 100))}% {t}" for t, w in spec.items()]
        label = " + ".join(parts)
        return label, spec

    t = _get_single_benchmark_ticker(wave_name)
    return t, {t: 1.0}


def _download_price_series(ticker: str, period: str = "10y") -> pd.Series:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return pd.Series(dtype=float)
        col = None
        for candidate in ["Adj Close", "Close"]:
            if candidate in df.columns:
                col = candidate
                break
        if col is None:
            return pd.Series(dtype=float)
        s = df[col].copy()
        s.name = ticker
        return s
    except Exception:
        return pd.Series(dtype=float)


def _get_fast_intraday_return(ticker: str) -> Tuple[float, float]:
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        last_price = float(info.get("last_price", np.nan))
        prev_close = float(info.get("previous_close", np.nan))
        if np.isfinite(last_price) and np.isfinite(prev_close) and prev_close != 0:
            ret = (last_price / prev_close) - 1.0
        else:
            ret = 0.0
        return last_price, ret
    except Exception:
        return np.nan, 0.0


# ---------------------------------------------------------------------
# VIX & SmartSafe 2.0
# ---------------------------------------------------------------------

def _get_vix_level() -> Optional[float]:
    try:
        s = _download_price_series("^VIX", period="1y")
        if s.empty:
            return None
        return float(s.dropna().iloc[-1])
    except Exception:
        return None


def _compute_sweep_fraction_from_vix(vix_level: Optional[float]) -> float:
    if vix_level is None:
        return 0.0
    if vix_level < 20:
        return 0.0
    for threshold, frac in VIX_LEVELS:
        if vix_level >= threshold:
            return frac
    return 0.15


def _compute_vol_by_ticker(tickers: List[str], period: str = "90d") -> pd.DataFrame:
    records = []
    for t in tickers:
        s = _download_price_series(t, period=period)
        if s.empty:
            continue
        returns = s.pct_change().dropna()
        if returns.empty:
            continue
        returns = returns.clip(lower=-DAILY_RETURN_CLIP, upper=DAILY_RETURN_CLIP)
        vol = float(returns.std())
        records.append({"ticker": t, "vol": vol})
    if not records:
        return pd.DataFrame(columns=["ticker", "vol"])
    return pd.DataFrame(records)


def apply_smartsafe_sweep(
    wave_name: str,
    positions: pd.DataFrame,
    vix_level: Optional[float],
) -> pd.DataFrame:
    """
    SmartSafe 2.0 weighted sweep:

    - Skip SmartSafe Wave itself.
    - Use VIX ladder to compute sweep_fraction.
    - Trim highest-vol equities first, route to BIL.
    """
    if positions.empty:
        return positions
    if "smartsafe" in wave_name.lower():
        return positions

    sweep_fraction = _compute_sweep_fraction_from_vix(vix_level)
    if sweep_fraction <= 0.0:
        return positions

    df = positions.copy()
    total_weight = df["weight"].sum()
    if total_weight <= 0:
        return positions
    df["weight"] = df["weight"] / total_weight

    tickers = df["ticker"].unique().tolist()
    vol_df = _compute_vol_by_ticker(tickers, period="90d")
    if vol_df.empty:
        df["weight"] = df["weight"] * (1.0 - sweep_fraction)
        bil_last, bil_intraday = _get_fast_intraday_return("BIL")
        bil_row = {
            "ticker": "BIL",
            "weight": sweep_fraction,
            "last_price": bil_last,
            "intraday_return": bil_intraday,
        }
        df = pd.concat([df, pd.DataFrame([bil_row])], ignore_index=True)
        return df

    df = df.merge(vol_df, on="ticker", how="left")
    df["vol"] = df["vol"].fillna(0.0)

    df["is_bil"] = df["ticker"].str.upper().eq("BIL")
    df = df.sort_values(["is_bil", "vol"], ascending=[True, False])

    target_sweep = sweep_fraction
    remaining_sweep = target_sweep
    bil_weight = 0.0

    new_weights = []
    for _, row in df.iterrows():
        w = float(row["weight"])
        if row["is_bil"]:
            new_weights.append(w)
            continue
        if remaining_sweep <= 0.0:
            new_weights.append(w)
            continue
        cut = min(w, remaining_sweep)
        new_w = w - cut
        bil_weight += cut
        remaining_sweep -= cut
        new_weights.append(new_w)

    df["weight"] = new_weights
    df = df.drop(columns=["vol", "is_bil"])

    if bil_weight <= 0.0:
        return positions

    bil_last, bil_intraday = _get_fast_intraday_return("BIL")
    bil_mask = df["ticker"].str.upper().eq("BIL")
    if bil_mask.any():
        df.loc[bil_mask, "last_price"] = bil_last
        df.loc[bil_mask, "intraday_return"] = bil_intraday
        df.loc[bil_mask, "weight"] += bil_weight
    else:
        bil_row = {
            "ticker": "BIL",
            "weight": bil_weight,
            "last_price": bil_last,
            "intraday_return": bil_intraday,
        }
        df = pd.concat([df, pd.DataFrame([bil_row])], ignore_index=True)

    total_weight = df["weight"].sum()
    if total_weight > 0:
        df["weight"] = df["weight"] / total_weight

    return df


# ---------------------------------------------------------------------
# SmartSafe 3.0 LIVE overlay
# ---------------------------------------------------------------------

def apply_smartsafe3_extra_sweep(
    wave_name: str,
    positions: pd.DataFrame,
    extra_fraction: float,
) -> pd.DataFrame:
    """
    SmartSafe 3.0 extra sweep (LIVE overlay):

    - extra_fraction is 0.0–0.30, already decided by the regime function.
    - Works on *top of* SmartSafe 2.0.
    - Trims the highest-vol, non-BIL names first and routes into BIL.
    """
    if positions.empty:
        return positions
    if extra_fraction <= 0.0:
        return positions
    if "smartsafe" in wave_name.lower():
        return positions

    df = positions.copy()
    total_weight = df["weight"].sum()
    if total_weight <= 0:
        return positions
    df["weight"] = df["weight"] / total_weight

    tickers = df["ticker"].unique().tolist()
    vol_df = _compute_vol_by_ticker(tickers, period="90d")
    if vol_df.empty:
        df["weight"] = df["weight"] * (1.0 - extra_fraction)
        bil_last, bil_intraday = _get_fast_intraday_return("BIL")
        bil_row = {
            "ticker": "BIL",
            "weight": extra_fraction,
            "last_price": bil_last,
            "intraday_return": bil_intraday,
        }
        df = pd.concat([df, pd.DataFrame([bil_row])], ignore_index=True)
    else:
        df = df.merge(vol_df, on="ticker", how="left")
        df["vol"] = df["vol"].fillna(0.0)
        df["is_bil"] = df["ticker"].str.upper().eq("BIL")
        df = df.sort_values(["is_bil", "vol"], ascending=[True, False])

        remaining = extra_fraction
        bil_weight = 0.0
        new_weights = []
        for _, row in df.iterrows():
            w = float(row["weight"])
            if row["is_bil"]:
                new_weights.append(w)
                continue
            if remaining <= 0.0:
                new_weights.append(w)
                continue
            cut = min(w, remaining)
            new_w = w - cut
            bil_weight += cut
            remaining -= cut
            new_weights.append(new_w)

        df["weight"] = new_weights
        df = df.drop(columns=["vol", "is_bil"])

        bil_last, bil_intraday = _get_fast_intraday_return("BIL")
        bil_mask = df["ticker"].str.upper().eq("BIL")
        if bil_mask.any():
            df.loc[bil_mask, "last_price"] = bil_last
            df.loc[bil_mask, "intraday_return"] = bil_intraday
            df.loc[bil_mask, "weight"] += bil_weight
        else:
            bil_row = {
                "ticker": "BIL",
                "weight": bil_weight,
                "last_price": bil_last,
                "intraday_return": bil_intraday,
            }
            df = pd.concat([df, pd.DataFrame([bil_row])], ignore_index=True)

    total_weight = df["weight"].sum()
    if total_weight > 0:
        df["weight"] = df["weight"] / total_weight

    return df


# ---------------------------------------------------------------------
# Momentum (Alpha Engine)
# ---------------------------------------------------------------------

def _window_total_return(r: pd.Series, days: Optional[int] = None) -> float:
    if r.empty:
        return 0.0
    if days is not None:
        r = r.tail(days)
        if r.empty:
            return 0.0
    return float((1 + r).prod() - 1.0)


def _compute_momentum_scores(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Compute 30D and 90D total returns for each ticker, plus a combined momentum score.

    Returns DataFrame:
        ticker, mom_30, mom_90, mom_combo
    """
    records = []
    for t in tickers:
        s = _download_price_series(t, period=period)
        if s.empty:
            continue
        r = s.pct_change().dropna()
        if r.empty:
            continue
        r = r.clip(lower=-DAILY_RETURN_CLIP, upper=DAILY_RETURN_CLIP)
        mom_30 = _window_total_return(r, 30)
        mom_90 = _window_total_return(r, 90)
        mom_combo = 0.6 * mom_90 + 0.4 * mom_30
        records.append(
            {
                "ticker": t,
                "mom_30": mom_30,
                "mom_90": mom_90,
                "mom_combo": mom_combo,
            }
        )
    if not records:
        return pd.DataFrame(columns=["ticker", "mom_30", "mom_90", "mom_combo"])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Engineered concentration helper
# ---------------------------------------------------------------------

def _apply_concentration(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Engineered concentration:
    - Works on the equity sleeve only (non-BIL).
    - Raises weights to a power (p > 1) to concentrate capital into
      already-important names.
    - Caps single-name weights by mode; any excess flows to BIL.
    """
    if df.empty or "weight" not in df.columns or "ticker" not in df.columns:
        return df

    mode = (mode or MODE_STANDARD).lower()
    df = df.copy()

    is_bil = df["ticker"].str.upper().eq("BIL")
    eq_mask = ~is_bil

    eq_weight_total = float(df.loc[eq_mask, "weight"].sum())
    if eq_weight_total <= 0:
        return df

    w = df.loc[eq_mask, "weight"].values.astype(float)

    # Mode-specific concentration strength & cap
    if mode == MODE_STANDARD:
        p = 1.15
        cap = 0.18  # 18% per name
    elif mode == MODE_AMB:
        p = 1.05
        cap = 0.14  # more diversified
    elif mode == MODE_PRIVATE_LOGIC:
        p = 1.35
        cap = 0.25  # allow more concentration
    else:
        p = 1.15
        cap = 0.18

    # Power transform (convex)
    w_trans = np.power(w, p)
    if w_trans.sum() > 0:
        w_trans = w_trans / w_trans.sum() * eq_weight_total
    else:
        return df

    # Clip to cap; excess flows into BIL as extra defense
    w_clipped = np.minimum(w_trans, cap)
    eq_new_total = float(w_clipped.sum())
    excess = eq_weight_total - eq_new_total

    df.loc[eq_mask, "weight"] = w_clipped

    if excess > 0:
        if is_bil.any():
            df.loc[is_bil, "weight"] += excess
        else:
            bil_last, bil_intraday = _get_fast_intraday_return("BIL")
            bil_row = {
                "ticker": "BIL",
                "weight": excess,
                "last_price": bil_last,
                "intraday_return": bil_intraday,
            }
            df = pd.concat([df, pd.DataFrame([bil_row])], ignore_index=True)

    total_weight = df["weight"].sum()
    if total_weight > 0:
        df["weight"] = df["weight"] / total_weight

    return df


# ---------------------------------------------------------------------
# Mode adjustments (momentum + concentration)
# ---------------------------------------------------------------------

def _apply_mode_adjustments(
    wave_name: str,
    df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """
    Apply per-mode portfolio adjustments BEFORE SmartSafe 2.0.

    Pipeline:
    1) Ensure BIL row exists.
    2) Normalize starting weights.
    3) Compute momentum scores (30D+90D) and apply mode-specific momentum tilt.
    4) If AMB mode: move 15% of equity sleeve to BIL.
    5) Apply engineered concentration on equity sleeve.
    """
    if df.empty:
        return df

    mode = (mode or MODE_STANDARD).lower()
    df = df.copy()

    # Ensure BIL exists
    if not df["ticker"].str.upper().eq("BIL").any():
        bil_last, bil_intraday = _get_fast_intraday_return("BIL")
        bil_row = {
            "ticker": "BIL",
            "weight": 0.0,
            "last_price": bil_last,
            "intraday_return": bil_intraday,
        }
        df = pd.concat([df, pd.DataFrame([bil_row])], ignore_index=True)

    # Normalize starting weights
    total_weight = df["weight"].sum()
    if total_weight > 0:
        df["weight"] = df["weight"] / total_weight

    is_bil = df["ticker"].str.upper().eq("BIL")
    eq_mask = ~is_bil

    # Momentum scores for equity sleeve
    eq_df = df.loc[eq_mask].copy()
    if not eq_df.empty:
        mom_df = _compute_momentum_scores(eq_df["ticker"].tolist(), period="1y")
    else:
        mom_df = pd.DataFrame(columns=["ticker", "mom_30", "mom_90", "mom_combo"])

    if not eq_df.empty and not mom_df.empty:
        eq_df = eq_df.merge(mom_df, on="ticker", how="left")
        eq_df["mom_combo"] = eq_df["mom_combo"].fillna(0.0)
        intraday = eq_df["intraday_return"].fillna(0.0)

        base_score = eq_df["mom_combo"] + 0.1 * intraday
        mean = float(base_score.mean())
        std = float(base_score.std())
        if std > 0:
            z = (base_score - mean) / std
        else:
            z = pd.Series(0.0, index=eq_df.index)

        # Mode-specific momentum strength
        if mode == MODE_STANDARD:
            k = 0.15   # mild tilt
            min_factor, max_factor = 0.7, 1.3
        elif mode == MODE_AMB:
            k = 0.10   # very light tilt
            min_factor, max_factor = 0.8, 1.2
        elif mode == MODE_PRIVATE_LOGIC:
            k = 0.60   # strong tilt
            min_factor, max_factor = 0.5, 1.8
        else:
            k = 0.15
            min_factor, max_factor = 0.7, 1.3

        tilt_factor = 1.0 + k * z
        tilt_factor = tilt_factor.clip(lower=min_factor, upper=max_factor)

        eq_df["weight"] = eq_df["weight"] * tilt_factor
        eq_df["weight"] = eq_df["weight"].clip(lower=0.0)

        eq_before = df.loc[eq_mask, "weight"].sum()
        eq_after = eq_df["weight"].sum()
        if eq_after > 0 and eq_before > 0:
            eq_df["weight"] = eq_df["weight"] * (eq_before / eq_after)

        df.loc[eq_mask, "weight"] = eq_df["weight"]

    # AMB mode beta cut: move 15% of equity sleeve to BIL after momentum
    if mode == MODE_AMB:
        base_cut = 0.15
        eq_weight = df.loc[eq_mask, "weight"].sum()
        if eq_weight > 0:
            cut_amount = eq_weight * base_cut
            df.loc[eq_mask, "weight"] *= (1.0 - base_cut)
            df.loc[is_bil, "weight"] += cut_amount

    # Engineered concentration on equity sleeve
    df = _apply_concentration(df, mode)

    return df


# ---------------------------------------------------------------------
# Beta target helper
# ---------------------------------------------------------------------

def _get_beta_target(wave_name: str, mode: str) -> float:
    """
    Mode- and Wave-type-specific beta targets.

    Defaults:
        Standard: 0.90
        AMB:      0.78
        PL:       1.05

    Special cases:
        SmartSafe: ~0.05
        Income / defensive Waves: slightly lower base
        S&P Wave: ~1.00 in Standard
    """
    mode = (mode or MODE_STANDARD).lower()
    name_lower = wave_name.lower()

    # SmartSafe is basically cash
    if "smartsafe" in name_lower:
        return 0.05

    base = 0.90

    if "s&p" in name_lower:
        base = 1.00
    elif "income" in name_lower:
        base = 0.80
    elif "crypto" in name_lower:
        base = 1.10  # crypto waves allowed hotter beta
    elif "small cap" in name_lower:
        base = 1.05

    if mode == MODE_AMB:
        return max(0.50, base - 0.12)
    if mode == MODE_PRIVATE_LOGIC:
        return base + 0.15
    return base


# ---------------------------------------------------------------------
# Performance & risk metrics (incl. beta)
# ---------------------------------------------------------------------

def _compute_composite_benchmark_returns(
    benchmark_spec: Dict[str, float],
    period: str = "10y",
) -> pd.Series:
    if not benchmark_spec:
        return pd.Series(dtype=float)

    price_frames = []
    for t in benchmark_spec.keys():
        s = _download_price_series(t, period=period)
        if not s.empty:
            price_frames.append(s)
    if not price_frames:
        return pd.Series(dtype=float)

    prices_df = pd.concat(price_frames, axis=1).sort_index()
    prices_df = prices_df.ffill().dropna(how="all")

    weights = pd.Series(benchmark_spec)
    weights = weights.reindex(prices_df.columns).fillna(0.0)
    if weights.sum() != 0:
        weights = weights / weights.sum()

    bench_values = (prices_df * weights).sum(axis=1)
    bench_returns = bench_values.pct_change().dropna()
    bench_returns = bench_returns.clip(lower=-DAILY_RETURN_CLIP, upper=DAILY_RETURN_CLIP)
    return bench_returns


def _max_drawdown(values: pd.Series) -> float:
    """
    Max drawdown over full series; negative number (e.g., -0.32 = -32%).
    """
    if values.empty:
        return 0.0
    values = values.dropna()
    if values.empty:
        return 0.0
    running_max = values.cummax()
    drawdowns = values / running_max - 1.0
    return float(drawdowns.min())


def _estimate_beta(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """
    Estimate 1Y beta using daily returns (covariance/variance).
    """
    if port_returns.empty or bench_returns.empty:
        return 0.0

    r_p, r_b = port_returns.align(bench_returns, join="inner")
    if r_p.empty or r_b.empty:
        return 0.0

    r_p = r_p.tail(TRADING_DAYS_1Y)
    r_b = r_b.tail(TRADING_DAYS_1Y)
    if r_p.empty or r_b.empty:
        return 0.0

    var_b = float(r_b.var(ddof=1))
    if var_b <= 0:
        return 0.0
    cov = float(np.cov(r_b, r_p)[0, 1])
    return cov / var_b


def _compute_portfolio_trailing_returns(
    positions: pd.DataFrame,
    benchmark_spec: Dict[str, float],
    period: str = "10y",
) -> Dict[str, float]:
    """
    Returns:
        ret_30d, ret_60d, ret_1y, ret_si,
        alpha_30d, alpha_60d, alpha_1y, alpha_si,
        vol_1y, maxdd, ir_1y, hit_rate_1y,
        beta_1y
    """
    base_result = {
        "ret_30d": 0.0,
        "ret_60d": 0.0,
        "ret_1y": 0.0,
        "ret_si": 0.0,
        "alpha_30d": 0.0,
        "alpha_60d": 0.0,
        "alpha_1y": 0.0,
        "alpha_si": 0.0,
        "vol_1y": 0.0,
        "maxdd": 0.0,
        "ir_1y": 0.0,
        "hit_rate_1y": 0.0,
        "beta_1y": 0.0,
    }

    if positions.empty:
        return base_result

    tickers = positions["ticker"].unique().tolist()
    weights = positions.set_index("ticker")["weight"]
    weights = weights / weights.sum() if weights.sum() != 0 else weights

    price_frames = []
    for t in tickers:
        s = _download_price_series(t, period=period)
        if not s.empty:
            price_frames.append(s)
    if not price_frames:
        return base_result

    prices_df = pd.concat(price_frames, axis=1).sort_index()
    prices_df = prices_df.ffill().dropna(how="all")

    aligned_weights = weights.reindex(prices_df.columns).fillna(0.0)
    if aligned_weights.sum() != 0:
        aligned_weights = aligned_weights / aligned_weights.sum()

    port_values = (prices_df * aligned_weights).sum(axis=1)
    port_returns = port_values.pct_change().dropna()
    port_returns = port_returns.clip(lower=-DAILY_RETURN_CLIP, upper=DAILY_RETURN_CLIP)

    bench_returns = _compute_composite_benchmark_returns(benchmark_spec, period=period)
    if bench_returns.empty:
        bench_returns = pd.Series(0.0, index=port_returns.index)
    else:
        port_returns, bench_returns = port_returns.align(bench_returns, join="inner")
        port_values = port_values.loc[port_returns.index]

    port_30 = _window_total_return(port_returns, 30)
    port_60 = _window_total_return(port_returns, 60)
    port_1y = _window_total_return(port_returns, TRADING_DAYS_1Y)
    port_si = _window_total_return(port_returns, None)

    bench_30 = _window_total_return(bench_returns, 30)
    bench_60 = _window_total_return(bench_returns, 60)
    bench_1y = _window_total_return(bench_returns, TRADING_DAYS_1Y)
    bench_si = _window_total_return(bench_returns, None)

    excess = port_returns - bench_returns
    excess_1y = excess.tail(TRADING_DAYS_1Y)
    port_1y_window = port_returns.tail(TRADING_DAYS_1Y)

    if not port_1y_window.empty:
        vol_1y = float(port_1y_window.std() * np.sqrt(252))
    else:
        vol_1y = 0.0

    if not excess_1y.empty:
        std_excess = float(excess_1y.std())
        if std_excess > 0:
            ir_1y = float(excess_1y.mean() / std_excess * np.sqrt(252))
        else:
            ir_1y = 0.0
        hit_rate_1y = float((excess_1y > 0).mean())
    else:
        ir_1y = 0.0
        hit_rate_1y = 0.0

    maxdd = _max_drawdown(port_values)
    beta_1y = _estimate_beta(port_returns, bench_returns)

    return {
        "ret_30d": port_30,
        "ret_60d": port_60,
        "ret_1y": port_1y,
        "ret_si": port_si,
        "alpha_30d": port_30 - bench_30,
        "alpha_60d": port_60 - bench_60,
        "alpha_1y": port_1y - bench_1y,
        "alpha_si": port_si - bench_si,
        "vol_1y": vol_1y,
        "maxdd": maxdd,
        "ir_1y": ir_1y,
        "hit_rate_1y": hit_rate_1y,
        "beta_1y": beta_1y,
    }


def _log_positions(wave_name: str, positions: pd.DataFrame) -> None:
    if positions.empty:
        return
    today_str = datetime.now().strftime("%Y%m%d")
    file_path = POSITIONS_LOG_DIR / f"{wave_name.replace(' ', '_')}_positions_{today_str}.csv"
    try:
        positions.to_csv(file_path, index=False)
    except Exception:
        pass


def _log_performance(wave_name: str, metrics: Dict[str, float]) -> None:
    today_str = datetime.now().strftime("%Y-%m-%d")
    row = {
        "date": today_str,
        "ret_30d": metrics.get("ret_30d", 0.0),
        "ret_60d": metrics.get("ret_60d", 0.0),
        "ret_1y": metrics.get("ret_1y", 0.0),
        "ret_si": metrics.get("ret_si", 0.0),
        "alpha_30d": metrics.get("alpha_30d", 0.0),
        "alpha_60d": metrics.get("alpha_60d", 0.0),
        "alpha_1y": metrics.get("alpha_1y", 0.0),
        "alpha_si": metrics.get("alpha_si", 0.0),
        "vol_1y": metrics.get("vol_1y", 0.0),
        "maxdd": metrics.get("maxdd", 0.0),
        "ir_1y": metrics.get("ir_1y", 0.0),
        "hit_rate_1y": metrics.get("hit_rate_1y", 0.0),
        "beta_1y": metrics.get("beta_1y", 0.0),
    }
    file_path = PERF_LOG_DIR / f"{wave_name.replace(' ', '_')}_performance_daily.csv"
    try:
        if file_path.exists():
            existing = pd.read_csv(file_path)
            existing = existing[existing["date"] != today_str]
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            existing.to_csv(file_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(file_path, index=False)
    except Exception:
        pass


def _load_last_logged_metrics(wave_name: str) -> Optional[Dict[str, float]]:
    file_path = PERF_LOG_DIR / f"{wave_name.replace(' ', '_')}_performance_daily.csv"
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        last = df.sort_values("date").iloc[-1]
        return {
            "ret_30d": float(last.get("ret_30d", 0.0)),
            "ret_60d": float(last.get("ret_60d", 0.0)),
            "ret_1y": float(last.get("ret_1y", 0.0)),
            "ret_si": float(last.get("ret_si", 0.0)),
            "alpha_30d": float(last.get("alpha_30d", 0.0)),
            "alpha_60d": float(last.get("alpha_60d", 0.0)),
            "alpha_1y": float(last.get("alpha_1y", 0.0)),
            "alpha_si": float(last.get("alpha_si", 0.0)),
            "vol_1y": float(last.get("vol_1y", 0.0)),
            "maxdd": float(last.get("maxdd", 0.0)),
            "ir_1y": float(last.get("ir_1y", 0.0)),
            "hit_rate_1y": float(last.get("hit_rate_1y", 0.0)),
            "beta_1y": float(last.get("beta_1y", 0.0)),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------
# SmartSafe 3.0 regime logic
# ---------------------------------------------------------------------

def _compute_smartsafe3_recommendation(
    wave_name: str,
    mode: str,
    vix_level: Optional[float],
    trailing: Dict[str, float],
) -> Tuple[str, float]:
    """
    Returns:
        state: "Normal" | "Caution" | "Stress" | "Panic" | "Idle"
        extra_fraction: extra sweep fraction *on top of* SmartSafe 2.0 (0.0–0.30)
    """
    name_lower = wave_name.lower()
    if "smartsafe" in name_lower:
        return "Idle", 0.0

    vix = vix_level if vix_level is not None else 0.0
    ret_60d = trailing.get("ret_60d", 0.0)
    alpha_60d = trailing.get("alpha_60d", 0.0)
    maxdd = trailing.get("maxdd", 0.0)  # negative (e.g. -0.32)
    beta_1y = trailing.get("beta_1y", 0.0)

    beta_target = _get_beta_target(wave_name, mode)
    beta_drift = beta_1y - beta_target

    # PANIC
    if (
        vix >= 35.0
        or (maxdd <= -0.30 and ret_60d <= -0.15)
        or (beta_drift > 0.15 and ret_60d <= -0.10)
    ):
        return "Panic", 0.30

    # STRESS
    if (
        vix >= 28.0
        or (maxdd <= -0.20 and ret_60d <= -0.10)
        or (beta_drift > 0.10 and ret_60d <= -0.05)
    ):
        return "Stress", 0.15

    # CAUTION
    if (
        vix >= 22.0
        or (maxdd <= -0.15 and ret_60d < 0.0)
        or (beta_drift > 0.05 and alpha_60d < 0.0)
    ):
        return "Caution", 0.05

    return "Normal", 0.0


# ---------------------------------------------------------------------
# Core Wave Snapshot
# ---------------------------------------------------------------------

def _build_wave_positions(
    wave_name: str,
    vix_level: Optional[float],
    mode: str,
) -> pd.DataFrame:
    """
    Build positions **after** mode adjustments and SmartSafe 2.0.

    SmartSafe 3.0 overlay is applied later in get_wave_snapshot so we
    can use the 2.0-only portfolio for trailing metrics.
    """
    weights = _load_wave_weights()
    wave_df = weights[weights["wave"] == wave_name].copy()
    if wave_df.empty:
        return pd.DataFrame(columns=["ticker", "weight", "last_price", "intraday_return"])

    prices = []
    intraday_returns = []
    for _, row in wave_df.iterrows():
        ticker = row["ticker"]
        last_price, intraday_ret = _get_fast_intraday_return(ticker)
        prices.append(last_price)
        intraday_returns.append(intraday_ret)

    wave_df["last_price"] = prices
    wave_df["intraday_return"] = intraday_returns

    wave_df = _apply_mode_adjustments(wave_name, wave_df, mode)
    wave_df = apply_smartsafe_sweep(wave_name, wave_df, vix_level)

    return wave_df


def get_wave_snapshot(wave_name: str, mode: str = MODE_STANDARD) -> Dict:
    mode = (mode or MODE_STANDARD).lower()
    if mode not in {MODE_STANDARD, MODE_AMB, MODE_PRIVATE_LOGIC}:
        mode = MODE_STANDARD

    benchmark_label, benchmark_spec = _get_benchmark_for_wave(wave_name)

    vix_level = _get_vix_level()
    sweep_fraction = _compute_sweep_fraction_from_vix(vix_level)

    # 1) Build positions with mode logic + SmartSafe 2.0 only
    positions_core = _build_wave_positions(wave_name, vix_level=vix_level, mode=mode)

    # 2) Trailing metrics based on the 2.0 portfolio (history pre-3.0 overlay)
    trailing = _compute_portfolio_trailing_returns(
        positions_core,
        benchmark_spec=benchmark_spec,
        period="10y",
    )

    if (
        trailing["ret_30d"] == 0.0
        and trailing["ret_60d"] == 0.0
        and trailing["ret_1y"] == 0.0
        and trailing["ret_si"] == 0.0
        and trailing["alpha_30d"] == 0.0
        and trailing["alpha_60d"] == 0.0
        and trailing["alpha_1y"] == 0.0
        and trailing["alpha_si"] == 0.0
    ):
        logged = _load_last_logged_metrics(wave_name)
        if logged is not None:
            trailing = logged

    beta_1y = trailing.get("beta_1y", 0.0)
    beta_target = _get_beta_target(wave_name, mode)
    beta_drift = beta_1y - beta_target

    # 3) SmartSafe 3.0 regime + extra sweep (LIVE overlay on positions)
    ss3_state, ss3_extra = _compute_smartsafe3_recommendation(
        wave_name=wave_name,
        mode=mode,
        vix_level=vix_level,
        trailing=trailing,
    )
    positions_live = apply_smartsafe3_extra_sweep(
        wave_name=wave_name,
        positions=positions_core,
        extra_fraction=ss3_extra,
    )

    # 4) Intraday return is based on the LIVE positions (after 3.0 overlay)
    if not positions_live.empty:
        w = positions_live["weight"]
        w = w / w.sum() if w.sum() != 0 else w
        intraday_ret = float((positions_live["intraday_return"] * w).sum())
    else:
        intraday_ret = 0.0

    metrics = {
        "intraday_return": intraday_ret,
        "ret_30d": trailing["ret_30d"],
        "ret_60d": trailing["ret_60d"],
        "ret_1y": trailing["ret_1y"],
        "ret_si": trailing["ret_si"],
        "alpha_30d": trailing["alpha_30d"],
        "alpha_60d": trailing["alpha_60d"],
        "alpha_1y": trailing["alpha_1y"],
        "alpha_si": trailing["alpha_si"],
        "vol_1y": trailing["vol_1y"],
        "maxdd": trailing["maxdd"],
        "ir_1y": trailing["ir_1y"],
        "hit_rate_1y": trailing["hit_rate_1y"],
        "beta_1y": beta_1y,
        "beta_target": beta_target,
        "beta_drift": beta_drift,
        "vix_level": vix_level,
        "smartsafe_sweep_fraction": sweep_fraction,
        "smartsafe3_state": ss3_state,
        "smartsafe3_extra_fraction": ss3_extra,
        "mode": mode,
    }

    _log_positions(wave_name, positions_live)
    _log_performance(wave_name, metrics)

    return {
        "wave_name": wave_name,
        "benchmark": benchmark_label,
        "positions": positions_live,
        "metrics": metrics,
    }