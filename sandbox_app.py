# sandbox_app.py — WAVES Intelligence™ Sandbox (Wave B)
# Small–Mid Cap Value Acceleration Wave
#
# Design goals:
# - One file, deterministic, zero external dependencies beyond streamlit+pandas
# - Runs on Streamlit Cloud reliably (no paid fundamentals, no fragile APIs)
# - Lets Jason *see what it looks like* and validate the behavior/structure
#
# NOTE: This uses a mock fundamentals universe + simulated prices.
# Once you like the shape/behavior, we swap in real tickers + real fundamentals
# without changing the Wave structure.

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Utilities
# -----------------------------

def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x*100:.2f}%"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# Sandbox Registry (only what we need)
# -----------------------------

WAVES = {
    "SMID_VALUE_ACCEL": {
        "name": "Small–Mid Cap Value Acceleration Wave",
        "benchmark": {"IWM": 0.60, "IJR": 0.40},
        "beta_target": 0.80,
        "modes": ["Standard", "Alpha-Minus-Beta"],
        "description": "QoQ rev ≥20%, QoQ EPS ≥25%, P/E ≤12 (or EV/EBITDA ≤8). Plus OCF>0 + liquidity screen + risk sizing."
    }
}


# -----------------------------
# Mock Fundamentals Universe
# -----------------------------

def build_mock_smid_universe(seed: int = 7, n: int = 280) -> pd.DataFrame:
    """
    Creates a synthetic SMID universe.
    Fields approximate what we'd use in production (revenues/earnings growth, valuation, liquidity, risk).
    """
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        ticker = f"SMID{i:03d}"

        market_cap = rng.uniform(5e8, 1.2e10)          # $0.5B to $12B
        avg_dollar_volume = rng.uniform(1e6, 120e6)    # liquidity

        qoq_rev = clamp(rng.gauss(0.18, 0.20), -0.30, 1.50)   # -30% to 150%
        qoq_eps = clamp(rng.gauss(0.22, 0.35), -0.70, 2.50)   # -70% to 250%

        pe = clamp(rng.gauss(14.0, 7.0), 2.0, 70.0)
        ev_ebitda = clamp(rng.gauss(9.0, 4.0), 2.0, 40.0)

        ocf = rng.gauss(40e6, 140e6)  # operating cash flow

        vol_60d = clamp(rng.gauss(0.45, 0.20), 0.12, 1.40)
        gap_risk = clamp(rng.random(), 0.0, 1.0)

        rows.append({
            "ticker": ticker,
            "market_cap": market_cap,
            "avg_dollar_volume": avg_dollar_volume,
            "qoq_revenue_growth": qoq_rev,
            "qoq_eps_growth": qoq_eps,
            "pe_ratio": pe,
            "ev_to_ebitda": ev_ebitda,
            "operating_cash_flow": ocf,
            "volatility_60d": vol_60d,
            "gap_risk": gap_risk,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Selection + Construction
# -----------------------------

def select_value_acceleration(
    universe: pd.DataFrame,
    min_qoq_rev: float,
    min_qoq_eps: float,
    max_pe: float,
    max_names: int,
    min_adv: float,
    require_ocf_pos: bool,
    allow_ev_ebitda_alt: bool,
    max_ev_ebitda: float,
) -> pd.DataFrame:
    df = universe.copy()

    # liquidity / cap screen
    df = df[(df["avg_dollar_volume"] >= min_adv)]
    df = df[(df["market_cap"] >= 5e8) & (df["market_cap"] <= 1.2e10)]

    # growth screens
    df = df[(df["qoq_revenue_growth"] >= min_qoq_rev)]
    df = df[(df["qoq_eps_growth"] >= min_qoq_eps)]

    # cashflow
    if require_ocf_pos:
        df = df[df["operating_cash_flow"] > 0]

    # valuation screen
    if allow_ev_ebitda_alt:
        df = df[(df["pe_ratio"] <= max_pe) | (df["ev_to_ebitda"] <= max_ev_ebitda)]
    else:
        df = df[df["pe_ratio"] <= max_pe]

    if df.empty:
        return df

    # scoring: growth + value + stability
    # (These are percentiles to stay stable across varying distributions)
    df["growth_score"] = (
        df["qoq_revenue_growth"].rank(pct=True) +
        df["qoq_eps_growth"].rank(pct=True)
    )
    df["value_score"] = (
        (1.0 / df["pe_ratio"]).rank(pct=True) +
        (1.0 / df["ev_to_ebitda"]).rank(pct=True)
    )
    df["stability_score"] = (
        (1.0 / df["volatility_60d"]).rank(pct=True) +
        (1.0 / (1.0 + df["gap_risk"])).rank(pct=True)
    )

    df["composite_score"] = (
        0.52 * df["growth_score"] +
        0.33 * df["value_score"] +
        0.15 * df["stability_score"]
    )

    df = df.sort_values("composite_score", ascending=False).head(max_names)
    return df


def construct_weights(scored: pd.DataFrame, max_weight: float = 0.06) -> Dict[str, float]:
    if scored.empty:
        return {}

    # Inverse-vol weights, capped
    inv_vol = 1.0 / scored["volatility_60d"].clip(lower=0.10)
    w = inv_vol / inv_vol.sum()

    w = w.clip(upper=max_weight)
    w = w / w.sum()

    return {t: float(x) for t, x in zip(scored["ticker"], w)}


# -----------------------------
# Simulated Prices
# -----------------------------

def simulated_prices(tickers: List[str], days: int = 420, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic-ish price generator.
    Each ticker gets its own drift/vol derived from hash(ticker) so it's stable across reloads.
    """
    rng = random.Random(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="B")

    data = {}
    for t in tickers:
        h = abs(hash(t)) % 1000
        drift = (h / 1000.0 - 0.5) * 0.0007
        vol = 0.010 + (h % 35) / 1000.0

        px = 100.0
        series = []
        for _ in range(len(dates)):
            shock = rng.gauss(0.0, vol)
            px *= max(0.90, min(1.10, (1.0 + drift + shock)))
            series.append(px)
        data[t] = series

    return pd.DataFrame(data, index=dates)


# -----------------------------
# Portfolio Math
# -----------------------------

def portfolio_series(prices: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    if prices.empty or not weights:
        return pd.Series(dtype=float)

    cols = [c for c in prices.columns if c in weights]
    if not cols:
        return pd.Series(dtype=float)

    p = prices[cols].ffill().dropna(how="all")
    if p.empty:
        return pd.Series(dtype=float)

    base = p.iloc[0].replace(0, pd.NA)
    norm = p.divide(base)

    w = pd.Series({c: float(weights.get(c, 0.0)) for c in cols}, dtype=float)
    if w.sum() == 0:
        return pd.Series(dtype=float)
    w = w / w.sum()

    return norm.mul(w, axis=1).sum(axis=1)


def simple_return(series: pd.Series, days: int) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna()
    if len(s) < 10:
        return None
    end_val = s.iloc[-1]
    end_date = s.index[-1]
    start_date = end_date - pd.Timedelta(days=days)
    past = s[s.index <= start_date]
    if past.empty:
        return None
    start_val = past.iloc[-1]
    if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        return None
    return float(end_val / start_val - 1.0)


def benchmark_desc(bm: Dict[str, float]) -> str:
    if not bm:
        return "—"
    return " + ".join([f"{k} {v:.0%}" for k, v in bm.items()])


@dataclass
class WaveOutput:
    wave_name: str
    mode: str
    benchmark: str
    holdings: Dict[str, float]
    ret_30d: Optional[float]
    ret_60d: Optional[float]
    ret_365d: Optional[float]
    alpha_30d: Optional[float]
    alpha_60d: Optional[float]
    alpha_365d: Optional[float]


def compute_alpha(mode: str, port: pd.Series, bench: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    r30 = simple_return(port, 30)
    r60 = simple_return(port, 60)
    r365 = simple_return(port, 365)

    b30 = simple_return(bench, 30) if not bench.empty else None
    b60 = simple_return(bench, 60) if not bench.empty else None
    b365 = simple_return(bench, 365) if not bench.empty else None

    # Sandbox: Alpha-Minus-Beta and Standard both show benchmark-relative alpha
    # (Later we can add exposure scaling and beta targeting.)
    a30 = (r30 - b30) if (r30 is not None and b30 is not None) else None
    a60 = (r60 - b60) if (r60 is not None and b60 is not None) else None
    a365 = (r365 - b365) if (r365 is not None and b365 is not None) else None

    return r30, r60, r365, a30, a60, a365


def run_smid_value_accel_wave(
    mode: str,
    seed: int,
    min_qoq_rev: float,
    min_qoq_eps: float,
    max_pe: float,
    max_names: int,
    min_adv: float,
    require_ocf_pos: bool,
    allow_ev_ebitda_alt: bool,
    max_ev_ebitda: float,
) -> WaveOutput:
    universe = build_mock_smid_universe(seed=seed, n=280)

    scored = select_value_acceleration(
        universe=universe,
        min_qoq_rev=min_qoq_rev,
        min_qoq_eps=min_qoq_eps,
        max_pe=max_pe,
        max_names=max_names,
        min_adv=min_adv,
        require_ocf_pos=require_ocf_pos,
        allow_ev_ebitda_alt=allow_ev_ebitda_alt,
        max_ev_ebitda=max_ev_ebitda,
    )
    holdings = construct_weights(scored, max_weight=0.06)

    bm = WAVES["SMID_VALUE_ACCEL"]["benchmark"]
    tickers = sorted(set(list(holdings.keys()) + list(bm.keys())))

    if tickers:
        prices = simulated_prices(tickers, days=420, seed=42)
    else:
        prices = pd.DataFrame()

    port = portfolio_series(prices, holdings) if holdings else pd.Series(dtype=float)
    bench = portfolio_series(prices, bm) if bm else pd.Series(dtype=float)

    r30, r60, r365, a30, a60, a365 = compute_alpha(mode, port, bench)

    return WaveOutput(
        wave_name=WAVES["SMID_VALUE_ACCEL"]["name"],
        mode=mode,
        benchmark=benchmark_desc(bm),
        holdings=holdings,
        ret_30d=r30, ret_60d=r60, ret_365d=r365,
        alpha_30d=a30, alpha_60d=a60, alpha_365d=a365
    )


# -----------------------------
# Streamlit