# sandbox_app.py — WAVES Intelligence™ (One-File Sandbox)
# Purpose: Let Jason see the new "Small–Mid Cap Value Acceleration Wave" immediately
# WITHOUT touching the production engine/console.
#
# Runs on:
# - mock fundamentals universe
# - simulated prices (deterministic)
# Provides:
# - Standard + Alpha-Minus-Beta
# - Overview table + drill-down holdings

from __future__ import annotations

import math
import random
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# -----------------------------
# Wave Registry (Sandbox Only)
# -----------------------------

WAVES = {
    "SMID_VALUE_ACCEL": {
        "name": "Small–Mid Cap Value Acceleration Wave",
        "benchmark": {"IWM": 0.60, "IJR": 0.40},
        "beta_target": 0.80,
        "modes": ["Standard", "Alpha-Minus-Beta"],
        "description": "QoQ rev >=20%, QoQ EPS >=25%, P/E<=12 (or EV/EBITDA<=8), OCF>0, liquidity + risk sizing."
    },
    "MEGACAP_CORE": {
        "name": "US MegaCap Core Wave (Sandbox)",
        "benchmark": {"SPY": 1.00},
        "beta_target": 1.00,
        "modes": ["Standard", "Alpha-Minus-Beta"],
        "description": "Static basket (sandbox)."
    },
    "AI_CLOUD": {
        "name": "AI & Cloud MegaCap Wave (Sandbox)",
        "benchmark": {"QQQ": 1.00},
        "beta_target": 1.00,
        "modes": ["Standard", "Alpha-Minus-Beta"],
        "description": "Static basket (sandbox)."
    },
    "SMARTSAFE_TREASURY": {
        "name": "SmartSafe Treasury Cash Wave (Sandbox)",
        "benchmark": {"BIL": 1.00},
        "beta_target": 0.05,
        "modes": ["Standard", "Alpha-Minus-Beta"],
        "description": "Defensive proxy (sandbox)."
    },
}

STATIC_BASKETS = {
    "MEGACAP_CORE": {
        "AAPL": 0.10, "MSFT": 0.10, "AMZN": 0.08, "GOOGL": 0.07, "META": 0.06,
        "NVDA": 0.08, "BRK-B": 0.06, "LLY": 0.05, "JPM": 0.05, "AVGO": 0.07,
        "XOM": 0.05, "UNH": 0.05, "V": 0.08
    },
    "AI_CLOUD": {
        "NVDA": 0.14, "MSFT": 0.12, "AMZN": 0.10, "GOOGL": 0.08, "META": 0.08,
        "AVGO": 0.08, "AMD": 0.06, "CRM": 0.06, "NOW": 0.06, "PLTR": 0.05,
        "SNOW": 0.05, "ORCL": 0.06, "TSM": 0.06
    },
    "SMARTSAFE_TREASURY": {"BIL": 1.00},
}


# -----------------------------
# Mock Fundamentals Universe
# -----------------------------

def build_mock_smid_universe(seed: int = 7, n: int = 250) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        ticker = f"SMID{i:03d}"

        market_cap = rng.uniform(5e8, 1e10)
        avg_dollar_volume = rng.uniform(1e6, 80e6)

        qoq_rev = max(-0.2, min(1.2, rng.gauss(0.18, 0.18)))
        qoq_eps = max(-0.5, min(2.0, rng.gauss(0.22, 0.30)))

        pe = max(2.0, min(60.0, rng.gauss(14.0, 7.0)))
        ev_ebitda = max(2.0, min(30.0, rng.gauss(9.0, 4.0)))

        operating_cash_flow = rng.gauss(25e6, 120e6)

        vol_60d = max(0.12, min(1.2, rng.gauss(0.45, 0.20)))
        gap_risk = max(0.0, min(1.0, rng.random()))

        rows.append({
            "ticker": ticker,
            "market_cap": market_cap,
            "avg_dollar_volume": avg_dollar_volume,
            "qoq_revenue_growth": qoq_rev,
            "qoq_eps_growth": qoq_eps,
            "pe_ratio": pe,
            "ev_to_ebitda": ev_ebitda,
            "operating_cash_flow": operating_cash_flow,
            "volatility_60d": vol_60d,
            "gap_risk": gap_risk,
        })
    return pd.DataFrame(rows)


# -----------------------------
# SMID Value Acceleration Wave
# -----------------------------

def smid_value_accel_select(universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.copy()

    df = df[
        (df["market_cap"] >= 5e8) &
        (df["market_cap"] <= 1e10) &
        (df["avg_dollar_volume"] >= 5e6)
    ]

    df = df[
        (df["qoq_revenue_growth"] >= 0.20) &
        (df["qoq_eps_growth"] >= 0.25) &
        (
            (df["pe_ratio"] <= 12.0) |
            (df["ev_to_ebitda"] <= 8.0)
        ) &
        (df["operating_cash_flow"] > 0)
    ]

    if df.empty:
        return df

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
        0.50 * df["growth_score"] +
        0.35 * df["value_score"] +
        0.15 * df["stability_score"]
    )

    return df.sort_values("composite_score", ascending=False)


def smid_value_accel_construct(scored: pd.DataFrame, max_names: int = 35) -> Dict[str, float]:
    if scored.empty:
        return {}
    top = scored.head(max_names).copy()

    inv_vol = 1.0 / top["volatility_60d"].clip(lower=0.10)
    w = inv_vol / inv_vol.sum()
    w = w.clip(upper=0.06)
    w = w / w.sum()

    return dict(zip(top["ticker"], w.astype(float)))


# -----------------------------
# Simulated Prices
# -----------------------------

def simulated_prices(tickers: List[str], days: int = 420, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="B")

    data = {}
    for t in tickers:
        h = abs(hash(t)) % 1000
        drift = (h / 1000.0 - 0.5) * 0.0006
        vol = 0.012 + (h % 30) / 1000.0

        px = 100.0
        series = []
        for _ in range(len(dates)):
            shock = rng.gauss(0.0, vol)
            px *= max(0.90, min(1.10, (1.0 + drift + shock)))
            series.append(px)
        data[t] = series

    return pd.DataFrame(data, index=dates)


# -----------------------------
# Returns / Alpha
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
    if len(s) < 5:
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


def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x*100:.2f}%"


@dataclass
class WaveOutput:
    wave_id: str
    wave_name: str
    benchmark_desc: str
    mode: str
    holdings: Dict[str, float]
    ret_30d: Optional[float]
    ret_60d: Optional[float]
    ret_365d: Optional[float]
    alpha_30d: Optional[float]
    alpha_60d: Optional[float]
    alpha_365d: Optional[float]


def bm_desc(bm: Dict[str, float]) -> str:
    if not bm:
        return "—"
    return " + ".join([f"{k} {v:.0%}" for k, v in bm.items()])


def run_wave(wave_id: str, mode: str, universe: pd.DataFrame) -> WaveOutput:
    cfg = WAVES[wave_id]
    name = cfg["name"]
    bm = cfg.get("benchmark", {})
    desc = bm_desc(bm)

    # holdings
    if wave_id == "SMID_VALUE_ACCEL":
        scored = smid_value_accel_select(universe)
        holdings = smid_value_accel_construct(scored, max_names=35)
    else:
        holdings = STATIC_BASKETS.get(wave_id, {}).copy()

    tickers = sorted(set(list(holdings.keys()) + list(bm.keys())))
    prices = simulated_prices(tickers, days=420, seed=42)

    port = portfolio_series(prices, holdings)
    bench = portfolio_series(prices, bm) if bm else pd.Series(dtype=float)

    r30 = simple_return(port, 30)
    r60 = simple_return(port, 60)
    r365 = simple_return(port, 365)

    b30 = simple_return(bench, 30) if not bench.empty else None
    b60 = simple_return(bench, 60) if not bench.empty else None
    b365 = simple_return(bench, 365) if not bench.empty else None

    # Standard: show benchmark-relative alpha. AMB: same math for sandbox (scaler=1).
    if (r30 is not None and b30 is not None): a30 = r30 - b30
    else: a30 = None
    if (r60 is not None and b60 is not None): a60 = r60 - b60
    else: a60 = None
    if (r365 is not None and b365 is not None): a365 = r365 - b365
    else: a365 = None

    return WaveOutput(
        wave_id=wave_id,
        wave_name=name,
        benchmark_desc=desc,
        mode=mode,
        holdings=holdings,
        ret_30d=r30, ret_60d=r60, ret_365d=r365,
        alpha_30d=a30, alpha_60d=a60, alpha_365d=a365
    )


def run_all(mode: str, seed: int) -> List[WaveOutput]:
    universe = build_mock_smid_universe(seed=seed, n=250)
    outs = []
    for wid in WAVES.keys():
        if mode not in WAVES[wid].get("modes", ["Standard", "Alpha-Minus-Beta"]):
            continue
        outs.append(run_wave(wid, mode, universe))
    return outs


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="WAVES Sandbox — One File", layout="wide")
st.title("WAVES Intelligence™ — One-File Sandbox")
st.caption("Safe sandbox to preview the Small–Mid Cap Value Acceleration Wave without touching production.")

mode = st.radio("Mode", ["Standard", "Alpha-Minus-Beta"], horizontal=True, index=1)
seed = st.slider("Mock Universe Seed", 1, 999, 7, 1)

with st.spinner("Running sandbox…"):
    outs = run_all(mode=mode, seed=seed)

rows = []
for o in outs:
    rows.append({
        "Wave": o.wave_name,
        "Benchmark": o.benchmark_desc,
        "30D Return": fmt_pct(o.ret_30d),
        "30D Alpha": fmt_pct(o.alpha_30d),
        "60D Return": fmt_pct(o.ret_60d),
        "60D Alpha": fmt_pct(o.alpha_60d),
        "365D Return": fmt_pct(o.ret_365d),
        "365D Alpha": fmt_pct(o.alpha_365d),
        "Holdings": len(o.holdings),
    })

df = pd.DataFrame(rows)
st.subheader("All Waves (Sandbox)")
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Wave Drill-Down (Sandbox)")
wave_id = st.selectbox("Select a Wave", list(WAVES.keys()), format_func=lambda x: WAVES[x]["name"])

universe = build_mock_smid_universe(seed=seed, n=250)
o = run_wave(wave_id=wave_id, mode=mode, universe=universe)

c1, c2, c3, c4 = st.columns(4)
c1.metric("30D Return", fmt_pct(o.ret_30d))
c2.metric("30D Alpha", fmt_pct(o.alpha_30d))
c3.metric("60D Return", fmt_pct(o.ret_60d))
c4.metric("60D Alpha", fmt_pct(o.alpha_60d))

c5, c6, c7, c8 = st.columns(4)
c5.metric("365D Return", fmt_pct(o.ret_365d))
c6.metric("365D Alpha", fmt_pct(o.alpha_365d))
c7.metric("Benchmark", o.benchmark_desc)
c8.metric("Holdings", str(len(o.holdings)))

hold_df = pd.DataFrame([{"Ticker": k, "Weight": v} for k, v in o.holdings.items()]).sort_values("Weight", ascending=False)
st.markdown("**Top Holdings**")
st.dataframe(hold_df.head(15), use_container_width=True, hide_index=True)

st.info(
    "This sandbox uses synthetic tickers (SMID###) and simulated prices. "
    "Once you like the behavior, we swap in real tickers + real fundamentals with the same Wave structure."
)