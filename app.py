# pages/99_Backfill_History.py
# WAVES Intelligence™ — History Backfill Utility (safe)
#
# Purpose:
# - Rebuild missing performance history files so the Institutional Console
#   can show 30D/365D returns, alpha, WaveScore inputs, Risk Lab, Attribution, Correlation.
#
# What it does:
# - Reads wave_weights.csv
# - For each Wave (and selected Mode), downloads ~400 trading days of prices via yfinance
# - Builds a daily NAV series from weights (no engine logic modified)
# - Writes logs/performance/<wave>__<mode>_performance_daily.csv
#
# Notes:
# - This is a "restore the meat" tool. It does NOT change your optimizer/engine.
# - If you later prefer engine-authored logs, keep those — this just fills gaps.

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception as e:
    yf = None

ROOT = os.getcwd()
LOG_DIR = os.path.join(ROOT, "logs", "performance")
WEIGHTS_PATH = os.path.join(ROOT, "wave_weights.csv")

DEFAULT_LOOKBACK_DAYS = 520  # ~2 calendar years to ensure >= 365 trading days coverage
DEFAULT_MIN_ROWS = 80        # minimum daily rows required to consider “good enough”

MODE_ALIASES = {
    "Standard": ["Standard", "standard", "STANDARD", "Base", "BASE", "Normal", "NORMAL"],
    "Alpha-Minus-Beta": ["Alpha-Minus-Beta", "alpha-minus-beta", "AMB", "amb", "AlphaMinusBeta"],
    "Private Logic": ["Private Logic", "private logic", "PL", "pl", "PrivateLogic", "Private Logic™"],
}

def safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\s\-\.]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s

@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_prices(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not available in this environment.")
    tks = sorted(list({t.strip().upper() for t in tickers if str(t).strip()}))
    if not tks:
        return pd.DataFrame()

    df = yf.download(
        tickers=tks,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Normalize output to a simple columns=tickers, index=Date, values=Close
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Close if present else Adj Close else any price-like
        if ("Close" in df.columns.get_level_values(0)):
            px = df["Close"].copy()
        else:
            # fallback: pick first top level
            top0 = df.columns.get_level_values(0)[0]
            px = df[top0].copy()
    else:
        # single ticker case may come as Series-like columns
        if "Close" in df.columns:
            px = df[["Close"]].copy()
            px.columns = tks[:1]
        else:
            px = df.copy()

    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px.sort_index()
    px = px.dropna(how="all")
    return px

def build_nav_from_weights(weights_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """
    weights_df columns: ticker, weight
    returns a dataframe with: date, nav, ret
    """
    tickers = weights_df["ticker"].astype(str).str.upper().tolist()
    w = weights_df.set_index(weights_df["ticker"].astype(str).str.upper())["weight"].astype(float)

    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    px = fetch_prices(tickers, start, end)
    if px.empty:
        return pd.DataFrame()

    # Align weights to available tickers
    common = [c for c in px.columns if c in w.index]
    if not common:
        return pd.DataFrame()

    w2 = w.loc[common].copy()
    # If weights don’t sum to 1, normalize (safe)
    s = float(w2.sum())
    if s <= 0:
        return pd.DataFrame()
    w2 = w2 / s

    px2 = px[common].ffill().dropna()
    if px2.empty or len(px2) < 5:
        return pd.DataFrame()

    rets = px2.pct_change().fillna(0.0)
    port_ret = (rets * w2.values).sum(axis=1)

    nav = (1.0 + port_ret).cumprod() * 100.0  # base=100
    out = pd.DataFrame({"date": nav.index, "nav": nav.values, "ret": port_ret.values})
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    return out

def write_perf_csv(wave: str, mode: str, perf: pd.DataFrame) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    fname = f"{safe_slug(wave)}__{safe_slug(mode)}_performance_daily.csv"
    path = os.path.join(LOG_DIR, fname)
    perf.to_csv(path, index=False)
    return path

def load_weights() -> pd.DataFrame:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing wave_weights.csv at: {WEIGHTS_PATH}")
    df = pd.read_csv(WEIGHTS_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    # Expect: wave, ticker, weight
    for col in ["wave", "ticker", "weight"]:
        if col not in df.columns:
            raise ValueError(f"wave_weights.csv must include columns: wave,ticker,weight (missing {col})")
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df[df["ticker"].str.len() > 0]
    df = df[df["weight"] != 0]
    return df

st.set_page_config(page_title="Backfill History", layout="wide")

st.title("🧱 Backfill Missing History (Restore the “Meat”)")
st.caption("This page rebuilds daily performance history files used by the Institutional Console. Safe: does not change engine logic.")

if yf is None:
    st.error("yfinance is not available — add it to requirements.txt (yfinance==0.2.66) and redeploy.")
    st.stop()

try:
    wdf = load_weights()
except Exception as e:
    st.error(str(e))
    st.stop()

waves = sorted(wdf["wave"].unique().tolist())

colA, colB, colC = st.columns(3)
with colA:
    selected_waves = st.multiselect("Waves to backfill", waves, default=waves[: min(6, len(waves))])
with colB:
    mode = st.selectbox("Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic"], index=0)
with colC:
    lookback = st.number_input("Lookback (calendar days)", min_value=120, max_value=2000, value=DEFAULT_LOOKBACK_DAYS, step=20)

st.info(
    f"Will write files to: {LOG_DIR}\n\n"
    "Expected output naming: logs/performance/<Wave>__<Mode>_performance_daily.csv"
)

run = st.button("🚀 Run Backfill", use_container_width=True)

if run:
    if not selected_waves:
        st.warning("Select at least one Wave.")
        st.stop()

    results = []
    prog = st.progress(0)
    total = len(selected_waves)

    for i, wave in enumerate(selected_waves, start=1):
        try:
            sub = wdf[wdf["wave"] == wave][["ticker", "weight"]].copy()
            perf = build_nav_from_weights(sub, int(lookback))

            if perf.empty or len(perf) < DEFAULT_MIN_ROWS:
                results.append({"wave": wave, "mode": mode, "rows": 0, "status": "FAILED (not enough price history)"})
            else:
                path = write_perf_csv(wave, mode, perf)
                results.append({"wave": wave, "mode": mode, "rows": len(perf), "status": f"OK → {path}"})
        except Exception as e:
            results.append({"wave": wave, "mode": mode, "rows": 0, "status": f"ERROR: {e}"})

        prog.progress(int(i / total * 100))

    st.success("Backfill complete.")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.warning(
        "Next step: go back to the main Console page, pick the same Wave + Mode, and you should see History Rows populate.\n"
        "If a Wave still shows 0 rows, it likely has tickers yfinance can’t price (symbol format) or weights file issues."
    )