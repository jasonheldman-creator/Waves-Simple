# app.py — WAVES Intelligence™ — Institutional Console (Full Console + Built-in Backfill)
# FULL FILE (NO PATCHES)
#
# What this restores:
#   ✅ Full Console sections ("meat"): Summary, Holdings, Performance, Risk Lab, Correlation, Alpha Heatmap
#   ✅ Built-in Backfill Missing History tool (inside the same app)
#   ✅ Uses wave_weights.csv + wave_config.csv
#   ✅ Reads existing logs/performance/<wave>__<mode>_performance_daily.csv if present
#   ✅ Falls back to weights+prices build if logs are missing
#
# Notes:
#   • This file is designed to work standalone (no dependency on a custom engine).
#   • If yfinance is unavailable, it will tell you clearly.

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# ----------------------------
# Settings / Paths
# ----------------------------
ROOT = os.getcwd()
LOG_DIR = os.path.join(ROOT, "logs", "performance")
WEIGHTS_PATH = os.path.join(ROOT, "wave_weights.csv")
WAVE_CONFIG_PATH = os.path.join(ROOT, "wave_config.csv")

DEFAULT_LOOKBACK_DAYS = 520  # ~2 calendar years
DEFAULT_BM_FALLBACK = "SPY"

MODE_CHOICES = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

MODE_ALIASES = {
    "Standard": ["standard", "STANDARD", "Std", "STD", "Standard"],
    "Alpha-Minus-Beta": ["alpha-minus-beta", "Alpha-Minus-Beta", "AMB", "Alpha Minus Beta"],
    "Private Logic": ["private logic", "Private Logic", "PL", "PrivateLogic", "Private_Logic"],
}

# ----------------------------
# Helpers
# ----------------------------
def safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\s\-\.\&]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s

def normalize_mode(m: str) -> str:
    if not m:
        return "Standard"
    mm = m.strip()
    for canon, aliases in MODE_ALIASES.items():
        for a in aliases:
            if mm.lower() == a.lower():
                return canon
    return mm

def google_quote_url(ticker: str) -> str:
    t = ticker.replace(".", "-").upper().strip()
    return f"https://www.google.com/finance/quote/{t}"

def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)

def read_csv_safely(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ----------------------------
# Load config + weights
# ----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def load_wave_config() -> pd.DataFrame:
    if not os.path.exists(WAVE_CONFIG_PATH):
        return pd.DataFrame(columns=["wave", "benchmark", "mode", "beta_target"])
    df = pd.read_csv(WAVE_CONFIG_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    # expected: wave, benchmark, mode, beta_target
    if "wave" not in df.columns:
        df["wave"] = ""
    if "benchmark" not in df.columns:
        df["benchmark"] = DEFAULT_BM_FALLBACK
    if "mode" not in df.columns:
        df["mode"] = "Standard"
    if "beta_target" not in df.columns:
        df["beta_target"] = np.nan

    df["wave"] = df["wave"].astype(str).str.strip()
    df["benchmark"] = df["benchmark"].astype(str).str.strip()
    df["mode"] = df["mode"].astype(str).map(normalize_mode)
    df["beta_target"] = pd.to_numeric(df["beta_target"], errors="coerce")
    return df

@st.cache_data(show_spinner=False, ttl=60)
def load_weights() -> pd.DataFrame:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing {WEIGHTS_PATH}. Please add wave_weights.csv at repo root.")
    df = pd.read_csv(WEIGHTS_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    # expected: wave, ticker, weight
    for col in ["wave", "ticker", "weight"]:
        if col not in df.columns:
            raise ValueError(f"wave_weights.csv must include columns: wave,ticker,weight. Missing: {col}")
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df[(df["ticker"].str.len() > 0) & (df["weight"] != 0)]
    return df

def get_wave_list(weights_df: pd.DataFrame) -> list[str]:
    waves = sorted(weights_df["wave"].dropna().unique().tolist())
    return waves

def pick_benchmark(cfg: pd.DataFrame, wave: str, mode: str) -> str:
    mode = normalize_mode(mode)
    sub = cfg[(cfg["wave"] == wave) & (cfg["mode"] == mode)]
    if len(sub) > 0 and str(sub.iloc[0]["benchmark"]).strip():
        return str(sub.iloc[0]["benchmark"]).strip()
    sub2 = cfg[(cfg["wave"] == wave)]
    if len(sub2) > 0 and str(sub2.iloc[0]["benchmark"]).strip():
        return str(sub2.iloc[0]["benchmark"]).strip()
    return DEFAULT_BM_FALLBACK

def pick_beta_target(cfg: pd.DataFrame, wave: str, mode: str) -> float:
    mode = normalize_mode(mode)
    sub = cfg[(cfg["wave"] == wave) & (cfg["mode"] == mode)]
    if len(sub) and pd.notna(sub.iloc[0].get("beta_target", np.nan)):
        return float(sub.iloc[0]["beta_target"])
    # reasonable defaults if not specified
    if mode == "Alpha-Minus-Beta":
        return 0.80
    if mode == "Private Logic":
        return 1.10
    return 0.90

# ----------------------------
# Market data
# ----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 20)
def fetch_prices(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment. Add it to requirements.txt.")
    tks = sorted(list({t.strip().upper() for t in tickers if t and isinstance(t, str)}))
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

    # Normalize to simple close px dataframe
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        else:
            # take first field
            px = df.xs(df.columns.levels[0][0], axis=1, level=0)
    else:
        px = df.copy()

    px.index = pd.to_datetime(px.index)
    px = px.sort_index().dropna(how="all")
    # Ensure columns are tickers
    px.columns = [c.strip().upper() for c in px.columns]
    return px

def build_nav_from_weights(weights_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    tickers = weights_df["ticker"].astype(str).str.upper().tolist()
    w = weights_df.groupby("ticker", as_index=True)["weight"].sum().astype(float)
    w = w[w != 0].copy()
    if len(w) == 0:
        return pd.DataFrame()

    # normalize weights
    s = float(w.sum())
    if s <= 0:
        return pd.DataFrame()
    w = w / s

    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(lookback_days))

    px = fetch_prices(tickers=list(w.index), start=start, end=end + timedelta(days=1))
    if px.empty:
        return pd.DataFrame()

    common = [c for c in px.columns if c in w.index]
    if not common:
        return pd.DataFrame()

    px = px[common].dropna(how="all")
    px = px.ffill().dropna(how="all")
    if len(px) < 5:
        return pd.DataFrame()

    rets = px.pct_change().fillna(0.0)
    port_ret = (rets.values * w.loc[common].values.reshape(1, -1)).sum(axis=1)
    nav = (1.0 + port_ret).cumprod() * 100.0

    out = pd.DataFrame({"date": px.index, "nav": nav, "ret": port_ret})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.reset_index(drop=True)

def build_benchmark_nav(bm_ticker: str, lookback_days: int) -> pd.DataFrame:
    bm = bm_ticker.strip().upper()
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(lookback_days))
    px = fetch_prices([bm], start=start, end=end + timedelta(days=1))
    if px.empty or bm not in px.columns:
        return pd.DataFrame()
    s = px[bm].dropna().ffill()
    rets = s.pct_change().fillna(0.0)
    nav = (1.0 + rets).cumprod() * 100.0
    out = pd.DataFrame({"date": s.index, "bm_nav": nav, "bm_ret": rets.values})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.reset_index(drop=True)

def merge_wave_bm(wave_nav: pd.DataFrame, bm_nav: pd.DataFrame) -> pd.DataFrame:
    if wave_nav.empty or bm_nav.empty:
        return pd.DataFrame()
    df = pd.merge(wave_nav, bm_nav, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    df["alpha_ret"] = df["ret"] - df["bm_ret"]
    df["alpha_nav"] = (1.0 + df["alpha_ret"]).cumprod() * 100.0
    return df

def perf_path(wave: str, mode: str) -> str:
    ensure_dirs()
    return os.path.join(LOG_DIR, f"{safe_slug(wave)}__{safe_slug(mode)}_performance_daily.csv")

def load_or_build_history(weights_df: pd.DataFrame, cfg: pd.DataFrame, wave: str, mode: str, lookback_days: int) -> tuple[pd.DataFrame, str]:
    """
    Returns (history_df, source_label)
    history_df columns: date, nav, ret, bm_nav, bm_ret, alpha_ret, alpha_nav
    """
    mode = normalize_mode(mode)
    p = perf_path(wave, mode)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip().lower() for c in df.columns]
            # normalize expected cols
            # allow either already merged or minimal
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            # try to standardize names
            rename_map = {}
            if "bmnav" in df.columns: rename_map["bmnav"] = "bm_nav"
            if "bmret" in df.columns: rename_map["bmret"] = "bm_ret"
            if "alphanav" in df.columns: rename_map["alphanav"] = "alpha_nav"
            if "alpharet" in df.columns: rename_map["alpharet"] = "alpha_ret"
            df = df.rename(columns=rename_map)

            # if missing alpha, rebuild from ret/bm_ret
            if "alpha_ret" not in df.columns and "ret" in df.columns and "bm_ret" in df.columns:
                df["alpha_ret"] = df["ret"] - df["bm_ret"]
                df["alpha_nav"] = (1.0 + df["alpha_ret"]).cumprod() * 100.0

            return df, "logs/performance (saved history)"
        except Exception:
            pass

    # Build from live weights+prices
    wsub = weights_df[weights_df["wave"] == wave].copy()
    bm = pick_benchmark(cfg, wave, mode)
    wave_nav = build_nav_from_weights(wsub, lookback_days=lookback_days)
    bm_nav = build_benchmark_nav(bm, lookback_days=lookback_days)
    merged = merge_wave_bm(wave_nav, bm_nav)
    return merged, "weights+prices (live build)"

def pct_from_nav(df: pd.DataFrame, nav_col: str, days: int) -> float | None:
    if df.empty or nav_col not in df.columns:
        return None
    s = df[nav_col].dropna()
    if len(s) < 2:
        return None
    if days <= 0:
        return None
    if len(s) <= days:
        # use first value
        base = float(s.iloc[0])
        last = float(s.iloc[-1])
    else:
        base = float(s.iloc[-(days + 1)])
        last = float(s.iloc[-1])
    if base == 0:
        return None
    return (last / base) - 1.0

def realized_beta(df: pd.DataFrame, window: int = 252) -> float | None:
    if df.empty or "ret" not in df.columns or "bm_ret" not in df.columns:
        return None
    x = df["bm_ret"].astype(float).values
    y = df["ret"].astype(float).values
    if len(x) < 10:
        return None
    if len(x) > window:
        x = x[-window:]
        y = y[-window:]
    vx = np.var(x)
    if vx <= 1e-12:
        return None
    cov = np.cov(x, y)[0, 1]
    return float(cov / vx)

def apply_mode_scaling(df: pd.DataFrame, beta_target: float) -> tuple[pd.DataFrame, float]:
    """
    Optional: scale wave returns to approximate beta_target.
    Keeps benchmark unchanged.
    """
    df2 = df.copy()
    b = realized_beta(df2)
    if b is None or b <= 0.05:
        return df2, 1.0
    k = float(beta_target) / float(b)
    # cap to avoid extreme explosions
    k = max(0.25, min(2.5, k))
    df2["ret_scaled"] = df2["ret"] * k
    df2["nav_scaled"] = (1.0 + df2["ret_scaled"]).cumprod() * 100.0
    df2["alpha_ret_scaled"] = df2["ret_scaled"] - df2["bm_ret"]
    df2["alpha_nav_scaled"] = (1.0 + df2["alpha_ret_scaled"]).cumprod() * 100.0
    return df2, k

# ----------------------------
# UI
# ----------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ — Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("WAVES Intelligence™ — Institutional Console")
st.caption("Production console + built-in Backfill tool. Safe: does not change engine logic.")

if yf is None:
    st.error("yfinance is not available. Add `yfinance` to requirements.txt then reboot the app.")
    st.stop()

try:
    weights_df = load_weights()
except Exception as e:
    st.error(str(e))
    st.stop()

cfg_df = load_wave_config()
waves = get_wave_list(weights_df)

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Console", "Backfill Missing History"], index=0)

scan_mode = st.sidebar.toggle("Scan Mode (iPhone fast)", value=True)
if scan_mode:
    st.sidebar.caption("Scan Mode hides heavier charts for speed.")

# Top controls (common)
colA, colB, colC, colD = st.columns([3, 2, 2, 2])
with colA:
    sel_wave = st.selectbox("Wave", waves, index=0 if waves else None)
with colB:
    sel_mode = st.selectbox("Mode", MODE_CHOICES, index=0)
with colC:
    lookback_days = st.number_input("History lookback (days)", min_value=60, max_value=3650, value=DEFAULT_LOOKBACK_DAYS, step=20)
with colD:
    refresh = st.button("🔄 Refresh")

if refresh:
    st.cache_data.clear()

if not sel_wave:
    st.warning("No Waves found in wave_weights.csv.")
    st.stop()

bm_ticker = pick_benchmark(cfg_df, sel_wave, sel_mode)
beta_target = pick_beta_target(cfg_df, sel_wave, sel_mode)

hist, source_label = load_or_build_history(weights_df, cfg_df, sel_wave, sel_mode, int(lookback_days))

if hist.empty:
    st.error(
        f"No history could be built for {sel_wave} ({sel_mode}). "
        f"Try Backfill Missing History or check tickers in wave_weights.csv."
    )
    st.stop()

# Mode scaling (optional, but helps match beta targets)
hist_scaled, scale_k = apply_mode_scaling(hist, beta_target=beta_target)

# Pinned Summary
st.subheader("Summary (Pinned)")
st.caption(f"Mode: **{sel_mode}** • Benchmark: **{bm_ticker}** • Beta target: **{beta_target:.2f}** • Source: **{source_label}** • Scale k: **{scale_k:.2f}**")

intraday = pct_from_nav(hist_scaled, "nav_scaled", 1)
r30 = pct_from_nav(hist_scaled, "nav_scaled", 30)
r60 = pct_from_nav(hist_scaled, "nav_scaled", 60)
r1y = pct_from_nav(hist_scaled, "nav_scaled", 252)

a1d = pct_from_nav(hist_scaled, "alpha_nav_scaled", 1)
a30 = pct_from_nav(hist_scaled, "alpha_nav_scaled", 30)
a60 = pct_from_nav(hist_scaled, "alpha_nav_scaled", 60)
a1y = pct_from_nav(hist_scaled, "alpha_nav_scaled", 252)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Intraday Return", f"{(intraday or 0)*100:.2f}%", delta=f"Alpha {((a1d or 0)*100):.2f}%")
m2.metric("30D Return", f"{(r30 or 0)*100:.2f}%", delta=f"Alpha {((a30 or 0)*100):.2f}%")
m3.metric("60D Return", f"{(r60 or 0)*100:.2f}%", delta=f"Alpha {((a60 or 0)*100):.2f}%")
m4.metric("1Y Return", f"{(r1y or 0)*100:.2f}%", delta=f"Alpha {((a1y or 0)*100):.2f}%")

st.divider()

# ----------------------------
# Console Page
# ----------------------------
if page == "Console":
    # Tabs (the missing “meat”)
    tabs = st.tabs(["Overview", "Holdings", "Performance", "Risk Lab", "Correlation", "Alpha Heatmap"])

    # ---- Overview
    with tabs[0]:
        st.markdown("### Overview")
        if not scan_mode:
            st.line_chart(pd.DataFrame({"Wave NAV": hist_scaled["nav_scaled"].values}, index=pd.to_datetime(hist_scaled["date"])))
            st.line_chart(pd.DataFrame({"Alpha NAV": hist_scaled["alpha_nav_scaled"].values}, index=pd.to_datetime(hist_scaled["date"])))

        st.markdown("#### History Table (latest 20)")
        st.dataframe(hist_scaled.tail(20), use_container_width=True)

    # ---- Holdings
    with tabs[1]:
        st.markdown("### Top-10 Holdings (clickable)")
        wsub = weights_df[weights_df["wave"] == sel_wave].copy()
        wsub = wsub.groupby("ticker", as_index=False)["weight"].sum()
        wsub = wsub[wsub["weight"] != 0].copy()
        wsum = float(wsub["weight"].sum()) if len(wsub) else 0.0
        if wsum > 0:
            wsub["weight"] = wsub["weight"] / wsum
        wsub = wsub.sort_values("weight", ascending=False).reset_index(drop=True)

        top10 = wsub.head(10).copy()
        top10["ticker"] = top10["ticker"].astype(str).str.upper()

        # clickable ticker column (works well on desktop; on mobile it may show markdown)
        top10["ticker_link"] = top10["ticker"].apply(lambda t: f"[{t}]({google_quote_url(t)})")
        st.caption("Weights normalized within each wave.")
        st.dataframe(top10[["ticker_link", "weight"]], use_container_width=True)

        st.markdown("### Wave Composition")
        st.dataframe(wsub.rename(columns={"ticker": "ticker", "weight": "weight"}), use_container_width=True)

    # ---- Performance
    with tabs[2]:
        st.markdown("### Performance")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Wave NAV (scaled)")
            if not scan_mode:
                st.line_chart(pd.DataFrame({"nav_scaled": hist_scaled["nav_scaled"].values}, index=pd.to_datetime(hist_scaled["date"])))
            else:
                st.dataframe(hist_scaled[["date", "nav_scaled"]].tail(30), use_container_width=True)
        with c2:
            st.markdown("#### Alpha NAV (scaled)")
            if not scan_mode:
                st.line_chart(pd.DataFrame({"alpha_nav_scaled": hist_scaled["alpha_nav_scaled"].values}, index=pd.to_datetime(hist_scaled["date"])))
            else:
                st.dataframe(hist_scaled[["date", "alpha_nav_scaled"]].tail(30), use_container_width=True)

        st.markdown("#### Market Intel")
        intel = pd.DataFrame(
            {
                "symbol": ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"],
                "label": ["S&P 500 (SPY)", "Nasdaq 100 (QQQ)", "Russell 2000 (IWM)", "Long Treasuries (TLT)", "Gold (GLD)", "Bitcoin (BTC-USD)", "VIX (^VIX)", "10Y Yield (^TNX)"],
            }
        )
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=10)
            px = fetch_prices(intel["symbol"].tolist(), start, end + timedelta(days=1))
            rows = []
            for sym, label in zip(intel["symbol"], intel["label"]):
                if sym in px.columns and px[sym].dropna().shape[0] >= 2:
                    s = px[sym].dropna()
                    last = float(s.iloc[-1])
                    d1 = float(s.iloc[-1] / s.iloc[-2] - 1.0) * 100.0
                    rows.append({"symbol": sym, "label": label, "last": last, "1d%": d1})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.info(f"Market Intel unavailable: {e}")

    # ---- Risk Lab
    with tabs[3]:
        st.markdown("### Risk Lab")
        b = realized_beta(hist_scaled)
        st.metric("Beta (vs benchmark)", f"{(b or 0):.2f}")

        # drawdown
        nav = hist_scaled["nav_scaled"].astype(float).values
        peak = np.maximum.accumulate(nav)
        dd = (nav / peak) - 1.0
        dd_df = pd.DataFrame({"drawdown": dd}, index=pd.to_datetime(hist_scaled["date"]))

        if not scan_mode:
            st.line_chart(dd_df)
        else:
            st.dataframe(dd_df.tail(60), use_container_width=True)

        # rolling vol (21d)
        rets = pd.Series(hist_scaled["ret_scaled"].astype(float).values, index=pd.to_datetime(hist_scaled["date"]))
        roll_vol = rets.rolling(21).std() * math.sqrt(252)
        if not scan_mode:
            st.line_chart(pd.DataFrame({"rolling_vol_21d": roll_vol}))
        else:
            st.dataframe(pd.DataFrame({"rolling_vol_21d": roll_vol}).tail(60), use_container_width=True)

    # ---- Correlation (All Waves)
    with tabs[4]:
        st.markdown("### Correlation Matrix (All Waves)")
        # Build nav series for each wave (scaled standard mode by each wave's config)
        series = {}
        for w in waves:
            h, _ = load_or_build_history(weights_df, cfg_df, w, sel_mode, int(lookback_days))
            if h.empty:
                continue
            bt = pick_beta_target(cfg_df, w, sel_mode)
            hs, _k = apply_mode_scaling(h, beta_target=bt)
            s = pd.Series(hs["ret_scaled"].astype(float).values, index=pd.to_datetime(hs["date"]))
            series[w] = s

        if len(series) < 2:
            st.info("Not enough waves with history to compute correlation.")
        else:
            df = pd.concat(series, axis=1).dropna()
            corr = df.corr()
            st.dataframe(corr, use_container_width=True)

    # ---- Alpha Heatmap
    with tabs[5]:
        st.markdown("### Alpha Heatmap (All Waves x Timeframe)")
        windows = [1, 30, 60, 252]
        labels = ["1D", "30D", "60D", "1Y"]

        rows = []
        for w in waves:
            h, _ = load_or_build_history(weights_df, cfg_df, w, sel_mode, int(lookback_days))
            if h.empty:
                continue
            bt = pick_beta_target(cfg_df, w, sel_mode)
            hs, _k = apply_mode_scaling(h, beta_target=bt)
            for d, lab in zip(windows, labels):
                a = pct_from_nav(hs, "alpha_nav_scaled", d)
                rows.append({"wave": w, "window": lab, "alpha": (a or 0.0)})

        if not rows:
            st.info("No alpha rows available.")
        else:
            pivot = pd.DataFrame(rows).pivot(index="wave", columns="window", values="alpha")
            pivot = pivot.reindex(columns=labels)
            st.dataframe((pivot * 100.0).round(2), use_container_width=True)

    st.caption("Tip: If a wave shows no history, use Backfill Missing History for that Wave+Mode.")

# ----------------------------
# Backfill Page (inside same app)
# ----------------------------
else:
    st.markdown("## 🧱 Backfill Missing History (Restore the “Meat”)")
    st.caption("This writes logs/performance/<wave>__<mode>_performance_daily.csv using weights+prices. Safe: does not change engine logic.")

    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        selected_waves = st.multiselect("Waves to backfill", waves, default=[sel_wave])
    with col2:
        mode_bf = st.selectbox("Mode", MODE_CHOICES, index=MODE_CHOICES.index(sel_mode) if sel_mode in MODE_CHOICES else 0)
    with col3:
        lookback_bf = st.number_input("Lookback (calendar days)", min_value=60, max_value=3650, value=int(lookback_days), step=20)

    st.info(
        f"Will write files to: {LOG_DIR}\n\n"
        "Expected output naming: logs/performance/<wave>__<mode>_performance_daily.csv"
    )

    run = st.button("🚀 Run Backfill", use_container_width=True)

    if run:
        if not selected_waves:
            st.warning("Select at least one Wave.")
            st.stop()

        ensure_dirs()
        results = []
        prog = st.progress(0)
        total = len(selected_waves)

        for i, w in enumerate(selected_waves, start=1):
            try:
                bm = pick_benchmark(cfg_df, w, mode_bf)
                wsub = weights_df[weights_df["wave"] == w].copy()
                wave_nav = build_nav_from_weights(wsub, lookback_days=int(lookback_bf))
                bm_nav = build_benchmark_nav(bm, lookback_days=int(lookback_bf))
                merged = merge_wave_bm(wave_nav, bm_nav)

                if merged.empty or len(merged) < 20:
                    results.append({"wave": w, "mode": mode_bf, "rows": 0, "status": "FAILED (no data)"} )
                else:
                    p = perf_path(w, mode_bf)
                    merged.to_csv(p, index=False)
                    results.append({"wave": w, "mode": mode_bf, "rows": len(merged), "status": "OK", "file": p})

            except Exception as e:
                results.append({"wave": w, "mode": mode_bf, "rows": 0, "status": f"ERROR: {e}"})

            prog.progress(int(i / total * 100))

        st.success("Backfill complete.")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.caption("Next: go back to Console and press Refresh. You should see History populate from logs/performance.")