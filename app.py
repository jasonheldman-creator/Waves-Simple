# app.py — WAVES Intelligence™ Institutional Console (Production)
# FULL FILE (no patches) — SAFE RESET
#
# What this fixes:
# ✅ app.py is the MAIN console again (Backfill tool is no longer the entrypoint)
# ✅ Backfill Missing History is included as a built-in tool (no /pages folder required)
#
# What it includes:
# • Pinned Summary Bar (scan-first)
# • Scan Mode (fast iPhone demo)
# • Wave + Mode selection
# • Intraday / 30D / 60D / 1Y returns + Alpha Capture vs Benchmark
# • Top-10 holdings with Google quote links
# • Market Intel (SPY/QQQ/IWM/TLT/GLD/BTC/VIX/TNX)
# • Correlation Matrix (Wave returns)
# • Rolling Vol / Rolling Alpha
# • Drawdown Monitor
# • Backfill Missing History tool (writes logs/performance/*.csv)
#
# Notes:
# • Does NOT require waves_engine.py to run. If you have it, we can integrate later.
# • Uses yfinance for pricing. If yfinance fails, app still loads but backfill/prices won't.

from __future__ import annotations

import os
import re
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# Paths / Constants
# -----------------------------
ROOT = os.getcwd()
LOG_DIR = os.path.join(ROOT, "logs", "performance")
WEIGHTS_PATH = os.path.join(ROOT, "wave_weights.csv")
CONFIG_PATH = os.path.join(ROOT, "wave_config.csv")

DEFAULT_LOOKBACK_DAYS = 520  # ~2 calendar years
DEFAULT_ROLL_WINDOW = 30

MODE_ALIASES = {
    "Standard": ["Standard", "standard", "STD", "Std"],
    "Alpha-Minus-Beta": ["Alpha-Minus-Beta", "alpha-minus-beta", "AMB", "amb", "Alpha Minus Beta"],
    "Private Logic": ["Private Logic", "private logic", "PL", "pl", "PrivateLogic"],
}

MARKET_INTEL = [
    ("SPY", "S&P 500 (SPY)"),
    ("QQQ", "Nasdaq 100 (QQQ)"),
    ("IWM", "Russell 2000 (IWM)"),
    ("TLT", "Long Treasuries (TLT)"),
    ("GLD", "Gold (GLD)"),
    ("BTC-USD", "Bitcoin (BTC-USD)"),
    ("^VIX", "VIX (^VIX)"),
    ("^TNX", "10Y Yield (^TNX)"),
]

# -----------------------------
# Helpers
# -----------------------------
def safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\s\-\.\&/]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s

def normalize_mode(m: str) -> str:
    if not m:
        return "Standard"
    m_strip = m.strip()
    for canon, aliases in MODE_ALIASES.items():
        if m_strip == canon or m_strip in aliases:
            return canon
    return m_strip

def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:,.2f}%"

def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:,.2f}"

def google_quote_url(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    # Simple mapping for common formats (best effort)
    # BRK.B often works as BRK.B; yfinance uses BRK-B
    return f"https://www.google.com/finance/quote/{t}:NASDAQ"

def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)

@st.cache_data(ttl=60 * 20, show_spinner=False)
def yf_download_prices(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    tks = sorted(list({(t or "").strip().upper() for t in tickers if (t or "").strip()}))
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

    # Normalize to Close-only DataFrame with columns = tickers
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        else:
            # try first field
            px = df[df.columns.get_level_values(0)[0]].copy()
        px.columns = [c.upper() for c in px.columns]
    else:
        # Single ticker returns Series-like; handle
        if "Close" in df.columns:
            px = df[["Close"]].copy()
            px.columns = [tks[0]]
        else:
            px = df.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index().dropna(how="all")
    return px

def load_weights() -> pd.DataFrame:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing {WEIGHTS_PATH}")
    df = pd.read_csv(WEIGHTS_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["wave", "ticker", "weight"]:
        if col not in df.columns:
            raise ValueError("wave_weights.csv must have columns: wave, ticker, weight")
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df[(df["ticker"].str.len() > 0) & (df["weight"] != 0)]
    return df

def load_wave_config() -> pd.DataFrame:
    if not os.path.exists(CONFIG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(CONFIG_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def list_waves(weights_df: pd.DataFrame) -> List[str]:
    waves = sorted(weights_df["wave"].dropna().unique().tolist())
    return waves

def get_wave_holdings(weights_df: pd.DataFrame, wave: str) -> pd.DataFrame:
    w = weights_df[weights_df["wave"] == wave][["ticker", "weight"]].copy()
    w["ticker"] = w["ticker"].astype(str).str.strip().str.upper()
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
    # Deduplicate by summing
    w = w.groupby("ticker", as_index=False)["weight"].sum()
    # Normalize to sum=1
    s = float(w["weight"].sum())
    if s > 0:
        w["weight"] = w["weight"] / s
    w = w.sort_values("weight", ascending=False)
    return w

def perf_path(wave: str, mode: str) -> str:
    ensure_dirs()
    return os.path.join(LOG_DIR, f"{safe_slug(wave)}__{safe_slug(mode)}_performance_daily.csv")

def read_perf_csv(wave: str, mode: str) -> pd.DataFrame:
    path = perf_path(wave, mode)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.set_index("date")
    return df

def write_perf_csv(wave: str, mode: str, df: pd.DataFrame) -> str:
    ensure_dirs()
    out = df.copy()
    out = out.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    path = perf_path(wave, mode)
    out.to_csv(path, index=False)
    return path

def build_nav_from_weights(weights_df_wave: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    # weights_df_wave columns: ticker, weight (normalized)
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(lookback_days))
    tickers = weights_df_wave["ticker"].astype(str).str.upper().tolist()
    px = yf_download_prices(tickers, start, end)
    if px.empty:
        return pd.DataFrame()
    # Align weights to priced tickers
    w = weights_df_wave.set_index("ticker")["weight"]
    common = [c for c in px.columns if c in w.index]
    if not common:
        return pd.DataFrame()
    w2 = w.loc[common].copy()
    s = float(w2.sum())
    if s <= 0:
        return pd.DataFrame()
    w2 = w2 / s
    px2 = px[common].copy().dropna(how="all")
    rets = px2.pct_change().fillna(0.0)
    port_ret = (rets.values * w2.values).sum(axis=1)
    nav = (1.0 + pd.Series(port_ret, index=rets.index)).cumprod() * 100.0
    out = pd.DataFrame({"nav": nav})
    out["ret"] = out["nav"].pct_change().fillna(0.0)
    return out

def compute_return_windows(nav_df: pd.DataFrame) -> Dict[str, float]:
    if nav_df is None or nav_df.empty or "nav" not in nav_df.columns:
        return {"intraday": np.nan, "30d": np.nan, "60d": np.nan, "1y": np.nan}
    nav = nav_df["nav"].dropna()
    if len(nav) < 3:
        return {"intraday": np.nan, "30d": np.nan, "60d": np.nan, "1y": np.nan}

    def window_ret(days: int) -> float:
        if len(nav) < 2:
            return np.nan
        # use last available vs closest past date
        end_dt = nav.index[-1]
        start_dt = end_dt - pd.Timedelta(days=days)
        past = nav.loc[nav.index <= start_dt]
        if past.empty:
            return np.nan
        return float(nav.iloc[-1] / past.iloc[-1] - 1.0)

    intraday = float(nav.iloc[-1] / nav.iloc[-2] - 1.0) if len(nav) >= 2 else np.nan
    return {
        "intraday": intraday,
        "30d": window_ret(30),
        "60d": window_ret(60),
        "1y": window_ret(365),
    }

def get_benchmark_for_wave(wave: str, cfg: pd.DataFrame) -> str:
    # best effort: use wave_config.csv if it has wave->benchmark
    if cfg is not None and not cfg.empty:
        cols = set(cfg.columns)
        if "wave" in cols and "benchmark" in cols:
            sub = cfg[cfg["wave"].astype(str).str.strip() == str(wave).strip()]
            if not sub.empty:
                b = str(sub.iloc[0]["benchmark"]).strip()
                if b:
                    return b
    # default heuristic:
    w = wave.lower()
    if "crypto" in w:
        return "BTC-USD"
    if "small" in w and "cap" in w:
        return "IWM"
    if "gold" in w:
        return "GLD"
    if "income" in w or "muni" in w or "bond" in w:
        return "TLT"
    if "tech" in w or "ai" in w or "cloud" in w or "quantum" in w:
        return "QQQ"
    return "SPY"

def alpha_capture(wave_ret: float, bm_ret: float) -> float:
    if any(map(lambda x: x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))), [wave_ret, bm_ret])):
        return np.nan
    return float(wave_ret - bm_ret)

def compute_drawdown(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    return dd

# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ — Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .big-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.25rem; }
    .subtle { opacity: 0.85; }
    .pill { padding: 0.25rem 0.6rem; border-radius: 999px; display:inline-block; margin-right: 0.35rem; }
    .good { background: rgba(0,255,160,0.12); }
    .bad { background: rgba(255,80,80,0.12); }
    .neutral { background: rgba(160,160,160,0.12); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">WAVES Intelligence™ — Institutional Console</div>', unsafe_allow_html=True)
st.caption("Production console + built-in Backfill tool. Safe: does not change engine logic.")

# Load data (weights/config)
weights_df = None
cfg_df = None
weights_err = None
try:
    weights_df = load_weights()
    cfg_df = load_wave_config()
except Exception as e:
    weights_err = str(e)

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Console", "Backfill Missing History"], index=0)
    st.divider()
    scan_mode = st.toggle("Scan Mode (iPhone fast)", value=True)
    st.caption("Scan Mode hides heavier charts for speed.")

if weights_err:
    st.error(f"Could not load wave_weights.csv: {weights_err}")
    st.stop()

waves = list_waves(weights_df)

# -----------------------------
# Page: Backfill
# -----------------------------
if page == "Backfill Missing History":
    st.subheader("🧱 Backfill Missing History (Restore the “Meat”)")
    st.caption("Builds daily performance history files used by the console. Writes to logs/performance/.")
    if yf is None:
        st.error("yfinance is not available in this environment. Add yfinance to requirements.txt.")
        st.stop()

    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        selected_waves = st.multiselect("Waves to backfill", waves, default=waves[: min(8, len(waves))])
    with colB:
        mode = st.selectbox("Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic"], index=0)
        mode = normalize_mode(mode)
    with colC:
        lookback = st.number_input("Lookback (calendar days)", min_value=60, max_value=2500, value=DEFAULT_LOOKBACK_DAYS, step=10)

    st.info(
        f"Will write files to: {LOG_DIR}\n\n"
        f"Expected output naming: logs/performance/<Wave>__<Mode>_performance_daily.csv"
    )

    run = st.button("🚀 Run Backfill", use_container_width=True)
    if run:
        ensure_dirs()
        results = []
        prog = st.progress(0)
        total = max(1, len(selected_waves))
        for i, wave in enumerate(selected_waves, start=1):
            try:
                w = get_wave_holdings(weights_df, wave)
                perf = build_nav_from_weights(w, int(lookback))
                if perf is None or perf.empty or len(perf) < 5:
                    results.append({"wave": wave, "mode": mode, "rows": 0, "status": "NO DATA (pricing/weights)"})
                else:
                    path = write_perf_csv(wave, mode, perf)
                    results.append({"wave": wave, "mode": mode, "rows": int(len(perf)), "status": f"WROTE: {os.path.basename(path)}"})
            except Exception as e:
                results.append({"wave": wave, "mode": mode, "rows": 0, "status": f"ERROR: {e}"})
            prog.progress(min(1.0, i / total))

        st.success("Backfill complete.")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.warning(
            "Next step (recommended): go back to the Console page, pick the same Wave + Mode, and verify History Rows + "
            "returns/alpha populate. If a wave shows 0 rows, it likely has tickers yfinance can’t price or weights issues."
        )

    st.stop()

# -----------------------------
# Page: Console
# -----------------------------
# Console controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    wave = st.selectbox("Wave", waves, index=0 if waves else None)
with col2:
    mode = st.selectbox("Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic"], index=0)
    mode = normalize_mode(mode)
with col3:
    lookback_days = st.number_input("History lookback (days)", min_value=60, max_value=2500, value=DEFAULT_LOOKBACK_DAYS, step=10)
with col4:
    refresh = st.button("🔄 Refresh", use_container_width=True)

# Load history (prefer logs/performance; fallback to weights+prices)
bench = get_benchmark_for_wave(wave, cfg_df)

hist = read_perf_csv(wave, mode)
hist_source = "logs/performance"
if hist.empty:
    # fallback: build on the fly from weights
    hist_source = "weights+prices (live build)"
    if yf is not None:
        w = get_wave_holdings(weights_df, wave)
        hist = build_nav_from_weights(w, int(lookback_days))
        if not hist.empty:
            hist.index = pd.to_datetime(hist.index)
    else:
        hist = pd.DataFrame()

# Benchmark history
bm_hist = pd.DataFrame()
if yf is not None and bench:
    try:
        end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=int(lookback_days))
        bm_px = yf_download_prices([bench], start, end)
        if not bm_px.empty:
            s = bm_px.iloc[:, 0].dropna()
            nav = (s / s.iloc[0]) * 100.0
            bm_hist = pd.DataFrame({"nav": nav})
            bm_hist["ret"] = bm_hist["nav"].pct_change().fillna(0.0)
    except Exception:
        bm_hist = pd.DataFrame()

# Compute headline metrics
wave_ret = compute_return_windows(hist) if not hist.empty else {"intraday": np.nan, "30d": np.nan, "60d": np.nan, "1y": np.nan}
bm_ret = compute_return_windows(bm_hist) if not bm_hist.empty else {"intraday": np.nan, "30d": np.nan, "60d": np.nan, "1y": np.nan}

alpha = {k: alpha_capture(wave_ret.get(k, np.nan), bm_ret.get(k, np.nan)) for k in ["intraday", "30d", "60d", "1y"]}

# Sticky / pinned summary (top)
st.markdown("### Summary (Pinned)")
s1, s2, s3, s4, s5 = st.columns([1.3, 1, 1, 1, 1.3])
with s1:
    st.metric("Wave", wave)
    st.caption(f"Mode: **{mode}**  • Source: **{hist_source}**")
with s2:
    st.metric("Intraday Return", fmt_pct(wave_ret["intraday"]))
    st.caption(f"Alpha: {fmt_pct(alpha['intraday'])}")
with s3:
    st.metric("30D Return", fmt_pct(wave_ret["30d"]))
    st.caption(f"Alpha: {fmt_pct(alpha['30d'])}")
with s4:
    st.metric("60D Return", fmt_pct(wave_ret["60d"]))
    st.caption(f"Alpha: {fmt_pct(alpha['60d'])}")
with s5:
    st.metric("1Y Return", fmt_pct(wave_ret["1y"]))
    st.caption(f"Alpha: {fmt_pct(alpha['1y'])} • Benchmark: **{bench}**")

st.divider()

# Holdings + Top-10
hold = get_wave_holdings(weights_df, wave)
top10 = hold.head(10).copy()
top10["google"] = top10["ticker"].apply(lambda t: google_quote_url(t))
top10["ticker_link"] = top10.apply(lambda r: f"[{r['ticker']}]({r['google']})", axis=1)

cA, cB = st.columns([1.2, 1])
with cA:
    st.subheader("Top-10 Holdings (clickable)")
    show = top10[["ticker_link", "weight"]].copy()
    show = show.rename(columns={"ticker_link": "ticker", "weight": "weight (norm)"})
    st.markdown("Weights normalized within each wave.")
    st.dataframe(show, use_container_width=True)
with cB:
    st.subheader("Wave Composition")
    st.caption("Full holdings list (normalized, deduped).")
    st.dataframe(hold, use_container_width=True, height=420)

# Market Intel
st.subheader("Market Intel")
if yf is None:
    st.warning("yfinance not available — Market Intel won’t update.")
else:
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=7)
        intel_tks = [t[0] for t in MARKET_INTEL]
        intel_px = yf_download_prices(intel_tks, start, end)
        rows = []
        if not intel_px.empty:
            for tk, label in MARKET_INTEL:
                if tk.upper() in intel_px.columns:
                    s = intel_px[tk.upper()].dropna()
                elif tk in intel_px.columns:
                    s = intel_px[tk].dropna()
                else:
                    s = pd.Series(dtype=float)
                if len(s) >= 2:
                    d1 = float(s.iloc[-1] / s.iloc[-2] - 1.0)
                    w1 = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 1 else np.nan
                    rows.append({"symbol": tk, "label": label, "last": float(s.iloc[-1]), "1d": d1, "7d": w1})
                else:
                    rows.append({"symbol": tk, "label": label, "last": np.nan, "1d": np.nan, "7d": np.nan})
        intel_df = pd.DataFrame(rows)
        intel_df["1d"] = intel_df["1d"].apply(lambda x: fmt_pct(x))
        intel_df["7d"] = intel_df["7d"].apply(lambda x: fmt_pct(x))
        intel_df["last"] = intel_df["last"].apply(lambda x: fmt_num(x))
        st.dataframe(intel_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Market Intel unavailable: {e}")

# Deeper analytics (optional)
if not scan_mode:
    st.divider()
    st.subheader("Risk Lab")

    if hist is None or hist.empty or "nav" not in hist.columns:
        st.warning("No history NAV available for this wave+mode yet. Use Backfill tool or ensure logs/performance exists.")
    else:
        nav = hist["nav"].dropna()
        dd = compute_drawdown(nav)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Max Drawdown", fmt_pct(float(dd.min())) if len(dd) else "—")
        with c2:
            vol30 = float(hist["ret"].rolling(30).std().iloc[-1] * math.sqrt(252)) if "ret" in hist.columns and len(hist) > 40 else np.nan
            st.metric("Rolling Vol (30d ann.)", fmt_pct(vol30) if not math.isnan(vol30) else "—")
        with c3:
            if not bm_hist.empty and "ret" in bm_hist.columns and "ret" in hist.columns:
                aligned = pd.concat([hist["ret"], bm_hist["ret"]], axis=1).dropna()
                aligned.columns = ["wave", "bm"]
                beta = np.nan
                if len(aligned) > 50:
                    cov = np.cov(aligned["wave"], aligned["bm"])[0, 1]
                    var = np.var(aligned["bm"])
                    beta = float(cov / var) if var > 0 else np.nan
                st.metric("Beta (vs benchmark)", fmt_num(beta) if beta == beta else "—")
            else:
                st.metric("Beta (vs benchmark)", "—")

        st.line_chart(pd.DataFrame({"NAV": nav}), height=240)
        st.line_chart(pd.DataFrame({"Drawdown": dd}), height=200)

        # Rolling Alpha (30d) vs benchmark
        if not bm_hist.empty and "ret" in bm_hist.columns and "ret" in hist.columns:
            aligned = pd.concat([hist["ret"], bm_hist["ret"]], axis=1).dropna()
            aligned.columns = ["wave", "bm"]
            roll_alpha = (aligned["wave"] - aligned["bm"]).rolling(DEFAULT_ROLL_WINDOW).sum()
            st.subheader(f"Rolling Alpha ({DEFAULT_ROLL_WINDOW}d sum)")
            st.line_chart(roll_alpha, height=200)

    st.divider()
    st.subheader("Correlation Matrix (All Waves)")

    # Build a returns matrix from available logs (fast, best effort)
    # For each wave, try to read Standard history; fallback to live build if needed.
    try:
        mat = {}
        for wv in waves:
            dfw = read_perf_csv(wv, "Standard")
            if dfw.empty and yf is not None:
                ww = get_wave_holdings(weights_df, wv)
                dfw = build_nav_from_weights(ww, min(int(lookback_days), 520))
            if dfw is not None and not dfw.empty and "ret" in dfw.columns:
                s = dfw["ret"].copy()
                s.name = wv
                mat[wv] = s
        if len(mat) >= 3:
            R = pd.concat(mat.values(), axis=1).dropna()
            corr = R.corr()
            st.dataframe(corr, use_container_width=True, height=360)
        else:
            st.info("Not enough wave histories available yet to compute correlation matrix.")
    except Exception as e:
        st.warning(f"Correlation matrix unavailable: {e}")

# Footer
st.caption("Tip: If the Console loads but a wave shows no history, run Backfill Missing History for that wave+mode.")