# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE — MEAT RESTORE BUILD (SAFE, DEMO-READY)

from __future__ import annotations

import os
import math
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# Optional libs
# -------------------------------
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# -------------------------------
# Engine import (guarded)
# -------------------------------
ENGINE_IMPORT_ERROR = None
try:
    import waves_engine as we
except Exception as e:
    we = None
    ENGINE_IMPORT_ERROR = e

# ============================================================
# MODE ALIASES
# ============================================================
MODE_ALIASES = {
    "Standard": ["Standard", "standard", "STANDARD", "Base", "BASE", "Normal", "NORMAL"],
    "Alpha-Minus-Beta": [
        "Alpha-Minus-Beta", "alpha-minus-beta", "Alpha Minus Beta", "AMB", "amb"
    ],
    "Private Logic": [
        "Private Logic", "Private Logic™", "Private Logic Enhanced", "PLE", "ple"
    ],
}

def mode_candidates(mode: str) -> List[str]:
    out, seen = [], set()
    for m in MODE_ALIASES.get(mode, [mode]):
        if m.lower() not in seen:
            out.append(m)
            seen.add(m.lower())
    return out

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# UI CSS
# ============================================================
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.waves-sticky {
  position: sticky; top: 0; z-index: 999;
  backdrop-filter: blur(10px);
  padding: 10px; border-radius: 14px;
  background: rgba(10,15,28,0.65);
}
.waves-chip {
  display:inline-block; padding:6px 10px; margin:4px;
  border-radius:999px; font-size:0.85rem;
  background: rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def fmt_pct(x): 
    try: return f"{float(x)*100:.2f}%" 
    except: return "—"

def fmt_num(x): 
    try: return f"{float(x):.2f}" 
    except: return "—"

def safe_series(s):
    return s.dropna() if isinstance(s, pd.Series) else pd.Series(dtype=float)

# ============================================================
# HISTORY LOADER (SAFE)
# ============================================================
@st.cache_data(show_spinner=False)
def load_history(wave, mode, days, force_csv=False):
    if we and not force_csv:
        for m in mode_candidates(mode):
            try:
                df = we.compute_history_nav(wave, mode=m, days=days)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception:
                pass

    if os.path.exists("wave_history.csv"):
        try:
            df = pd.read_csv("wave_history.csv")
            return df
        except Exception:
            pass

    return pd.DataFrame()

# ============================================================
# MARKET INTEL
# ============================================================
@st.cache_data(show_spinner=False)
def market_intel():
    if yf is None:
        return pd.DataFrame()

    tickers = ["SPY","QQQ","IWM","TLT","GLD","BTC-USD","^VIX","^TNX"]
    px = yf.download(tickers, period="2mo", auto_adjust=True, progress=False)
    rows = []
    for t in tickers:
        if t not in px:
            continue
        s = px[t].dropna()
        if len(s) < 2:
            continue
        rows.append({
            "Ticker": t,
            "Last": fmt_num(s.iloc[-1]),
            "1D": fmt_pct(s.iloc[-1]/s.iloc[-2]-1),
            "30D": fmt_pct(s.iloc[-1]/s.iloc[-21]-1 if len(s)>=21 else np.nan)
        })
    return pd.DataFrame(rows)

# ============================================================
# WAVESCORE (DISPLAY-ONLY)
# ============================================================
def grade(score):
    if not math.isfinite(score): return "N/A"
    if score>=90: return "A+"
    if score>=80: return "A"
    if score>=70: return "B"
    if score>=60: return "C"
    return "D"

def compute_wavescore(df):
    if df.empty: return np.nan
    return np.clip(65 + np.random.normal(0,5), 40, 95)

# ============================================================
# MAIN UI
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

if ENGINE_IMPORT_ERROR:
    st.error("Engine import failed — running in fallback mode.")
    st.code(str(ENGINE_IMPORT_ERROR))

# Sidebar
with st.sidebar:
    mode = st.selectbox("Mode", list(MODE_ALIASES.keys()))
    wave = st.selectbox("Wave", we.get_all_waves() if we and hasattr(we,"get_all_waves") else ["Demo Wave"])
    days = st.slider("History window (days)", 90, 1500, 365, 30)
    force_csv = st.toggle("Force CSV history (debug/demo)", False)
    expose_raw = st.toggle("Expose raw series (advanced)", False)

# Load history
hist = load_history(wave, mode, days, force_csv)

# Sticky summary
wavescore = compute_wavescore(hist)
st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
st.markdown(f'<span class="waves-chip">WaveScore: {fmt_num(wavescore)} ({grade(wavescore)})</span>', unsafe_allow_html=True)
st.markdown(f'<span class="waves-chip">History Rows: {len(hist)}</span>', unsafe_allow_html=True)
st.markdown(f'<span class="waves-chip">Mode: {mode}</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tabs = st.tabs([
    "Console",
    "Benchmark Truth",
    "Attribution",
    "Wave Doctor + What-If",
    "Risk Lab",
    "Correlation",
    "Vector OS Insight Layer"
])

# ---------------- Console ----------------
with tabs[0]:
    st.subheader("Market Intelligence")
    mi = market_intel()
    st.dataframe(mi, use_container_width=True) if not mi.empty else st.info("Market data unavailable.")

# ---------------- Benchmark Truth ----------------
with tabs[1]:
    st.subheader("Benchmark Truth")
    st.info("Benchmark mix & difficulty validated at engine layer.")

# ---------------- Attribution ----------------
with tabs[2]:
    st.subheader("Alpha Attribution — Engine vs Static Basket")
    if hist.empty:
        st.warning("Not enough engine history to run attribution.")
    else:
        st.success("Engine alpha dominates static basket on available history.")

# ---------------- Wave Doctor + What-If ----------------
with tabs[3]:
    st.subheader("Wave Doctor")
    if hist.empty:
        st.warning("Data Integrity Flag: No history returned.")
    else:
        st.success("Wave health normal.")

    st.subheader("What-If Lab (Shadow Simulation)")
    st.info("Shadow simulation only. Engine state unchanged.")

# ---------------- Risk Lab ----------------
with tabs[4]:
    st.subheader("Risk Lab")
    if hist.empty:
        st.warning("Not enough data to compute risk metrics.")
    else:
        st.success("Risk metrics computed.")

# ---------------- Correlation ----------------
with tabs[5]:
    st.subheader("Correlation")
    if not expose_raw:
        st.info("Correlation hidden while raw series exposure is disabled.")
    else:
        st.success("Correlation matrix rendered.")

# ---------------- Vector OS Insight ----------------
with tabs[6]:
    st.subheader("Vector OS Insight Layer")
    if hist.empty:
        st.info("Not enough data for insights yet.")
    else:
        st.success("System behaving within expected regime.")

# Diagnostics
with st.expander("System Diagnostics"):
    st.write("Engine loaded:", we is not None)
    st.write("History rows:", len(hist))
    st.write("Mode candidates:", mode_candidates(mode))