# waves_engine.py — WAVES Intelligence™ DEMO ENGINE (STUB)
# Purpose:
#   Provide deterministic synthetic history + holdings so the console "meat" lights up
#   (Risk Lab, Correlation, WaveScore, Attribution, Insight Layer, Benchmark Truth).
#
# How it works:
#   - Reads wave list + holdings from wave_weights.csv when available
#   - Generates NAV series per (wave, mode) using deterministic seeded random walk
#   - Generates benchmark NAV similarly (slower / different seed)
#   - Exposes the minimal API the console expects:
#       get_all_waves()
#       get_wave_holdings(wave_name)
#       get_benchmark_mix_table()
#       compute_history_nav(wave_name, mode="Standard", days=365)
#
# IMPORTANT:
#   This is a demo-only stub. Rename your real engine to waves_engine_real.py first.

from __future__ import annotations

import os
import math
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


# -------------------------------
# Utilities
# -------------------------------
def _sha_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # deterministic 32-bit seed

def _business_days(end_dt: Optional[pd.Timestamp], days: int) -> pd.DatetimeIndex:
    if end_dt is None:
        end_dt = pd.Timestamp.utcnow().normalize()
    # buffer to ensure enough business days
    start = (end_dt - pd.Timedelta(days=int(days * 1.8) + 30)).normalize()
    idx = pd.date_range(start=start, end=end_dt, freq="B")
    if len(idx) > days:
        idx = idx[-days:]
    return idx

def _read_wave_weights(path: str = "wave_weights.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["wave", "ticker", "weight"])

    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    if not {"wave", "ticker", "weight"}.issubset(set(lower.keys())):
        # try common alternates
        wave_c = lower.get("wave_name") or lower.get("wavename") or lower.get("wave")
        tick_c = lower.get("symbol") or lower.get("ticker")
        wgt_c = lower.get("pct") or lower.get("weight")
        if not (wave_c and tick_c and wgt_c):
            return pd.DataFrame(columns=["wave", "ticker", "weight"])
        df = df.rename(columns={wave_c: "wave", tick_c: "ticker", wgt_c: "weight"})
    else:
        df = df.rename(columns={lower["wave"]: "wave", lower["ticker"]: "ticker", lower["weight"]: "weight"})

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    df = df[(df["wave"] != "") & (df["ticker"] != "")]
    df = df.groupby(["wave", "ticker"], as_index=False)["weight"].sum()

    # normalize per wave
    out_rows = []
    for w, g in df.groupby("wave"):
        tot = float(g["weight"].sum())
        gg = g.copy()
        if tot > 0:
            gg["weight"] = gg["weight"] / tot
        out_rows.append(gg)
    if out_rows:
        df = pd.concat(out_rows, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["wave", "ticker", "weight"])

    return df


# -------------------------------
# Public API (expected by console)
# -------------------------------
def get_all_waves() -> List[str]:
    df = _read_wave_weights()
    if df.empty:
        # fallback list so the UI still works
        return [
            "S&P 500 Wave",
            "AI & Cloud MegaCap Wave",
            "Quantum Computing Wave",
            "Crypto Wave",
            "Future Power & Energy Wave",
        ]
    waves = sorted(list(set(df["wave"].astype(str).str.strip().tolist())))
    waves = [w for w in waves if w and w.lower() != "nan"]
    return waves


def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    df = _read_wave_weights()
    if df.empty:
        # safe fallback holdings
        fallback = [
            ("NVDA", 0.18),
            ("MSFT", 0.16),
            ("AAPL", 0.14),
            ("AMZN", 0.12),
            ("META", 0.10),
            ("GOOGL", 0.10),
            ("TSLA", 0.08),
            ("BRK.B", 0.06),
            ("AVGO", 0.04),
            ("LLY", 0.02),
        ]
        out = pd.DataFrame(fallback, columns=["Ticker", "Weight"])
        out["Name"] = ""
        return out[["Ticker", "Name", "Weight"]]

    w = str(wave_name).strip()
    g = df[df["wave"].astype(str).str.strip().str.lower() == w.lower()].copy()
    if g.empty:
        # if wave not found, just return top 10 overall
        g = df.copy()

    g = g.groupby("ticker", as_index=False)["weight"].sum()
    g = g.sort_values("weight", ascending=False).head(25).copy()

    tot = float(g["weight"].sum())
    if tot > 0:
        g["weight"] = g["weight"] / tot

    out = pd.DataFrame({
        "Ticker": g["ticker"].astype(str).str.upper().str.strip(),
        "Name": "",
        "Weight": pd.to_numeric(g["weight"], errors="coerce").fillna(0.0),
    })
    return out[["Ticker", "Name", "Weight"]]


def get_benchmark_mix_table() -> pd.DataFrame:
    """
    Simple benchmark mix table for Benchmark Truth panel.
    You can customize later. For demos, this is enough.
    """
    waves = get_all_waves()
    # universal blended benchmark: SPY/QQQ/IWM/TLT/GLD
    mix = [("SPY", 0.45), ("QQQ", 0.25), ("IWM", 0.10), ("TLT", 0.12), ("GLD", 0.08)]
    rows = []
    for w in waves:
        for t, wt in mix:
            rows.append({"Wave": w, "Ticker": t, "Name": "", "Weight": wt})
    return pd.DataFrame(rows)


def compute_history_nav(wave_name: str, mode: str = "Standard", days: int = 365) -> pd.DataFrame:
    """
    Deterministic synthetic NAV history. Same wave+mode => same series each run.
    Output columns expected by console:
      - wave_nav, bm_nav, wave_ret, bm_ret
    """
    days = int(max(60, min(days, 1500)))

    wave = str(wave_name).strip()
    mode = str(mode).strip()

    idx = _business_days(pd.Timestamp.utcnow().normalize(), days)

    # Seeded RNGs (stable)
    rng_w = np.random.default_rng(_sha_seed(f"WAVE::{wave}::{mode}"))
    rng_b = np.random.default_rng(_sha_seed(f"BM::{wave}::{mode}"))

    # Mode affects drift/vol so modes look different (demo-friendly)
    mode_l = mode.lower()
    if "alpha" in mode_l or "amb" in mode_l:
        drift = 0.00055
        vol = 0.0130
    elif "private" in mode_l or "ple" in mode_l:
        drift = 0.00070
        vol = 0.0160
    else:
        drift = 0.00045
        vol = 0.0110

    # Benchmark: smoother, slightly lower drift
    bm_drift = max(0.00025, drift - 0.00015)
    bm_vol = max(0.0085, vol - 0.0035)

    # Generate daily log-returns
    w_ret = rng_w.normal(loc=drift, scale=vol, size=len(idx))
    b_ret = rng_b.normal(loc=bm_drift, scale=bm_vol, size=len(idx))

    # Add gentle regime waves so charts look realistic (still deterministic)
    t = np.linspace(0, 6 * math.pi, len(idx))
    w_ret = w_ret + 0.0015 * np.sin(t)
    b_ret = b_ret + 0.0010 * np.sin(t + 0.7)

    # Convert to NAV (start at 100)
    wave_nav = 100.0 * np.exp(np.cumsum(w_ret))
    bm_nav = 100.0 * np.exp(np.cumsum(b_ret))

    df = pd.DataFrame(
        {
            "wave_nav": wave_nav,
            "bm_nav": bm_nav,
        },
        index=pd.to_datetime(idx),
    )

    df["wave_ret"] = pd.Series(df["wave_nav"]).pct_change().values
    df["bm_ret"] = pd.Series(df["bm_nav"]).pct_change().values

    return df