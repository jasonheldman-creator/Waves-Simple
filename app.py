# app.py
#
# WAVES Intelligence™ - Institutional Console
# Per-Wave Benchmarks, VIX-gated exposure + SmartSafe™,
# Alpha windows, Realized Beta, Momentum / Leadership Engine,
# Beta Drift Alerts, Suggested Allocation, and "Mini Bloomberg" UI.
#
# IMPORTANT:
# - Keep your last fully-working app.py saved separately as app_fallback.py.
# - This file is the next-stage version; if it breaks, revert to the fallback.

import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "."  # Folder where CSVs live (adjust if needed)

WAVE_CONFIG_FILE = os.path.join(DATA_DIR, "wave_config.csv")
WAVE_WEIGHTS_FILE = os.path.join(DATA_DIR, "wave_weights.csv")

# Default benchmark if missing in config
DEFAULT_BENCHMARK = "SPY"

# Alpha windows in trading days (approx)
ALPHA_WINDOWS = {
    "30D": 30,
    "60D": 60,
    "6M": 126,
    "1Y": 252,
}

# Momentum window mix (used by leadership engine)
MOMENTUM_WINDOWS = {
    "30D": 30,
    "60D": 60,
    "6M": 126,
}

# Momentum weightings (must sum to 1)
MOMENTUM_WEIGHTS = {
    "30D": 0.4,
    "60D": 0.3,
    "6M": 0.3,
}

# Realized beta uses last N days
BETA_LOOKBACK_DAYS = 252  # 1Y-ish

# How far beta can drift from target before we flag it
BETA_DRIFT_THRESHOLD = 0.07  # |β_real - β_target| > 0.07 -> alert

# VIX -> equity/SmartSafe ladder (you can tweak)
VIX_LADDER = [
    # (max_vix, equity_pct, smartsafe_pct)
    (18, 1.00, 0.00),
    (24, 0.80, 0.20),
    (30, 0.60, 0.40),
    (40, 0.40, 0.60),
    (999, 0.20, 0.80),
]

SMARTSAFE_NAME = "SmartSafe™"

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_wave_config():
    if not os.path.exists(WAVE_CONFIG_FILE):
        st.error(f"wave_config.csv not found at: {WAVE_CONFIG_FILE}")
        return None

    cfg = pd.read_csv(WAVE_CONFIG_FILE)

    # Expected columns: Wave, Benchmark, Mode, Beta_Target
    if "Wave" not in cfg.columns:
        st.error("wave_config.csv must have a 'Wave' column.")
        return None

    if "Benchmark" not in cfg.columns:
        cfg["Benchmark"] = DEFAULT_BENCHMARK

    if "Mode" not in cfg.columns:
        cfg["Mode"] = "Standard"

    if "Beta_Target" not in cfg.columns:
        cfg["Beta_Target"] = 0.90  # default

    cfg["Wave"] = cfg["Wave"].astype(str)
    return cfg


@st.cache_data(show_spinner=False)
def load_wave_weights():
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        st.error(f"wave_weights.csv not found at: {WAVE_WEIGHTS_FILE}")
        return None

    weights = pd.read_csv(WAVE_WEIGHTS_FILE)

    # Expected minimally: Wave, Ticker, Weight
    missing = [c for c in ["Wave", "Ticker", "Weight"] if c not in weights.columns]
    if missing:
        st.error(
            f"wave_weights.csv is missing required columns: {', '.join(missing)}"
        )
        return None

    weights["Weight"] = weights["Weight"].astype(float)
    weights["Wave"] = weights["Wave"].astype(str)
    weights["Ticker"] = weights["Ticker"].astype(str)

    # Normalize per-wave weights to 1.0
    weights = (
        weights.groupby("Wave")
        .apply(lambda df: df.assign(Weight=df["Weight"] / df["Weight"].sum()))
        .reset_index(drop=True)
    )

    return weights


def google_finance_link(ticker: str) -> str:
    return f"https://www.google.com/finance?q={ticker}"


@st.cache_data(show_spinner=False)
def download_price_history(tickers, start_date, end_date=None):
    if end_date is None:
        end_date = dt.date.today() + dt.timedelta(days=1)

    tickers = sorted(list(set(tickers)))
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # yfinance shape: (date, (field, ticker))
    if isinstance(data.columns, pd.MultiIndex):
        px_close = data["Close"].copy()
    else:
        px_close = data.copy()
        px_close.columns = tickers

    return px_close.dropna(how="all")


def compute_portfolio_returns(weights_df, prices_df):
    """
    Compute daily returns for each Wave based on static weights and price history.
    """
    if prices_df.empty:
        return pd.DataFrame()

    daily_returns = prices_df.pct_change().fillna(0.0)
    wave_returns = {}

    for wave, wdf in weights_df.groupby("Wave"):
        w = wdf.set_index("Ticker")["Weight"]
        common_tickers = [t for t in w.index if t in daily_returns.columns]
        if not common_tickers:
            continue
        w = w.loc[common_tickers]
        r = daily_returns[common_tickers]
        wave_returns[wave] = r.dot(w)

    return pd.DataFrame(wave_returns)


@st.cache_data(show_spinner=False)
def get_vix_and_spy_history(lookback_days=365):
    end_date = dt.date.today() + dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=lookback_days + 10)

    tickers = ["^VIX", "SPY"]
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data.copy()

    close = close.dropna(how="all")
    return close

# ============================================================
# METRIC HELPERS (per Wave, per Benchmark)
# ============================================================

def align_wave_and_benchmark(wave_series, bench_series):
    """Align a Wave's return series with its benchmark."""
    df = pd.concat(
        [wave_series.rename("wave"), bench_series.rename("bench")],
        axis=1,
        join="inner",
    ).dropna()
    return df["wave"], df["bench"]


def compute_alpha_windows_series(wave_ret, bench_ret, windows):
    """
    Compute annualized alpha over given windows for a single Wave vs its benchmark.
    Returns dict window_label -> alpha.
    """
    results = {}
    if wave_ret.empty or bench_ret.empty:
        return results

    for label, days in windows.items():
        if len(wave_ret) < 3:
            continue
        sub_w = wave_ret.iloc[-days:]
        sub_b = bench_ret.iloc[-days:]
        sub_w, sub_b = align_wave_and_benchmark(sub_w, sub_b)
        if sub_w.empty:
            continue

        daily_excess = sub_w - sub_b
        ann_alpha = daily_excess.mean() * 252.0
        results[label] = ann_alpha

    return results


def compute_beta_series(wave_ret, bench_ret, lookback_days
