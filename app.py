# app.py
#
# WAVES Intelligence™ - Institutional Console
# Per-Wave Benchmarks, VIX-gated exposure + SmartSafe™,
# Excess return windows, Realized Beta, Momentum / Leadership Engine,
# Beta Drift Alerts, Mode-Aware Suggested Allocation,
# Position-Level Allocation, 1-Year Alpha Contribution,
# and "Mini Bloomberg" UI.
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

# Excess return windows in trading days (approx)
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
BETA_LOOKBACK_DAYS = 252  # ~1Y

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

# Mode-aware tilt caps: (max_overweight, min_underweight)
MODE_TILT_LIMITS = {
    "Standard": (1.15, 0.85),
    "Alpha-Minus-Beta": (1.10, 0.90),
    "Alpha Minus Beta": (1.10, 0.90),  # just in case of slight naming
    "Private Logic": (1.30, 0.70),
    "Private Logic™": (1.30, 0.70),
}

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
    Compute total excess return over each window for a single Wave vs its benchmark.
    Returns dict window_label -> excess_return (Wave minus Benchmark), not annualized.
    """
    results = {}
    if wave_ret.empty or bench_ret.empty:
        return results

    for label, days in windows.items():
        # Need some history to say anything meaningful
        if len(wave_ret) < 3:
            continue

        sub_w = wave_ret.iloc[-days:]
        sub_b = bench_ret.iloc[-days:]
        sub_w, sub_b = align_wave_and_benchmark(sub_w, sub_b)
        if sub_w.empty:
            continue

        # Total period return for Wave and Benchmark over this window
        wave_total = (1.0 + sub_w).prod() - 1.0
        bench_total = (1.0 + sub_b).prod() - 1.0

        # Excess return (Wave − Benchmark) for that period
        period_excess = wave_total - bench_total
        results[label] = period_excess

    return results


def compute_beta_series(wave_ret, bench_ret, lookback_days):
    """
    Compute realized beta for a single Wave vs its benchmark.
    """
    if wave_ret.empty or bench_ret.empty:
        return np.nan

    sub_w = wave_ret.iloc[-lookback_days:]
    sub_b = bench_ret.iloc[-lookback_days:]
    sub_w, sub_b = align_wave_and_benchmark(sub_w, sub_b)

    if len(sub_w) < 30:
        return np.nan

    cov = np.cov(sub_w, sub_b)[0, 1]
    var_b = np.var(sub_b)
    if var_b <= 0:
        return np.nan
    return cov / var_b


def compute_momentum_series(wave_ret, bench_ret, windows, weights):
    """
    Momentum score: weighted blend of total excess returns over 30D / 60D / 6M.
    """
    if wave_ret.empty or bench_ret.empty:
        return np.nan

    total_score = 0.0
    total_weight = 0.0

    for label, days in windows.items():
        w = weights.get(label, 0.0)
        if w <= 0:
            continue

        sub_w = wave_ret.iloc[-days:]
        sub_b = bench_ret.iloc[-days:]
        sub_w, sub_b = align_wave_and_benchmark(sub_w, sub_b)
        if sub_w.empty:
            continue

        # total excess return over the window
        excess = (1 + sub_w).prod() / (1 + sub_b).prod() - 1.0
        total_score += w * excess
        total_weight += w

    if total_weight <= 0:
        return np.nan

    return total_score / total_weight


def vix_to_exposure(vix_value):
    """
    Map VIX level to equity & SmartSafe exposures.
    """
    if vix_value is None or np.isnan(vix_value):
        return 1.0, 0.0

    for max_vix, eq, ss in VIX_LADDER:
        if vix_value <= max_vix:
            return eq, ss

    return 1.0, 0.0


def style_wave_table(df):
    """
    Formatting for wave summary table.
    """
    if df.empty:
        return df

    # Columns that should display as percentages
    fmt_pct_cols = [
        c
        for c in df.columns
        if any(c2 in c for c2 in ["Alpha_", "Momentum_Score", "Equity_Alloc", "SmartSafe_Alloc"])
    ]

    # Columns that represent beta values (not drift)
    fmt_beta_cols = [c for c in df.columns if "Beta" in c and "Drift" not in c]

    def fmt(x, kind):
        if pd.isna(x):
            return ""
        if kind == "pct":
            return f"{x*100:,.2f}%"
        if kind == "beta":
            return f"{x:,.2f}"
        return x

    df_fmt = df.copy()

    for c in fmt_pct_cols:
        df_fmt[c] = df_fmt[c].apply(lambda x: fmt(x, "pct"))

    for c in fmt_beta_cols:
        df_fmt[c] = df_fmt[c].apply(lambda x: fmt(x, "beta"))

    return df_fmt


def compute_suggested_allocations(summary_df, equity_pct, smartsafe_pct):
    """
    Build Suggested Allocation table:
    - Start from equal weight per Wave.
    - Apply Tilt_Multiplier.
    - Normalize to 100% of equity slice.
    - Return DataFrame with equity weights (%) per Wave and SmartSafe allocation.
    """
    if summary_df.empty or "Tilt_Multiplier" not in summary_df.columns:
        return pd.DataFrame()

    n = len(summary_df)
    if n == 0:
        return pd.DataFrame()

    base_weight = 1.0 / n
    tilted = base_weight * summary_df["Tilt_Multiplier"].astype(float)
    total = tilted.sum()

    if total <= 0:
        tilted_norm = pd.Series([1.0 / n] * n, index=summary_df.index)
    else:
        tilted_norm = tilted / total

    # Convert to % of overall portfolio
    equity_slice = equity_pct  # 0–1
    equity_weights_pct = tilted_norm * equity_slice * 100.0
    smartsafe_weight_pct = smartsafe_pct * 100.0

    alloc_df = pd.DataFrame(
        {
            "Wave": summary_df["Wave"],
            "Mode": summary_df["Mode"],
            "Momentum_Tier": summary_df.get("Momentum_Tier", ""),
            "Tilt_Label": summary_df.get("Tilt_Label", ""),
            "Tilt_Multiplier": summary_df["Tilt_Multiplier"],
            "Equity_Weight_%": equity_weights_pct,
        }
    )

    alloc_df = alloc_df.sort_values("Equity_Weight_%", ascending=False).reset_index(drop=True)

    # Add a footer row for SmartSafe
    smartsafe_row = pd.DataFrame(
        {
            "Wave": [SMARTSAFE_NAME],
            "Mode": [""],
            "Momentum_Tier": [""],
            "Tilt_Label": [""],
            "Tilt_Multiplier": [np.nan],
            "Equity_Weight_%": [smartsafe_weight_pct],
        }
    )

    alloc_df = pd.concat([alloc_df, smartsafe_row], ignore_index=True)
    return alloc_df


def compute_position_allocations(alloc_df, weights_df):
    """
    Expand Wave-level suggested allocations down to individual holdings.
    """
    if alloc_df is None or alloc_df.empty:
        return pd.DataFrame()

    waves_only = alloc_df[alloc_df["Wave"] != SMARTSAFE_NAME].copy()
    if waves_only.empty:
        return pd.DataFrame()

    waves_only["Equity_Weight_%"] = pd.to_numeric(
        waves_only["Equity_Weight_%"], errors="coerce"
    ).fillna(0.0)
    waves_only["Equity_Weight_Fraction"] = waves_only["Equity_Weight_%"] / 100.0

    positions = []

    for _, row in waves_only.iterrows():
        wave_name = row["Wave"]
        wave_frac = float(row["Equity_Weight_Fraction"])

        wdf = weights_df[weights_df["Wave"] == wave_name]
        if wdf.empty:
            continue

        for _, h in wdf.iterrows():
            ticker = h["Ticker"]
            intra = float(h["Weight"])         # 0–1 inside wave
            pos_frac = wave_frac * intra       # fraction of total portfolio

            positions.append(
                {
                    "Wave": wave_name,
                    "Ticker": ticker,
                    "IntraWave_Weight": intra,
                    "Wave_Equity_Weight_%": row["Equity_Weight_%"],
                    "Position_Equity_Weight_%": pos_frac * 100.0,
                }
            )

    pos_df = pd.DataFrame(positions)
    if pos_df.empty:
        return pos_df

    pos_df = pos_df.sort_values(
        "Position_Equity_Weight_%", ascending=False
    ).reset_index(drop=True)
    return pos_df


def compute_1yr_contribution(summary_df, alloc_df):
    """
    For each Wave:
      1Y Excess Return (Alpha_1Y)
      Suggested Equity Weight (%)
      Contribution = (Equity_Weight_%/100) * Alpha_1Y
    SmartSafe™ is ignored.
    """
    if summary_df.empty or alloc_df.empty:
        return pd.DataFrame()

    waves_only = alloc_df[alloc_df["Wave"] != SMARTSAFE_NAME].copy()
    if waves_only.empty:
        return pd.DataFrame()

    rows = []

    for _, r in waves_only.iterrows():
        wave = r["Wave"]
        weight_pct = float(r["Equity_Weight_%"])
        weight_frac = weight_pct / 100.0

        alpha_row = summary_df[summary_df["Wave"] == wave]
        if alpha_row.empty:
            continue

        alpha_1y = alpha_row["Alpha_1Y"].values[0]

        if pd.isna(alpha_1y):
            contrib = np.nan
        else:
            contrib = weight_frac * alpha_1y

        rows.append(
            {
                "Wave": wave,
                "Equity_Weight_%": weight_pct,
                "Alpha_1Y": alpha_1y,
                "Contribution": contrib,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("Contribution", ascending=False).reset_index(drop=True)
    return df

# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(
        page_title="WAVES Institutional Console",
        layout="wide",
    )

    # Header
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("### 🌊")
    with col_title:
        st.markdown(
            """
            # WAVES Institutional Console  
            **AI-Driven Multi-Wave Engine — VIX-Gated Exposure + SmartSafe™**
            """
        )

    st.markdown(
        """
        <small>WAVES Intelligence™ • Adaptive Portfolio Waves (AIWs/APWs) • Vector OS™</small>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.header("Console Controls")

    lookback_days = st.sidebar.select_slider(
        "History Lookback (for charts & metrics)",
        options=[252, 365, 730],
        value=365,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Mode Legend**")
    st.sidebar.markdown(
        """
        - **Standard** — Baseline Wave behavior  
        - **Alpha-Minus-Beta** — Target β ~ 0.80–0.90, higher downside discipline  
        - **Private Logic™** — Higher turnover, more aggressive alpha capture  
        """
    )

    # Load config & weights
    cfg = load_wave_config()
    weights = load_wave_weights()

    if cfg is None or weights is None:
        st.stop()

    waves = sorted(cfg["Wave"].unique())
    st.sidebar.write(f"Loaded Waves: **{len(waves)}**")

    # Tickers
    holding_tickers = sorted(weights["Ticker"].unique())
    benchmark_tickers = sorted(cfg["Benchmark"].unique())
    all_tickers = list(set(holding_tickers + benchmark_tickers + [DEFAULT_BENCHMARK, "SPY"]))

    price_start = dt.date.today() - dt.timedelta(days=lookback_days + 30)
    prices = download_price_history(tickers=all_tickers, start_date=price_start)

    if prices.empty:
        st.error("Could not download price history. Check tickers or network.")
        st.stop()

    # Wave returns
    wave_returns = compute_portfolio_returns(weights, prices)

    if wave_returns.empty:
        st.error("Wave returns could not be computed. Check wave_weights.csv.")
        st.stop()

    # VIX + SPY for regime strip & chart
    vs_hist = get_vix_and_spy_history(lookback_days=lookback_days)
    latest_vix = None
    if "^VIX" in vs_hist.columns:
        latest_vix = float(vs_hist["^VIX"].dropna().iloc[-1])

    equity_pct, smartsafe_pct = vix_to_exposure(latest_vix)

    # ===================== TOP METRICS STRIP =====================

    st.markdown("## Market Regime & SmartSafe™ Overlay")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "VIX (Real-Time Regime)",
            f"{latest_vix:,.2f}" if latest_vix is not None else "N/A",
        )

    with col2:
        st.metric(
            "Target Equity Exposure",
            f"{equity_pct*100:,.0f}%",
        )

    with col3:
        st.metric(
            f"{SMARTSAFE_NAME} Allocation",
            f"{smartsafe_pct*100:,.0f}%",
        )

    with col4:
        st.metric("Number of Waves", len(waves))

    # VIX + SPY chart
    st.markdown("### VIX vs S&P (SPY)")

    if "^VIX" in vs_hist.columns and "SPY" in vs_hist.columns:
        chart_df = vs_hist[["^VIX", "SPY"]].dropna()
        chart_df = chart_df.rename(columns={"^VIX": "VIX", "SPY": "SPY"})
        chart_df = chart_df.reset_index().rename(columns={"Date": "Date"})

        fig = px.line(
            chart_df,
            x="Date",
            y=["VIX", "SPY"],
            title="VIX & SPY — Regime Context",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("VIX or SPY history unavailable for chart.")

    # ===================== WAVE METRICS (PER-WAVE BENCHMARKS) =====================

    st.markdown("## Wave-Level Summary (Per-Wave Benchmarks, Excess Returns, Beta, Momentum, Tilt)")

    basic_rows = []

    for wave in waves:
        cfg_row = cfg[cfg["Wave"] == wave].iloc[0]
        bench_ticker = cfg_row["Benchmark"] if cfg_row["Benchmark"] in prices.columns else DEFAULT_BENCHMARK

        wave_ret = wave_returns[wave].dropna()
        bench_px = prices[bench_ticker].dropna()
        bench_ret = bench_px.pct_change().fillna(0.0)

        alpha_dict = compute_alpha_windows_series(wave_ret, bench_ret, ALPHA_WINDOWS)
        beta_val = compute_beta_series(wave_ret, bench_ret, BETA_LOOKBACK_DAYS)
        momentum_score = compute_momentum_series(wave_ret, bench_ret, MOMENTUM_WINDOWS, MOMENTUM_WEIGHTS)

        row = {
            "Wave": wave,
            "Mode": cfg_row["Mode"],
            "Benchmark": bench_ticker,
            "Beta_Target": cfg_row["Beta_Target"],
            "Realized_Beta": beta_val,
            "Momentum_Score": momentum_score,
        }

        for label in ALPHA_WINDOWS.keys():
            col_name = f"Alpha_{label}"
            row[col_name] = alpha_dict.get(label, np.nan)

        basic_rows.append(row)

    summary_df = pd.DataFrame(basic_rows)

    # Momentum tiers + beta drift + tilt (mode-aware)
    if not summary_df.empty:
        summary_df["Momentum_Rank"] = summary_df["Momentum_Score"].rank(
            ascending=False, method="min"
        )

        n = len(summary_df)
        if n >= 3:
            summary_df["Momentum_Tier"] = pd.qcut(
                summary_df["Momentum_Rank"],
                q=3,
                labels=["Leader", "Neutral", "Laggard"],
            )
        elif n == 2:
            summary_df["Momentum_Tier"] = np.where(
                summary_df["Momentum_Rank"] == 1, "Leader", "Laggard"
            )
        elif n == 1:
            summary_df["Momentum_Tier"] = "Leader"
        else:
            summary_df["Momentum_Tier"] = ""

        beta_drift_list = []
        beta_flag_list = []
        tilt_label_list = []
        tilt_mult_list = []

        for _, r in summary_df.iterrows():
            beta_target = r["Beta_Target"]
            beta_val = r["Realized_Beta"]
            tier = r["Momentum_Tier"]
            mode = r["Mode"]

            if pd.isna(beta_val):
                beta_drift = np.nan
                drift_flag = False
            else:
                beta_drift = beta_val - beta_target
                drift_flag = abs(beta_drift) > BETA_DRIFT_THRESHOLD

            # Base tilt by momentum tier
            tilt_label = "Neutral"
            tilt_mult = 1.0
            if tier == "Leader":
                tilt_label = "Overweight"
                tilt_mult = 1.2
            elif tier == "Laggard":
                tilt_label = "Underweight"
                tilt_mult = 0.8

            # Mode-aware caps
            max_over, min_under = MODE_TILT_LIMITS.get(mode, (1.15, 0.85))
            if tilt_mult > 1.0:
                tilt_mult = min(tilt_mult, max_over)
            elif tilt_mult < 1.0:
                tilt_mult = max(tilt_mult, min_under)

            # If beta too high vs target, trim slightly
            if drift_flag and not pd.isna(beta_val) and beta_val > beta_target:
                tilt_label = (
                    tilt_label + " (Trim β)" if tilt_label != "Neutral" else "Neutral (Trim β)"
                )
                tilt_mult *= 0.9

            beta_drift_list.append(beta_drift)
            beta_flag_list.append(drift_flag)
            tilt_label_list.append(tilt_label)
            tilt_mult_list.append(tilt_mult)

        summary_df["Beta_Drift"] = beta_drift_list
        summary_df["Beta_Drift_Flag"] = beta_flag_list
        summary_df["Tilt_Label"] = tilt_label_list
        summary_df["Tilt_Multiplier"] = tilt_mult_list

        summary_df["Equity_Alloc"] = equity_pct
        summary_df["SmartSafe_Alloc"] = smartsafe_pct

    summary_df = summary_df.sort_values("Momentum_Score", ascending=False)

    st.dataframe(
        style_wave_table(summary_df),
        use_container_width=True,
    )

    # ===================== TOP HOLDINGS VIEW =====================

    st.markdown("## Top Holdings by Wave")

    selected_wave = st.selectbox(
        "Select Wave",
        waves,
        index=0 if waves else None,
    )

    if selected_wave:
        wdf = (
            weights[weights["Wave"] == selected_wave]
            .sort_values("Weight", ascending=False)
            .head(10)
            .copy()
        )

        wdf["Ticker_Link"] = wdf["Ticker"].apply(
            lambda t: f"[{t}]({google_finance_link(t)})"
        )
        wdf["Weight_%"] = wdf["Weight"] * 100.0

        wdf_display = wdf[["Ticker_Link", "Weight_%"]]
        wdf_display = wdf_display.rename(
            columns={"Ticker_Link": "Ticker", "Weight_%": "Weight (%)"}
        )

        st.markdown(
            f"### {selected_wave} — Top 10 Holdings (click ticker for Google Finance)"
        )
        st.markdown(wdf_display.to_markdown(index=False), unsafe_allow_html=True)

    # ===================== MOMENTUM / LEADERSHIP DETAIL =====================

    st.markdown("## Momentum / Leadership Engine — Detail")

    if not summary_df.empty:
        st.write(
            """
            **Momentum_Score** is a weighted blend of 30D / 60D / 6M excess returns
            vs each Wave's own benchmark.  
            **Momentum_Tier** buckets Waves into Leaders / Neutral / Laggards for capital rotation.
            **Mode-aware tilt caps** keep Standard waves calmer than Private Logic™ waves.
            """
        )
        st.dataframe(
            summary_df[
                [
                    "Wave",
                    "Mode",
                    "Benchmark",
                    "Momentum_Score",
                    "Momentum_Rank",
                    "Momentum_Tier",
                    "Tilt_Label",
                    "Tilt_Multiplier",
                ]
            ].set_index("Wave"),
            use_container_width=True,
        )
    else:
        st.info("Momentum metrics not available yet (insufficient history).")

    # ===================== RISK ALERTS =====================

    st.markdown("## Risk Alerts — Beta Drift Discipline")

    alert_df = summary_df[summary_df.get("Beta_Drift_Flag", False) == True]

    if not alert_df.empty:
        st.warning(
            "These Waves have realized beta outside the allowed drift band "
            f"(>|{BETA_DRIFT_THRESHOLD:.2f}| vs target)."
        )
        st.dataframe(
            alert_df[
                [
                    "Wave",
                    "Mode",
                    "Benchmark",
                    "Beta_Target",
                    "Realized_Beta",
                    "Beta_Drift",
                    "Momentum_Tier",
                    "Tilt_Label",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.success("No Waves currently breaching the beta-drift threshold.")

    # ===================== SUGGESTED ALLOCATION (WAVES + SMARTSAFE) =====================

    st.markdown("## Suggested Allocation — Waves + SmartSafe™")

    alloc_df = compute_suggested_allocations(summary_df, equity_pct, smartsafe_pct)

    positions_df = pd.DataFrame()

    if not alloc_df.empty:
        st.write(
            """
            Suggested allocation starts from **equal weight** across Waves, then **applies
            mode-aware leadership tilts (Tilt_Multiplier)** and normalizes to your current
            **Equity / SmartSafe™ overlay** from the VIX regime.
            """
        )
        alloc_df_display = alloc_df.copy()
        alloc_df_display["Equity_Weight_%"] = alloc_df_display["Equity_Weight_%"].apply(
            lambda x: "" if pd.isna(x) else f"{x:,.1f}%"
        )
        st.dataframe(alloc_df_display, use_container_width=True)

        csv_bytes = alloc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Wave-Level Allocation CSV",
            data=csv_bytes,
            file_name="waves_suggested_allocation.csv",
            mime="text/csv",
        )

        # ===================== POSITION-LEVEL ALLOCATION =====================

        st.markdown("## Position-Level Allocation (Across All Waves)")

        positions_df = compute_position_allocations(alloc_df, weights)

        if not positions_df.empty:
            pos_display = positions_df.copy()
            pos_display["IntraWave_Weight"] = pos_display["IntraWave_Weight"].apply(
                lambda x: f"{x*100:,.1f}%"
            )
            pos_display["Wave_Equity_Weight_%"] = pos_display["Wave_Equity_Weight_%"].apply(
                lambda x: f"{x:,.1f}%"
            )
            pos_display["Position_Equity_Weight_%"] = pos_display["Position_Equity_Weight_%"].apply(
                lambda x: f"{x:,.2f}%"
            )

            st.dataframe(pos_display, use_container_width=True)

            pos_csv_bytes = positions_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Position-Level Allocation CSV",
                data=pos_csv_bytes,
                file_name="waves_position_allocation.csv",
                mime="text/csv",
            )
        else:
            st.info("No position-level allocations could be computed.")

        # ===================== 1-YEAR ALPHA CONTRIBUTION =====================

        st.markdown("## 1-Year Contribution to Portfolio Alpha")

        contrib_df = compute_1yr_contribution(summary_df, alloc_df)

        if not contrib_df.empty:
            df_disp = contrib_df.copy()
            df_disp["Equity_Weight_%"] = df_disp["Equity_Weight_%"].apply(
                lambda x: f"{x:,.1f}%"
            )
            df_disp["Alpha_1Y"] = df_disp["Alpha_1Y"].apply(
                lambda x: f"{x*100:,.2f}%" if not pd.isna(x) else ""
            )
            df_disp["Contribution"] = df_disp["Contribution"].apply(
                lambda x: f"{x*100:,.2f}%" if not pd.isna(x) else ""
            )

            st.dataframe(df_disp, use_container_width=True)

            total_alpha = contrib_df["Contribution"].sum() * 100
            st.metric("Total Portfolio Alpha (1-Year)", f"{total_alpha:,.2f}%")
        else:
            st.info("Not enough data to compute 1-year alpha contribution.")
    else:
        st.info("Suggested allocation unavailable (missing tilt data).")

    st.markdown("---")
    st.caption("WAVES Intelligence™ • Adaptive Portfolio Waves • SmartSafe™ • Vector OS™")


if __name__ == "__main__":
    main()
