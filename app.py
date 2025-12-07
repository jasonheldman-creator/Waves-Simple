# app.py
#
# WAVES Intelligence™ - Institutional Console
# VIX-gated exposure + SmartSafe™, Alpha windows, Realized Beta,
# Momentum / Leadership Engine, and "Mini Bloomberg" UI.
#
# NOTE:
# - Keep your existing FALLBACK app.py separately as a safety net.
# - This is the "current" version with VIX / SmartSafe and momentum logic.

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
# UTILITIES
# ============================================================

@st.cache_data(show_spinner=False)
def load_wave_config():
    if not os.path.exists(WAVE_CONFIG_FILE):
        st.error(f"wave_config.csv not found at: {WAVE_CONFIG_FILE}")
        return None

    cfg = pd.read_csv(WAVE_CONFIG_FILE)

    # Expected columns: Wave, Benchmark, Mode, Beta_Target
    # We'll fill missing pieces gracefully.
    if "Wave" not in cfg.columns:
        st.error("wave_config.csv must have a 'Wave' column.")
        return None

    if "Benchmark" not in cfg.columns:
        cfg["Benchmark"] = DEFAULT_BENCHMARK

    if "Mode" not in cfg.columns:
        cfg["Mode"] = "Standard"

    if "Beta_Target" not in cfg.columns:
        cfg["Beta_Target"] = 0.90  # default

    # Ensure Wave is string
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

    # Normalize per-wave weights to 1.0 just in case
    weights["Weight"] = weights["Weight"].astype(float)
    weights["Wave"] = weights["Wave"].astype(str)
    weights["Ticker"] = weights["Ticker"].astype(str)

    weights = (
        weights.groupby("Wave")
        .apply(lambda df: df.assign(Weight=df["Weight"] / df["Weight"].sum()))
        .reset_index(drop=True)
    )

    return weights


def google_finance_link(ticker: str) -> str:
    # Use query URL so we don't have to guess the exchange.
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
        # Single ticker case
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
        # Filter prices to tickers present in this wave
        common_tickers = [t for t in w.index if t in daily_returns.columns]
        if not common_tickers:
            continue
        w = w.loc[common_tickers]
        r = daily_returns[common_tickers]
        wave_returns[wave] = r.dot(w)

    wave_ret_df = pd.DataFrame(wave_returns)
    return wave_ret_df


def compute_alpha_beta(wave_returns, benchmark_returns, windows, beta_lookback):
    """
    Returns:
      alpha_df: Wave x window alpha (annualized excess return)
      beta_df: Wave x ['Realized_Beta'] from regression vs benchmark
    """
    if wave_returns.empty or benchmark_returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Align indexes
    both = wave_returns.join(benchmark_returns, how="inner", rsuffix="_bench")
    bench_col = "Benchmark"

    alpha_rows = []
    beta_rows = []

    for wave in wave_returns.columns:
        if wave not in both.columns:
            continue

        wave_series = both[wave]
        bench_series = both[bench_col]

        # Alpha windows
        for label, days in windows.items():
            sub = both.iloc[-days:]
            if sub.empty:
                continue

            wr = sub[wave]
            br = sub[bench_col]

            # Average daily excess return * 252
            daily_excess = wr - br
            ann_alpha = daily_excess.mean() * 252.0

            alpha_rows.append(
                {
                    "Wave": wave,
                    "Window": label,
                    "Annualized_Alpha": ann_alpha,
                }
            )

        # Realized beta (1-year lookback)
        sub = both.iloc[-beta_lookback:]
        if len(sub) >= 30:
            wr = sub[wave]
            br = sub[bench_col]
            # Simple regression beta
            cov = np.cov(wr, br)[0, 1]
            var_b = np.var(br)
            beta = cov / var_b if var_b > 0 else np.nan
        else:
            beta = np.nan

        beta_rows.append(
            {
                "Wave": wave,
                "Realized_Beta": beta,
            }
        )

    alpha_df = pd.DataFrame(alpha_rows)
    beta_df = pd.DataFrame(beta_rows)

    if not alpha_df.empty:
        alpha_df = alpha_df.pivot(index="Wave", columns="Window", values="Annualized_Alpha")

    return alpha_df, beta_df


def compute_momentum_scores(wave_returns, benchmark_returns, windows, weights):
    """
    Momentum / leadership engine.
    Momentum = weighted sum of excess returns over given windows.
    Returns DataFrame: Wave, Momentum_Score, Tier
    """
    if wave_returns.empty or benchmark_returns.empty:
        return pd.DataFrame()

    # Align
    both = wave_returns.join(benchmark_returns, how="inner", rsuffix="_bench")
    bench_col = "Benchmark"

    scores = []
    for wave in wave_returns.columns:
        if wave not in both.columns:
            continue

        wave_series = both[wave]
        bench_series = both[bench_col]

        total_score = 0.0
        total_weight = 0.0

        for label, days in windows.items():
            w = weights.get(label, 0.0)
            if w <= 0:
                continue

            sub = both.iloc[-days:]
            if sub.empty:
                continue

            wr = sub[wave]
            br = sub[bench_col]
            excess = (1 + wr).prod() / (1 + br).prod() - 1.0  # total excess over that window
            total_score += w * excess
            total_weight += w

        if total_weight > 0:
            score = total_score / total_weight
        else:
            score = np.nan

        scores.append(
            {
                "Wave": wave,
                "Momentum_Score": score,
            }
        )

    df = pd.DataFrame(scores).set_index("Wave")

    if df.empty:
        return df

    # Rank & tier
    df["Rank"] = df["Momentum_Score"].rank(ascending=False, method="min")

    n = len(df)
    if n > 0:
        df["Tier"] = pd.qcut(
            df["Rank"],
            q=min(3, n),  # up to 3 tiers
            labels=["Leader", "Neutral", "Laggard"][: min(3, n)],
        )

    return df.reset_index()


def vix_to_exposure(vix_value):
    """
    Map VIX level to equity & SmartSafe exposures.
    """
    if vix_value is None or np.isnan(vix_value):
        # If we don't have VIX, default to full equity
        return 1.0, 0.0

    for max_vix, eq, ss in VIX_LADDER:
        if vix_value <= max_vix:
            return eq, ss

    # Fallback
    return 1.0, 0.0


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


def style_wave_table(df):
    """
    Simple formatting for wave summary table.
    """
    if df.empty:
        return df

    fmt_cols_pct = [
        c
        for c in df.columns
        if any(x in c for x in ["Alpha", "Momentum", "Equity_Alloc", "SmartSafe_Alloc"])
    ]
    fmt_cols_beta = [c for c in df.columns if "Beta" in c]

    def fmt(x, kind):
        if pd.isna(x):
            return ""
        if kind == "pct":
            return f"{x*100:,.2f}%"
        if kind == "beta":
            return f"{x:,.2f}"
        return x

    df_fmt = df.copy()

    for c in fmt_cols_pct:
        df_fmt[c] = df_fmt[c].apply(lambda x: fmt(x, "pct"))
    for c in fmt_cols_beta:
        df_fmt[c] = df_fmt[c].apply(lambda x: fmt(x, "beta"))

    return df_fmt


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

    # Collect all tickers we need
    holding_tickers = sorted(weights["Ticker"].unique())
    benchmark_tickers = sorted(cfg["Benchmark"].unique())
    all_tickers = list(set(holding_tickers + benchmark_tickers))

    # Download price history
    price_start = dt.date.today() - dt.timedelta(days=lookback_days + 30)
    prices = download_price_history(
        tickers=all_tickers, start_date=price_start
    )

    if prices.empty:
        st.error("Could not download price history. Check tickers or network.")
        st.stop()

    # Compute Wave returns
    wave_returns = compute_portfolio_returns(weights, prices)

    # Build benchmark returns frame (we'll use a single synthetic 'Benchmark' per Wave set)
    # For now: use SPY as universal benchmark, or mix later.
    if DEFAULT_BENCHMARK not in prices.columns:
        st.error(f"Default benchmark {DEFAULT_BENCHMARK} not found in price history.")
        st.stop()

    bench_px = prices[DEFAULT_BENCHMARK]
    bench_ret = bench_px.pct_change().fillna(0.0)
    benchmark_returns = pd.DataFrame({"Benchmark": bench_ret})

    # Alpha & beta
    alpha_df, beta_df = compute_alpha_beta(
        wave_returns, benchmark_returns, ALPHA_WINDOWS, BETA_LOOKBACK_DAYS
    )

    # Momentum / leadership
    momentum_df = compute_momentum_scores(
        wave_returns,
        benchmark_returns,
        MOMENTUM_WINDOWS,
        MOMENTUM_WEIGHTS,
    )

    # VIX + SPY history for charts
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

    # ===================== WAVE SUMMARY TABLE =====================

    st.markdown("## Wave-Level Summary (Alpha, Beta, Momentum, SmartSafe™)")

    # Build summary DataFrame
    summary_rows = []

    for wave in waves:
        cfg_row = cfg[cfg["Wave"] == wave].iloc[0]

        # Alpha
        alpha_row = alpha_df.loc[wave] if (not alpha_df.empty and wave in alpha_df.index) else None

        # Beta
        beta_val = (
            beta_df.set_index("Wave").loc[wave]["Realized_Beta"]
            if (not beta_df.empty and wave in beta_df["Wave"].values)
            else np.nan
        )

        # Momentum
        mom_row = (
            momentum_df.set_index("Wave").loc[wave]["Momentum_Score"]
            if (momentum_df is not None and not momentum_df.empty and wave in momentum_df["Wave"].values)
            else np.nan
        )
        tier = (
            momentum_df.set_index("Wave").loc[wave]["Tier"]
            if (momentum_df is not None and not momentum_df.empty and wave in momentum_df["Wave"].values)
            else ""
        )

        row = {
            "Wave": wave,
            "Mode": cfg_row["Mode"],
            "Benchmark": cfg_row["Benchmark"],
            "Beta_Target": cfg_row["Beta_Target"],
            "Realized_Beta": beta_val,
            "Momentum_Score": mom_row,
            "Momentum_Tier": tier,
            "Equity_Alloc": equity_pct,
            "SmartSafe_Alloc": smartsafe_pct,
        }

        # Add alpha windows
        for label in ALPHA_WINDOWS.keys():
            col_name = f"Alpha_{label}"
            if alpha_row is not None and label in alpha_row.index:
                row[col_name] = alpha_row[label]
            else:
                row[col_name] = np.nan

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort by Momentum
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

        # Add Google links
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

    if momentum_df is not None and not momentum_df.empty:
        st.write(
            """
            **Momentum_Score** is a weighted blend of 30D / 60D / 6M excess returns vs benchmark.  
            **Tier** buckets Waves into Leaders / Neutral / Laggards for capital rotation.
            """
        )
        st.dataframe(momentum_df.set_index("Wave"), use_container_width=True)
    else:
        st.info("Momentum metrics not available yet (insufficient history).")

    st.markdown("---")
    st.caption("WAVES Intelligence™ • Adaptive Portfolio Waves • SmartSafe™ • Vector OS™")


if __name__ == "__main__":
    main()
