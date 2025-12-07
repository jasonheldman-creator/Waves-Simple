import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

WAVE_WEIGHTS_PATH = "wave_weights.csv"
BENCHMARK_TICKER = "SPY"
VIX_TICKER = "^VIX"
BETA_TARGET = 0.90

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# CODE-MANAGED DEFAULT WAVE WEIGHTS
# (Used if CSV is missing or invalid)
# -------------------------------------------------------------------

DEFAULT_WAVE_WEIGHTS = [
    # --- AI_Wave ----------------------------------------------------
    {"wave": "AI_Wave", "ticker": "NVDA", "weight": 0.14},
    {"wave": "AI_Wave", "ticker": "MSFT", "weight": 0.12},
    {"wave": "AI_Wave", "ticker": "META", "weight": 0.11},
    {"wave": "AI_Wave", "ticker": "GOOGL", "weight": 0.10},
    {"wave": "AI_Wave", "ticker": "AMZN", "weight": 0.09},
    {"wave": "AI_Wave", "ticker": "AVGO", "weight": 0.08},
    {"wave": "AI_Wave", "ticker": "CRM",  "weight": 0.07},
    {"wave": "AI_Wave", "ticker": "PLTR", "weight": 0.06},
    {"wave": "AI_Wave", "ticker": "AMD",  "weight": 0.05},
    {"wave": "AI_Wave", "ticker": "SNOW", "weight": 0.04},

    # --- Growth_Wave (example) -------------------------------------
    {"wave": "Growth_Wave", "ticker": "AAPL", "weight": 0.12},
    {"wave": "Growth_Wave", "ticker": "MSFT", "weight": 0.11},
    {"wave": "Growth_Wave", "ticker": "NVDA", "weight": 0.11},
    {"wave": "Growth_Wave", "ticker": "META", "weight": 0.10},
    {"wave": "Growth_Wave", "ticker": "AMZN", "weight": 0.09},
    {"wave": "Growth_Wave", "ticker": "TSLA", "weight": 0.09},
    {"wave": "Growth_Wave", "ticker": "GOOGL", "weight": 0.09},
    {"wave": "Growth_Wave", "ticker": "AVGO", "weight": 0.08},
    {"wave": "Growth_Wave", "ticker": "ADBE", "weight": 0.07},
    {"wave": "Growth_Wave", "ticker": "NFLX", "weight": 0.06},

    # --- CleanTransitInfra_Wave (example) --------------------------
    {"wave": "CleanTransitInfra_Wave", "ticker": "TSLA", "weight": 0.14},
    {"wave": "CleanTransitInfra_Wave", "ticker": "NIO",  "weight": 0.10},
    {"wave": "CleanTransitInfra_Wave", "ticker": "F",    "weight": 0.10},
    {"wave": "CleanTransitInfra_Wave", "ticker": "GM",   "weight": 0.10},
    {"wave": "CleanTransitInfra_Wave", "ticker": "BLNK", "weight": 0.09},
    {"wave": "CleanTransitInfra_Wave", "ticker": "CHPT", "weight": 0.09},
    {"wave": "CleanTransitInfra_Wave", "ticker": "CAT",  "weight": 0.09},
    {"wave": "CleanTransitInfra_Wave", "ticker": "DE",   "weight": 0.09},
    {"wave": "CleanTransitInfra_Wave", "ticker": "PLUG", "weight": 0.10},
    {"wave": "CleanTransitInfra_Wave", "ticker": "NEE",  "weight": 0.10},
]


# -------------------------------------------------------------------
# DATA LOADING HELPERS
# -------------------------------------------------------------------

@st.cache_data
def load_weights(path: str = WAVE_WEIGHTS_PATH) -> pd.DataFrame:
    """
    Load wave weights, preferably from CSV. If the CSV is missing or invalid,
    fall back to the built-in DEFAULT_WAVE_WEIGHTS so the engine always runs.
    """
    # Try CSV first if it exists
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)

            if df.empty:
                st.warning(f"{path} is empty. Using built-in code defaults.")
                raise ValueError("Empty CSV")

            # Normalize column names (Wave / wave / WAVE etc.)
            col_map = {c.strip().lower(): c for c in df.columns}
            required = ["wave", "ticker", "weight"]
            missing = [r for r in required if r not in col_map]
            if missing:
                st.warning(
                    f"{path} is missing columns {missing}. "
                    "Using built-in code defaults instead."
                )
                raise ValueError("Missing required columns")

            wave_col = col_map["wave"]
            ticker_col = col_map["ticker"]
            weight_col = col_map["weight"]

            # Clean types & whitespace
            df[wave_col] = df[wave_col].astype(str).str.strip()
            df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
            df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")

            # Drop invalid weights
            df = df.dropna(subset=[weight_col])
            if df.empty:
                st.warning(
                    f"All weights in {path} are invalid after cleaning. "
                    "Using built-in code defaults."
                )
                raise ValueError("No valid rows")

            # Normalize weights within each wave
            df["weight"] = df.groupby(wave_col)[weight_col].transform(
                lambda x: x / x.sum()
            )

            df_out = df[[wave_col, ticker_col, "weight"]].rename(
                columns={wave_col: "wave", ticker_col: "ticker"}
            )

            return df_out
        except Exception as e:
            st.warning(f"Could not parse {path}: {e}. Falling back to code defaults.")

    # Fallback: code-managed defaults
    df_default = pd.DataFrame(DEFAULT_WAVE_WEIGHTS)
    df_default["wave"] = df_default["wave"].astype(str).str.strip()
    df_default["ticker"] = df_default["ticker"].astype(str).str.strip().str.upper()
    df_default["weight"] = pd.to_numeric(df_default["weight"], errors="coerce")

    # Normalize within wave
    df_default["weight"] = df_default.groupby("wave")["weight"].transform(
        lambda x: x / x.sum()
    )
    return df_default


@st.cache_data
def fetch_price_history(tickers, start_date, end_date) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers from Yahoo Finance
    using yfinance. Returns a DataFrame indexed by date.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns different shapes depending on #tickers
    if "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")
    return prices


# -------------------------------------------------------------------
# PORTFOLIO / METRIC LOGIC
# -------------------------------------------------------------------

def compute_wave_timeseries(
    weights_df: pd.DataFrame,
    wave_name: str,
    lookback_days: int,
) -> dict:
    """
    Compute daily returns, cumulative returns, alpha, and benchmark data
    for a given Wave over the lookback window.
    """
    wave_weights = (
        weights_df[weights_df["wave"] == wave_name]
        .copy()
        .reset_index(drop=True)
    )
    if wave_weights.empty:
        raise ValueError(f"No weights defined for wave '{wave_name}'")

    tickers = sorted(wave_weights["ticker"].unique().tolist())
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days + 10)

    # Fetch prices for wave tickers + benchmark + VIX
    all_tickers = tickers + [BENCHMARK_TICKER, VIX_TICKER]
    prices = fetch_price_history(all_tickers, start_date.date(), end_date.date())

    missing = [t for t in tickers if t not in prices.columns]
    if len(missing) == len(tickers):
        raise ValueError("No price data available for any tickers in this wave.")

    # Align weights to available tickers
    available_tickers = [t for t in tickers if t in prices.columns]
    wave_weights = wave_weights[wave_weights["ticker"].isin(available_tickers)].copy()
    wave_weights["weight"] = wave_weights["weight"] / wave_weights["weight"].sum()

    # Daily returns
    returns = prices.pct_change().dropna()

    wave_ret = (returns[available_tickers] * wave_weights.set_index("ticker")["weight"]).sum(
        axis=1
    )
    bench_ret = returns[BENCHMARK_TICKER]
    vix_level = prices[VIX_TICKER]

    # Cumulative
    wave_cum = (1 + wave_ret).cumprod() - 1
    bench_cum = (1 + bench_ret).cumprod() - 1
    alpha_daily = wave_ret - bench_ret
    alpha_cum = (1 + alpha_daily).cumprod() - 1

    return {
        "wave_returns": wave_ret,
        "bench_returns": bench_ret,
        "wave_cum": wave_cum,
        "bench_cum": bench_cum,
        "alpha_daily": alpha_daily,
        "alpha_cum": alpha_cum,
        "vix_level": vix_level,
        "prices": prices,
        "weights": wave_weights,
    }


def compute_metrics(ts_data: dict) -> dict:
    """
    Compute summary metrics for the top bar.
    """
    wave_ret = ts_data["wave_returns"]
    bench_ret = ts_data["bench_returns"]
    alpha_daily = ts_data["alpha_daily"]
    vix_level = ts_data["vix_level"]

    if len(wave_ret) == 0:
        raise ValueError("Not enough data to compute metrics.")

    # Today = last row
    wave_today = wave_ret.iloc[-1]
    bench_today = bench_ret.iloc[-1]
    alpha_today = alpha_daily.iloc[-1]

    # Beta estimate (simple regression-style)
    if len(wave_ret) > 1:
        cov = np.cov(wave_ret, bench_ret)[0, 1]
        var_b = np.var(bench_ret)
        beta_est = cov / var_b if var_b != 0 else np.nan
    else:
        beta_est = np.nan

    # Simple exposure / SmartSafe split based on beta
    if np.isnan(beta_est):
        exposure = 1.0
    else:
        # Clamp between 0 and 1, tilt toward BETA_TARGET
        exposure = max(0.0, min(1.0, beta_est / BETA_TARGET))
    smart_safe_alloc = 1.0 - exposure

    vix_latest = float(vix_level.iloc[-1]) if len(vix_level) else np.nan

    return {
        "wave_today": wave_today,
        "bench_today": bench_today,
        "alpha_today": alpha_today,
        "beta_est": beta_est,
        "exposure": exposure,
        "smart_safe_alloc": smart_safe_alloc,
        "vix_latest": vix_latest,
    }


# -------------------------------------------------------------------
# UI HELPERS
# -------------------------------------------------------------------

def pct(x: float) -> str:
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


def render_top_holdings(ts_data: dict):
    weights = ts_data["weights"].copy()
    if weights.empty:
        st.info("No holdings to display.")
        return

    # Get today's return per ticker from returns Series
    wave_prices = ts_data["prices"]
    last_two = wave_prices.iloc[-2:]
    rets = last_two.pct_change().iloc[-1]

    weights["Weight%"] = weights["weight"] * 100
    weights["Today%"] = weights["ticker"].map(rets) * 100

    # Google Finance links
    weights["Ticker"] = weights["ticker"].apply(
        lambda t: f"[{t}](https://www.google.com/finance/quote/{t}:NASDAQ)"
    )

    display_cols = ["Ticker", "Weight%", "Today%"]
    st.markdown("**Top Holdings (Live)**")
    st.dataframe(
        weights[display_cols].sort_values("Weight%", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

def main():
    weights_df = load_weights(WAVE_WEIGHTS_PATH)

    st.markdown(
        "<h1 style='margin-bottom:0.25rem;'>WAVES Institutional Console</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='opacity:0.8;'>Adaptive Portfolio Waves • AIWs/APWs • "
        "SmartSafe™ • VIX-gated risk • Alpha-Minus-Beta & Private Logic™</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("Wave & Mode")

    wave_names = sorted(weights_df["wave"].unique().tolist())
    if not wave_names:
        st.error("No waves found in weights configuration.")
        return

    selected_wave = st.sidebar.selectbox("Select Wave", wave_names, index=0)

    mode = st.sidebar.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    lookback_days = st.sidebar.slider(
        "Lookback (trading days)", min_value=60, max_value=365, value=365, step=5
    )

    show_alpha_curve = st.sidebar.checkbox("Show alpha curve", value=True)
    show_drawdown = st.sidebar.checkbox("Show drawdown", value=False)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # Engine status
    now_utc = datetime.now(timezone.utc)
    status_col1, status_col2 = st.columns([3, 1])
    with status_col2:
        st.markdown(
            "<div style='text-align:right; font-size:0.9rem;'>"
            "<span style='color:#00FF7F;'>Engine Status: LIVE / SANDBOX</span><br>"
            f"<span style='opacity:0.75;'>Last refresh: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Try to compute time series
    try:
        ts_data = compute_wave_timeseries(weights_df, selected_wave, lookback_days)
        metrics = compute_metrics(ts_data)
    except Exception as e:
        st.error(
            f"Could not compute returns for {selected_wave}. "
            f"Please verify tickers and weights. Details: {e}"
        )
        if show_debug:
            st.write("Weights DF:", weights_df)
        return

    # Top metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Wave Today", pct(metrics["wave_today"]))
    with m2:
        st.metric("Benchmark Today (SPY)", pct(metrics["bench_today"]))
    with m3:
        st.metric("Today Alpha", pct(metrics["alpha_today"]))
    with m4:
        st.metric("Estimated Beta vs SPY", f"{metrics['beta_est']:.2f}")
    with m5:
        st.metric("Exposure", pct(metrics["exposure"]))
    with m6:
        st.metric("SmartSafe™ Allocation", pct(metrics["smart_safe_alloc"]))

    st.markdown(
        f"<p style='font-size:0.85rem; opacity:0.75;'>"
        f"Mode: <b>{mode}</b> • Target β: {BETA_TARGET:.2f} • "
        f"VIX (latest): {metrics['vix_latest']:.1f if pd.notna(metrics['vix_latest']) else '—'}"
        "</p>",
        unsafe_allow_html=True,
    )

    # Main charts row
    upper_left, upper_right = st.columns([2, 1])

    # Wave vs Benchmark
    with upper_left:
        st.markdown(f"**{selected_wave} vs Benchmark (SPY)**")
        chart_df = pd.DataFrame(
            {
                "Wave": ts_data["wave_cum"],
                "Benchmark": ts_data["bench_cum"],
            }
        )
        chart_df.index.name = "Date"
        st.line_chart(chart_df, use_container_width=True)

        if show_alpha_curve:
            st.markdown("**Cumulative Alpha (Wave − Scaled Benchmark)**")
            alpha_df = ts_data["alpha_cum"].to_frame("Alpha")
            alpha_df.index.name = "Date"
            st.line_chart(alpha_df, use_container_width=True)

    with upper_right:
        render_top_holdings(ts_data)

    # Lower charts: SPY price and VIX
    lower_left, lower_right = st.columns(2)

    with lower_left:
        st.markdown("**SPY (Benchmark) – Price**")
        spy_price = ts_data["prices"][BENCHMARK_TICKER]
        spy_df = spy_price.to_frame("SPY")
        spy_df.index.name = "Date"
        st.line_chart(spy_df, use_container_width=True)

    with lower_right:
        st.markdown("**VIX – Level**")
        vix_df = ts_data["vix_level"].to_frame("VIX")
        vix_df.index.name = "Date"
        st.line_chart(vix_df, use_container_width=True)

    # Optional debug pane
    if show_debug:
        st.markdown("---")
        st.subheader("Debug Info")
        st.write("Raw weights DF:", weights_df)
        st.write(f"Selected wave: {selected_wave}")
        st.write("Wave weights used:", ts_data["weights"])
        st.write("Price data (tail):", ts_data["prices"].tail())


if __name__ == "__main__":
    main()
