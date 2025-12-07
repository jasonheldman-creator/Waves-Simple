import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta

# -----------------------------------------------------------------------------
# BASIC CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Where to optionally write engine logs in the cloud
LOG_DIR = Path("logs")
PERF_DIR = LOG_DIR / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)

# Human-facing vs internal wave names
WAVE_NAME_MAP = {
    "AI Leaders Wave": "AI_Wave",
    "Growth Wave": "Growth_Wave",
    "Quantum Wave": "Quantum_Wave",
    "Future Power & Energy Wave": "Future_Power_Wave",
    "Income Wave": "Income_Wave",
    "Small Cap Growth Wave": "SmallCap_Growth_Wave",
    "Small–Mid Cap Growth Wave": "SMid_Growth_Wave",
    "Crypto Income Wave": "Crypto_Income_Wave",
    "Clean Transit & Infrastructure Wave": "Clean_Transit_Wave",
}

DISPLAY_WAVES = list(WAVE_NAME_MAP.keys())

BENCHMARK_TICKER = "SPY"
VIX_TICKER = "^VIX"

LOOKBACK_YEARS = 1  # main chart lookback window


# -----------------------------------------------------------------------------
# DATA HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_wave_weights() -> pd.DataFrame:
    """Load weights from wave_weights.csv."""
    df = pd.read_csv("wave_weights.csv")
    # Expect columns: Ticker, Weight, Wave
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Weight"] = df["Weight"].astype(float)
    return df


def compute_drawdown(series: pd.Series) -> float:
    """Return max drawdown as a negative percentage (e.g. -0.25 = -25%)."""
    if series.empty:
        return 0.0
    cum_max = series.cummax()
    dd = (series / cum_max) - 1.0
    return float(dd.min())


@st.cache_data(ttl=900)
def run_wave_engine(
    display_wave_name: str,
    equity_exposure_pct: int,
    lookback_years: int = LOOKBACK_YEARS,
):
    """
    Core ENGINE logic.

    - Pulls tickers & weights for the selected Wave
    - Downloads historical prices for Wave tickers + SPY + VIX
    - Applies equity_exposure (cash buffer implied)
    - Computes Wave vs SPY equity curves and key metrics
    - Returns: metrics dict, curve_df, holdings_df
    """
    internal_wave = WAVE_NAME_MAP[display_wave_name]
    weights_df = load_wave_weights()
    wave_weights = weights_df[weights_df["Wave"] == internal_wave].copy()

    if wave_weights.empty:
        raise ValueError(f"No weights found for wave '{internal_wave}' in wave_weights.csv")

    tickers = sorted(wave_weights["Ticker"].unique().tolist())

    # Normalize weights to 1.0
    w = wave_weights.set_index("Ticker")["Weight"].astype(float)
    w = w / w.sum()

    # Lookback window
    end_date = date.today()
    start_date = end_date - timedelta(days=int(lookback_years * 365) + 10)

    # Download historical prices (Adj Close)
    dl_tickers = tickers + [BENCHMARK_TICKER, VIX_TICKER]
    prices = yf.download(
        dl_tickers,
        start=start_date,
        end=end_date + timedelta(days=1),
        progress=False,
    )

    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Adj Close"]
    else:
        prices = prices["Adj Close"]

    prices = prices.dropna(how="all")

    # Ensure we have benchmark & VIX
    if BENCHMARK_TICKER not in prices.columns:
        raise ValueError(f"Benchmark {BENCHMARK_TICKER} not found in downloaded data.")
    if VIX_TICKER not in prices.columns:
        raise ValueError(f"VIX {VIX_TICKER} not found in downloaded data.")

    # Align universe tickers present in prices
    present_tickers = [t for t in tickers if t in prices.columns]
    if not present_tickers:
        raise ValueError("None of the wave's tickers were found in price data.")

    wave_prices = prices[present_tickers].copy()
    bench_prices = prices[BENCHMARK_TICKER].copy()
    vix_series = prices[VIX_TICKER].copy()

    # Reindex weights to present tickers and renormalize
    w = w.reindex(present_tickers).fillna(0.0)
    w = w / w.sum()

    # Daily returns
    wave_rets = wave_prices.pct_change().fillna(0.0)
    bench_rets = bench_prices.pct_change().fillna(0.0)

    # Apply equity exposure (cash buffer)
    equity_exposure = equity_exposure_pct / 100.0
    cash_buffer = 1.0 - equity_exposure

    wave_rets_exposed = wave_rets.mul(equity_exposure)
    # Daily portfolio return = sum(weighted equity returns) + 0 * cash
    port_rets = (wave_rets_exposed * w.values).sum(axis=1)

    # Equity curves (normalize to 100)
    wave_curve = (1.0 + port_rets).cumprod() * 100.0
    bench_curve = (1.0 + bench_rets).cumprod() * 100.0

    # Align
    common_index = wave_curve.index.intersection(bench_curve.index)
    wave_curve = wave_curve.loc[common_index]
    bench_curve = bench_curve.loc[common_index]
    vix_series = vix_series.loc[common_index]

    if len(common_index) < 2:
        raise ValueError("Not enough overlapping data to compute metrics.")

    # Metrics
    total_return = wave_curve.iloc[-1] / wave_curve.iloc[0] - 1.0
    bench_total = bench_curve.iloc[-1] / bench_curve.iloc[0] - 1.0

    wave_daily_ret = port_rets.loc[common_index]
    bench_daily_ret = bench_rets.loc[common_index]

    today_ret = float(wave_daily_ret.iloc[-1])
    alpha_total = total_return - bench_total
    max_dd = compute_drawdown(wave_curve)

    # SPY & VIX "today"
    spy_last = float(bench_prices.loc[common_index].iloc[-1])
    vix_last = float(vix_series.iloc[-1])

    # Holdings today % move (use last 2 prices)
    last_two = wave_prices.loc[common_index].iloc[-2:]
    today_moves = (last_two.iloc[-1] / last_two.iloc[0] - 1.0)

    holdings_df = pd.DataFrame(
        {
            "Ticker": present_tickers,
            "Weight %": (w.values * 100.0),
            "Today %": today_moves.reindex(present_tickers).fillna(0.0).values * 100.0,
        }
    ).sort_values("Weight %", ascending=False)

    top10_df = holdings_df.head(10).reset_index(drop=True)

    # Equity curve dataframe for plotting
    curve_df = pd.DataFrame(
        {
            "Date": common_index,
            display_wave_name: wave_curve.values,
            BENCHMARK_TICKER: bench_curve.values,
        }
    )

    # Optional: write a simple log CSV to cloud storage
    log_path = PERF_DIR / f"{internal_wave}_performance_cloud.csv"
    curve_df.to_csv(log_path, index=False)

    metrics = {
        "total_return": float(total_return),
        "bench_total": float(bench_total),
        "alpha_total": float(alpha_total),
        "today_ret": float(today_ret),
        "max_drawdown": float(max_dd),
        "spy_last": spy_last,
        "vix_last": vix_last,
        "equity_exposure": equity_exposure,
        "cash_buffer": cash_buffer,
    }

    return metrics, curve_df, top10_df


def style_holdings(df: pd.DataFrame):
    """Return a Styler with green/red today %."""
    def color_today(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 0:
            return "color: #00FF7F; font-weight: 600;"  # green
        if v < 0:
            return "color: #FF6B6B; font-weight: 600;"  # red
        return "color: #CCCCCC;"

    styler = (
        df.style
        .format({"Weight %": "{:.2f}%", "Today %": "{:+.2f}%"})
        .applymap(color_today, subset=["Today %"])
    )
    return styler


# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🌊 WAVES Intelligence™")
st.sidebar.markdown("**DESKTOP ENGINE + CLOUD SNAPSHOT**")

selected_wave = st.sidebar.selectbox(
    "Select Wave",
    DISPLAY_WAVES,
    index=0,
)

st.sidebar.markdown("**Risk Mode (label only)**")
risk_mode = st.sidebar.radio(
    "",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

equity_exposure_slider = st.sidebar.slider(
    "Equity Exposure (target)",
    min_value=0,
    max_value=100,
    value=90,
    step=5,
)

st.sidebar.caption(
    "Keep this console open while the engine logic runs in the background. "
    "Equity exposure is applied to all Waves in this view."
)

st.title("WAVES Institutional Console")

st.caption(
    f"Live / demo console for **WAVES Intelligence™** — showing **{selected_wave}**.  "
    f"Mode (label): **{risk_mode}**  |  Benchmark: **{BENCHMARK_TICKER}**"
)

# Run ENGINE in the cloud
with st.spinner("Running WAVES engine in cloud for this Wave…"):
    try:
        metrics, curve_df, top10_df = run_wave_engine(
            selected_wave,
            equity_exposure_slider,
            lookback_years=LOOKBACK_YEARS,
        )
    except Exception as e:
        st.error(
            "There was a problem running the engine for this Wave.\n\n"
            f"Details: {e}"
        )
        st.stop()

# -----------------------------------------------------------------------------
# TOP METRIC STRIP
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Return (lookback)",
        f"{metrics['total_return']*100:,.2f}%",
    )

with col2:
    st.metric(
        "Today",
        f"{metrics['today_ret']*100:,.2f}%",
    )

with col3:
    st.metric(
        "Max Drawdown",
        f"{metrics['max_drawdown']*100:,.2f}%",
    )

with col4:
    st.metric(
        f"Alpha vs {BENCHMARK_TICKER}",
        f"{metrics['alpha_total']*100:,.2f}%",
    )

with col5:
    st.metric(
        "Equity | Cash",
        f"{metrics['equity_exposure']*100:,.0f}% / {metrics['cash_buffer']*100:,.0f}%",
    )

st.markdown(
    f"**{BENCHMARK_TICKER} (benchmark)** and **{VIX_TICKER}** data are from Yahoo Finance "
    f"Adj Close. Last SPY price: **{metrics['spy_last']:.2f}**  |  VIX: **{metrics['vix_last']:.2f}**"
)

# -----------------------------------------------------------------------------
# MAIN PANELS
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([2.2, 1.1])

with left_col:
    st.subheader("Performance Curve")

    chart_df = curve_df.set_index("Date")
    st.line_chart(chart_df)

    st.caption(
        "Curve is normalized to 100 at the start of the lookback window. "
        "Source: Yahoo Finance (Adj Close)."
    )

with right_col:
    st.subheader("Holdings, Weights & Risk")

    st.markdown("**Top 10 Positions — Google Finance Links (Bloomberg-style)**")

    # Build a simple markdown table with clickable Google Finance links
    md_lines = ["| Ticker | Weight % | Today % |", "| :----- | -------: | ------: |"]
    for _, row in top10_df.iterrows():
        t = row["Ticker"]
        w = row["Weight %"]
        td = row["Today %"]
        url = f"https://www.google.com/finance/quote/{t}:NASDAQ"
        md_lines.append(f"| [{t}]({url}) | {w:,.2f}% | {td:+.2f}% |")

    st.markdown("\n".join(md_lines))

    st.markdown("—")
    st.markdown("**Full Wave universe table**")

    full_styler = style_holdings(top10_df)  # reuse, but you can switch to full holdings if preferred
    st.dataframe(full_styler, height=320, use_container_width=True)

st.markdown("---")
st.caption(
    "WAVES Institutional Console — demo view only. Returns & metrics are based on public "
    "market data via Yahoo Finance and do not represent live trading or an offer of advisory services."
)
