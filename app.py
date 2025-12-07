import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

LOOKBACK_DAYS = 365          # 1-year performance window
BENCHMARK_TICKER = "SPY"     # Equity benchmark
CASH_TICKER = "SGOV"         # SmartSafe-style cash proxy
VIX_TICKER = "^VIX"          # VIX index
WAVE_WEIGHTS_FILE = "wave_weights.csv"

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# DATA LOADERS
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_wave_universe(csv_path: str = WAVE_WEIGHTS_FILE) -> pd.DataFrame:
    """Load wave universe from wave_weights.csv."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path} in working directory.")

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Expect at least: Ticker, Wave, Weight or Weight %
    if "Weight" not in df.columns:
        if "Weight %" in df.columns:
            df["Weight"] = df["Weight %"].astype(float) / 100.0
        else:
            raise ValueError(
                "wave_weights.csv must contain either 'Weight' or 'Weight %' column."
            )

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Wave"] = df["Wave"].astype(str).str.strip()

    # Drop any blank rows
    df = df.dropna(subset=["Ticker", "Wave", "Weight"])

    # Normalize weights per Wave
    df["Weight"] = df["Weight"].astype(float)
    df["Weight"] = df.groupby("Wave")["Weight"].transform(
        lambda w: w / w.sum()
    )

    return df


@st.cache_data(show_spinner=False)
def fetch_prices(tickers, start_date: datetime) -> pd.DataFrame:
    """Fetch Adj Close prices for a list of tickers."""
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers,
        start=start_date,
        progress=False,
        auto_adjust=False,
    )

    # yfinance returns different shapes depending on # tickers
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        adj = data["Adj Close"]
    elif isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        adj = data["Adj Close"]
    else:
        # Single series fallback
        adj = data

    if isinstance(adj, pd.Series):
        adj = adj.to_frame(tickers[0])

    adj = adj.ffill().dropna(how="all")
    return adj


# ------------------------------------------------------------
# STRATEGY LOGIC
# ------------------------------------------------------------

def compute_vix_based_exposure(vix_value: float, slider_equity: float = 0.90):
    """
    Simple VIX ladder:
      VIX < 16  -> 95% equity
      16-22     -> 90%
      22-28     -> 80%
      28-35     -> 65%
      >35       -> 50%

    Final equity exposure = min(ladder, slider setting).
    """
    base_equity = slider_equity

    if vix_value is None or np.isnan(vix_value):
        eq = base_equity
    elif vix_value < 16:
        eq = 0.95
    elif vix_value < 22:
        eq = 0.90
    elif vix_value < 28:
        eq = 0.80
    elif vix_value < 35:
        eq = 0.65
    else:
        eq = 0.50

    # Respect the slider as an upper bound
    eq = min(eq, slider_equity)
    cash = 1.0 - eq
    return eq, cash


def compute_portfolio_curve(
    equity_prices: pd.DataFrame,
    weights: np.ndarray,
    cash_series: pd.Series,
    equity_exposure: float,
    cash_exposure: float,
) -> pd.Series:
    """Build blended portfolio: equity basket + SmartSafe cash."""
    # Normalize equity basket
    equity_norm = equity_prices / equity_prices.iloc[0]
    equity_port = (equity_norm * weights).sum(axis=1)

    # Normalize cash proxy
    cash_norm = cash_series / cash_series.iloc[0]

    # Blend
    blended = equity_exposure * equity_port + cash_exposure * cash_norm
    return blended


def compute_perf_stats(port_curve: pd.Series, benchmark_curve: pd.Series):
    """Return summary statistics for dashboard."""
    port_ret = port_curve.pct_change().fillna(0.0)
    bench_ret = benchmark_curve.pct_change().fillna(0.0)

    # Total return
    total_ret = (port_curve.iloc[-1] / port_curve.iloc[0] - 1.0) * 100.0
    bench_total = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1.0) * 100.0
    alpha_total = total_ret - bench_total

    # Today
    if len(port_curve) >= 2:
        today_ret = (port_curve.iloc[-1] / port_curve.iloc[-2] - 1.0) * 100.0
    else:
        today_ret = 0.0

    # Max drawdown
    running_max = port_curve.cummax()
    drawdown = port_curve / running_max - 1.0
    max_dd = drawdown.min() * 100.0

    return {
        "total_return": total_ret,
        "today_return": today_ret,
        "max_drawdown": max_dd,
        "alpha_vs_bench": alpha_total,
    }


def compute_ticker_daily_moves(price_df: pd.DataFrame) -> pd.Series:
    """Return today's % move per ticker from a price dataframe."""
    if price_df.shape[0] < 2:
        return pd.Series(0.0, index=price_df.columns)

    last = price_df.iloc[-1]
    prev = price_df.iloc[-2]
    moves = (last / prev - 1.0) * 100.0
    return moves


# ------------------------------------------------------------
# HOLDINGS TABLE STYLING
# ------------------------------------------------------------

def build_top_holdings_table(wave_slice: pd.DataFrame,
                             daily_moves: pd.Series) -> pd.DataFrame:
    """Construct dataframe for top 10 positions."""
    df = wave_slice.copy()
    df = df.sort_values("Weight", ascending=False).head(10).reset_index(drop=True)

    df["Weight %"] = df["Weight"] * 100.0
    df["Today %"] = df["Ticker"].map(daily_moves).fillna(0.0)

    return df[["Ticker", "Weight %", "Today %"]]


def style_top_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Styled DataFrame with red/green Today % cells."""
    def color_today(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #00ff88; font-weight: 600;"   # green
        elif val < 0:
            return "color: #ff4d4f; font-weight: 600;"   # red
        return ""

    styled = (
        df.style.format(
            {
                "Weight %": "{:.2f}%",
                "Today %": "{:.2f}%",
            }
        )
        .applymap(color_today, subset=["Today %"])
    )
    return styled


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

def main():
    # ----- Load universe -----
    try:
        universe = load_wave_universe(WAVE_WEIGHTS_FILE)
    except Exception as e:
        st.error(f"Problem loading {WAVE_WEIGHTS_FILE}: {e}")
        st.stop()

    wave_names = sorted(universe["Wave"].unique().tolist())

    # ----- Sidebar: Engine Controls -----
    st.sidebar.markdown("### 🌊 WAVES Intelligence™")
    st.sidebar.markdown("**DESKTOP ENGINE + CLOUD SNAPSHOT**")
    st.sidebar.caption(
        "Institutional console for WAVES Intelligence™ — select one of your locked Waves."
    )

    selected_wave = st.sidebar.selectbox("Select Wave", wave_names, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Risk Mode (label only)**")
    risk_mode = st.sidebar.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
        label_visibility="collapsed",
    )

    equity_slider = st.sidebar.slider(
        "Equity Exposure (target)",
        min_value=50,
        max_value=100,
        step=5,
        value=90,
        help="Target equity exposure; VIX ladder may dial this back automatically.",
    )
    target_equity_slider = equity_slider / 100.0

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Keep this console open while the desktop engine runs in the background, "
        "or let the cloud logic fetch prices live via Yahoo Finance."
    )

    # ----- Top-level layout -----
    st.markdown(
        "<h2 style='margin-bottom:0.2rem;'>WAVES Institutional Console</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Live / demo console for WAVES Intelligence™ — showing "
        f"<strong>{selected_wave}</strong>.  "
        f"Risk Mode: <strong>{risk_mode}</strong>.",
        unsafe_allow_html=True,
    )

    col_header_left, col_header_mid = st.columns([3, 1])

    # Determine time window
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    # ----- Filter universe for selected Wave -----
    this_wave = universe[universe["Wave"] == selected_wave].copy()
    if this_wave.empty:
        st.error(f"No holdings found in wave_weights.csv for Wave '{selected_wave}'.")
        st.stop()

    tickers = sorted(this_wave["Ticker"].unique().tolist())

    # ----- Fetch prices -----
    with st.spinner("Fetching price history..."):
        equity_prices = fetch_prices(tickers, start_date)
        bench_prices = fetch_prices([BENCHMARK_TICKER], start_date)
        cash_prices = fetch_prices([CASH_TICKER], start_date)
        vix_prices = fetch_prices([VIX_TICKER], start_date)

    # Align all series to common dates
    bench_series = bench_prices[BENCHMARK_TICKER].dropna()
    cash_series = cash_prices[CASH_TICKER].dropna()
    vix_series = vix_prices[VIX_TICKER].dropna()

    common_index = equity_prices.index.intersection(bench_series.index)
    common_index = common_index.intersection(cash_series.index)
    if len(common_index) < 10:
        st.warning("Not enough overlapping price history to build the curve.")
        st.stop()

    equity_prices = equity_prices.loc[common_index]
    bench_series = bench_series.loc[common_index]
    cash_series = cash_series.loc[common_index]

    latest_vix = float(vix_series.iloc[-1]) if len(vix_series) else np.nan
    vix_equity, vix_cash = compute_vix_based_exposure(
        latest_vix, slider_equity=target_equity_slider
    )

    # Use weights aligned to equity_prices columns
    weight_map = this_wave.set_index("Ticker")["Weight"]
    aligned_weights = []
    for t in equity_prices.columns:
        aligned_weights.append(weight_map.get(t, 0.0))
    aligned_weights = np.array(aligned_weights, dtype=float)
    if aligned_weights.sum() <= 0:
        st.error("Weights for this Wave sum to 0. Check wave_weights.csv.")
        st.stop()

    aligned_weights = aligned_weights / aligned_weights.sum()

    # ----- Build portfolio curve -----
    port_curve = compute_portfolio_curve(
        equity_prices,
        aligned_weights,
        cash_series,
        equity_exposure=vix_equity,
        cash_exposure=vix_cash,
    )

    # Normalize to 100 for display
    port_norm = port_curve / port_curve.iloc[0] * 100.0
    bench_norm = bench_series / bench_series.iloc[0] * 100.0

    stats = compute_perf_stats(port_norm, bench_norm)

    # ----- Header metrics -----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return (lookback)", f"{stats['total_return']:.2f}%")
    col2.metric("Today", f"{stats['today_return']:.2f}%")
    col3.metric("Max Drawdown", f"{stats['max_drawdown']:.2f}%")
    col4.metric("Alpha vs SPY", f"{stats['alpha_vs_bench']:.2f}%")

    # Show benchmark + VIX info
    bench_total = (bench_norm.iloc[-1] / bench_norm.iloc[0] - 1.0) * 100.0
    st.caption(
        f"SPY (benchmark) over this window: **{bench_total:.2f}%**  |  "
        f"VIX (spot): **{latest_vix:.2f}**  |  "
        f"Equity exposure in engine: **{vix_equity*100:.1f}%**  |  "
        f"SmartSafe™ / cash: **{vix_cash*100:.1f}%**"
    )

    # ----- Main two-column layout -----
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Performance Curve")
        perf_df = pd.DataFrame(
            {
                selected_wave: port_norm,
                BENCHMARK_TICKER: bench_norm,
            }
        )
        st.line_chart(perf_df)

        st.caption(
            "Curve is normalized to 100 at the start of the lookback window. "
            "Source: Yahoo Finance (Adj Close)."
        )

    with right:
        st.subheader("Holdings, Weights & Risk")

        # Daily moves per ticker
        ticker_moves = compute_ticker_daily_moves(equity_prices)

        top_holdings = build_top_holdings_table(this_wave, ticker_moves)

        # Use LinkColumn so the label is the ticker itself
        st.markdown("**Top 10 Positions — Google Finance Links (Bloomberg-style)**")

        st.dataframe(
            top_holdings,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.LinkColumn(
                    "Ticker",
                    help="Open in Google Finance",
                    url=lambda ticker: f"https://www.google.com/finance/quote/{ticker}:NASDAQ",
                ),
                "Weight %": st.column_config.NumberColumn(
                    "Weight %", format="%.2f%%"
                ),
                "Today %": st.column_config.NumberColumn(
                    "Today %", format="%.2f%%"
                ),
            },
        )

        # Add styled version with red/green today column below (optional)
        styled = style_top_holdings(top_holdings)
        st.write(styled)

        with st.expander("Full Wave universe table"):
            full_table = this_wave.copy()
            full_table["Weight %"] = full_table["Weight"] * 100.0
            st.dataframe(
                full_table[["Ticker", "Weight %"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Weight %": st.column_config.NumberColumn(
                        "Weight %", format="%.2f%%"
                    )
                },
            )

    st.markdown(
        "<hr style='margin-top:2rem;margin-bottom:0.5rem;'/>",
        unsafe_allow_html=True,
    )
    st.caption(
        "WAVES Institutional Console — demo view only. Returns & metrics are based on "
        "public market data via Yahoo Finance and do not represent live trading or an "
        "offer of advisory services."
    )


if __name__ == '__main__':
    main()
