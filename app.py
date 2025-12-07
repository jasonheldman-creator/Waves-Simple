# app.py — WAVES Institutional Console (clean build)

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ------------------------------------------------------------
# Basic page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="🌊",
    layout="wide",
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + underscored column names so we can handle variations."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def safe_pct(x, decimals=2):
    """Format a decimal (0.10 -> '10.00%'), robust to None / NaN / errors."""
    try:
        if x is None:
            return "—"
        if isinstance(x, (list, tuple)):
            return "—"
        if pd.isna(x):
            return "—"
        return f"{float(x) * 100:.{decimals}f}%"
    except Exception:
        return "—"


def max_drawdown(series: pd.Series) -> float:
    """
    Compute max drawdown of an equity curve indexed to 1.0.
    Returns a *negative* decimal (e.g. -0.25 for -25%).
    """
    if series is None or series.empty:
        return float("nan")
    running_max = series.cummax()
    drawdowns = series / running_max - 1.0
    return float(drawdowns.min())


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(tickers, start, end):
    """
    Fetch adjusted close prices for a list of tickers between start and end.
    Returns a DataFrame: index=dates, columns=tickers.
    """
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    # yfinance returns different shapes depending on # of tickers
    def extract_close(df):
        if isinstance(df.columns, pd.MultiIndex):
            if ("Adj Close" in df.columns.levels[0]):
                return df["Adj Close"]
            elif ("Close" in df.columns.levels[0]):
                return df["Close"]
            else:
                # take the first level as a fallback
                first_level = df.columns.levels[0][0]
                return df[first_level]
        else:
            if "Adj Close" in df.columns:
                return df["Adj Close"]
            elif "Close" in df.columns:
                return df["Close"]
            else:
                return df

    close = extract_close(data)

    # If only one ticker, make sure columns are [ticker]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    # Ensure columns names are plain tickers
    close.columns = [str(c).upper().split()[0] for c in close.columns]

    # Keep only requested tickers (in case of extras)
    cols = [t for t in tickers if t in close.columns]
    return close[cols].sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def load_weights() -> pd.DataFrame:
    """
    Load and standardize wave_weights.csv.
    Expected columns (case-insensitive, spaces ok):
        Wave, Ticker, Weight or Weight %
    """
    df = pd.read_csv("wave_weights.csv")
    df = normalize_cols(df)

    # Standardize column names
    # Try to resolve ticker column
    ticker_col = None
    for c in df.columns:
        if c in ("ticker", "symbol"):
            ticker_col = c
            break
    if ticker_col is None:
        raise ValueError("wave_weights.csv must have a 'Ticker' column.")

    # Resolve wave column
    wave_col = None
    for c in df.columns:
        if c in ("wave", "wavename", "wave_name"):
            wave_col = c
            break
    if wave_col is None:
        raise ValueError("wave_weights.csv must have a 'Wave' column.")

    # Resolve weight column (either weight or weight_%)
    weight_col = None
    for c in df.columns:
        if c in ("weight", "weight_pct", "weight_%", "weight_percent"):
            weight_col = c
            break
    if weight_col is None:
        raise ValueError(
            "wave_weights.csv must have a 'Weight' or 'Weight %' column."
        )

    w = df[[wave_col, ticker_col, weight_col]].copy()
    w.columns = ["wave", "ticker", "raw_weight"]

    # Convert to numeric weight in *decimal* form
    w["ticker"] = w["ticker"].astype(str).str.upper().str.strip()
    w["wave"] = w["wave"].astype(str).str.strip()

    w["raw_weight"] = pd.to_numeric(w["raw_weight"], errors="coerce")
    # Heuristic: if any weight > 1.5, treat as percent
    if (w["raw_weight"] > 1.5).any():
        w["weight"] = w["raw_weight"] / 100.0
    else:
        w["weight"] = w["raw_weight"]

    # Normalize weights per wave to sum to 1.0
    w["weight"] = w.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )

    return w[["wave", "ticker", "weight"]]


def compute_wave_vs_benchmark(
    weights_df: pd.DataFrame,
    wave_key: str,
    benchmark: str = "SPY",
    lookback_months: int = 6,
):
    """
    Build equity curves for a given Wave vs a benchmark.
    Returns:
        portfolio_curve (Series),
        benchmark_curve (Series),
        top_holdings (DataFrame: ticker, weight),
        daily_portfolio_returns (Series),
        last_prices (Series for today % change reference),
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(lookback_months * 30.5))

    # Filter weights for this wave
    wf = weights_df[weights_df["wave"].str.lower() == wave_key.lower()].copy()
    if wf.empty:
        raise ValueError(f"No weights found for wave '{wave_key}'.")

    tickers = sorted(wf["ticker"].unique().tolist())
    if benchmark.upper() not in tickers:
        tickers_all = tickers + [benchmark.upper()]
    else:
        tickers_all = tickers

    prices = fetch_price_history(tickers_all, start, end)
    if prices.empty:
        raise ValueError("No price data returned for selected wave / window.")

    # Split portfolio tickers vs benchmark
    bench = benchmark.upper()
    if bench not in prices.columns:
        raise ValueError(f"Benchmark {bench} not found in downloaded data.")

    price_port = prices[tickers]  # only the portfolio names
    price_bench = prices[bench]

    # Forward-fill to handle missing days, then drop rows where benchmark is NA
    price_port = price_port.ffill()
    price_bench = price_bench.ffill()
    common_idx = price_bench.dropna().index
    price_port = price_port.loc[common_idx].dropna(how="all")
    price_bench = price_bench.loc[common_idx]

    if len(price_port) < 5:
        raise ValueError("Not enough price history to compute performance.")

    # Align weights vector to columns
    w_vec = (
        wf.set_index("ticker")["weight"]
        .reindex(price_port.columns)
        .fillna(0.0)
    )

    # Normalize to 1.0 just in case
    total_w = w_vec.sum()
    if total_w == 0:
        raise ValueError("Weights sum to zero for this wave.")
    w_vec = w_vec / total_w

    # Build index curves normalized to 1.0 at start
    port_norm = price_port / price_port.iloc[0]
    bench_norm = price_bench / price_bench.iloc[0]

    portfolio_curve = (port_norm * w_vec).sum(axis=1)
    benchmark_curve = bench_norm

    # Daily returns
    portfolio_ret = portfolio_curve.pct_change().dropna()
    benchmark_ret = benchmark_curve.pct_change().dropna()

    # Today % change (from prices)
    today_close = prices.iloc[-1]
    if len(prices) >= 2:
        prev_close = prices.iloc[-2]
        today_move = (today_close / prev_close - 1.0)  # decimal
    else:
        today_move = pd.Series(index=prices.columns, dtype=float)

    # Top 10 holdings by weight
    top_holdings = (
        wf.sort_values("weight", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return (
        portfolio_curve,
        benchmark_curve,
        portfolio_ret,
        benchmark_ret,
        top_holdings,
        today_move,
    )


def style_top_holdings(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Style the Top 10 holdings table:
        - Link tickers to Google Finance
        - Today % change green/red
    """
    base = df.copy()

    # Build Google Finance URLs
    def google_link(t):
        t = str(t).upper().strip()
        if not t:
            return ""
        return f"https://www.google.com/finance/quote/{t}:NASDAQ"

    base["Google Finance"] = base["Ticker"].apply(google_link)

    # Order + format columns
    base = base[["Ticker", "Weight %", "Today %", "Google Finance"]]

    styler = base.style.format(
        {
            "Weight %": lambda x: f"{x:.2f}%",
            "Today %": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
            "Google Finance": lambda url: f"🔗 Open" if url else "",
        }
    )

    def highlight_today(val):
        if pd.isna(val):
            return ""
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 0:
            return "color: #2ecc71;"  # green
        elif v < 0:
            return "color: #e74c3c;"  # red
        else:
            return ""

    styler = styler.applymap(
        highlight_today, subset=pd.IndexSlice[:, ["Today %"]]
    )

    # Make the 'Ticker' column look clickable (we'll show URL in tooltip)
    def ticker_fmt(t):
        return f"{t}"

    styler = styler.format({"Ticker": ticker_fmt})

    return styler


# ------------------------------------------------------------
# Wave config (labels -> underlying wave keys)
# ------------------------------------------------------------

WAVE_CONFIG = {
    # label shown to user        # key used in wave_weights.csv "wave" column
    "AI Leaders Wave": "AI_Wave",
    "Quantum Wave": "Quantum_Wave",
    "Income Wave": "Income_Wave",
    "Future Power & Energy Wave": "Future_Power_Energy_Wave",
    "Small Cap Growth Wave": "Small_Cap_Growth_Wave",
    "Small–Mid Growth Wave": "Small_Mid_Growth_Wave",
    "Crypto Income Wave": "Crypto_Income_Wave",
    "Clean Transit & Infra Wave": "Clean_Transit_Infrastructure_Wave",
    "S&P 500 Wave": "SP500_Wave",
}

DEFAULT_BENCHMARK = "SPY"
LOOKBACK_MONTHS = 6

# ------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------

st.markdown(
    "<h1 style='margin-bottom:0.2rem;'>WAVES Institutional Console</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Live / demo console for **WAVES Intelligence™** — showing AI-managed Adaptive Index Waves™.",
    unsafe_allow_html=True,
)
st.markdown("---")

left_col, right_col = st.columns([0.25, 0.75])

# ---------------- Sidebar-like controls on left ----------------
with left_col:
    st.markdown("#### Engine Controls")

    weights_df = load_weights()

    wave_label = st.selectbox(
        "Select Wave",
        list(WAVE_CONFIG.keys()),
        index=0,
    )
    wave_key = WAVE_CONFIG[wave_label]

    st.markdown("**Risk Mode (label only)**")
    risk_mode = st.radio(
        "",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
        horizontal=False,
    )

    exposure = st.slider(
        "Equity Exposure (target)",
        min_value=0,
        max_value=100,
        value=90,
        step=5,
        help=(
            "Label only for now. In full engine mode this would control "
            "equity vs SmartSafe™ allocation."
        ),
    )

    st.caption(
        "Keep this console open while the engine logic runs in the background. "
        "Cloud view uses public market data via Yahoo Finance."
    )

# ---------------- Main panel on right ----------------
with right_col:
    st.markdown(
        f"**Live view** — {wave_label}.  "
        f"Risk Mode *(label)*: **{risk_mode}**  |  "
        f"Benchmark: **{DEFAULT_BENCHMARK}**  |  Lookback: **{LOOKBACK_MONTHS}m**"
    )

    with st.spinner("Fetching market data and computing performance…"):
        try:
            (
                curve_wave,
                curve_bench,
                ret_wave,
                ret_bench,
                top_holdings,
                today_move,
            ) = compute_wave_vs_benchmark(
                weights_df,
                wave_key=wave_key,
                benchmark=DEFAULT_BENCHMARK,
                lookback_months=LOOKBACK_MONTHS,
            )

            # -------- Metrics --------
            # Total return
            tot_wave = curve_wave.iloc[-1] / curve_wave.iloc[0] - 1.0
            tot_bench = curve_bench.iloc[-1] / curve_bench.iloc[0] - 1.0
            alpha_total = tot_wave - tot_bench

            # Today return (last daily)
            today_wave = ret_wave.iloc[-1] if not ret_wave.empty else float("nan")

            # Max drawdown
            mdd = max_drawdown(curve_wave)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return (lookback)", safe_pct(tot_wave))
            m2.metric("Today", safe_pct(today_wave))
            m3.metric("Max Drawdown", safe_pct(mdd))
            # robust alpha display
            try:
                alpha_display = safe_pct(alpha_total)
            except Exception:
                alpha_display = "—"
            m4.metric("Alpha vs SPY", alpha_display)

            # -------- Performance curve --------
            st.markdown("### Performance Curve")

            perf_df = pd.DataFrame(
                {
                    wave_label: curve_wave,
                    DEFAULT_BENCHMARK: curve_bench,
                }
            )
            perf_df = perf_df / perf_df.iloc[0] * 100.0  # index to 100

            st.line_chart(perf_df)

            st.caption(
                "Curve normalized to 100 at the start of the lookback window. "
                "Source: Yahoo Finance (Adj Close)."
            )

            # -------- Holdings table --------
            st.markdown("### Holdings, Weights & Risk")

            if top_holdings.empty:
                st.info("No holdings found for this Wave.")
            else:
                # Attach today % move for those tickers
                th = top_holdings.copy()
                th["Weight %"] = th["weight"] * 100.0
                th["Today %"] = th["ticker"].map(today_move) * 100.0

                pretty = pd.DataFrame(
                    {
                        "Ticker": th["ticker"].values,
                        "Weight %": th["Weight %"].values,
                        "Today %": th["Today %"].values,
                    }
                )

                styler = style_top_holdings(pretty)
                st.dataframe(styler, use_container_width=True)

                st.caption(
                    "Top 10 positions by target weight. "
                    "Today % is the latest daily change based on Yahoo Finance."
                )

        except Exception as e:
            st.error(
                "There was a problem running the engine view for this Wave.\n\n"
                f"Details (safe message): **{e}**"
            )

st.markdown("---")
st.caption(
    "WAVES Institutional Console — demo view only. Returns & metrics are based on "
    "public market data via Yahoo Finance and do **not** represent live trading or "
    "an offer of advisory services."
)
