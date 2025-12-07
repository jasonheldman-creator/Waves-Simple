import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="🌊",
    layout="wide",
)

# Small CSS touch-ups to get closer to a dark "Bloomberg-ish" look
st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background-color: #050711;
    }
    /* Remove top padding a bit */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #0f172a;
        padding: 1rem 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
    }
    div[data-testid="metric-container"] > div > div {
        color: #e5e7eb;
    }
    /* Section titles */
    h1, h2, h3, h4 {
        color: #f9fafb;
    }
    .waves-subtitle {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    /* Tables */
    .dataframe tbody tr:hover {
        background-color: #111827 !important;
    }
    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CONFIG – WAVE LABELS & BENCHMARK
# -----------------------------------------------------------------------------
# Under-the-hood IDs should match the "Wave" values in wave_weights.csv
WAVE_LABELS: Dict[str, str] = {
    "AI_Wave": "AI Leaders Wave",
    "Growth_Wave": "Growth Wave",
    "Income_Wave": "Income Wave",
    "SmallCap_Growth_Wave": "Small Cap Growth Wave",
    "SMID_Growth_Wave": "SMID Growth Wave",
    "Future_Power_Energy_Wave": "Future Power & Energy Wave",
    "Quantum_Compute_Wave": "Quantum Computing Wave",
    "Clean_Transit_Infra_Wave": "Clean Transit & Infrastructure Wave",
    "SP500_Wave": "S&P 500 Core Wave",
}

BENCH_TICKER = "SPY"  # Benchmark for alpha

# -----------------------------------------------------------------------------
# DATA LOADERS (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_wave_weights(path: str = "wave_weights.csv") -> pd.DataFrame:
    """
    Load the master weights universe.
    Expected columns (flexible):
        - 'Ticker'
        - either 'Weight %' or 'Weight' (0–1)
        - optional 'Wave'
    """
    df = pd.read_csv(path)

    # Normalize column names a bit
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Figure out weight column
    weight_col = None
    for cand in ["Weight %", "Weight_pct", "Weight%"]:
        if cand in df.columns:
            weight_col = cand
            df["Weight"] = df[cand] / 100.0
            break
    if weight_col is None:
        # Assume there is a Weight column in decimal form
        if "Weight" not in df.columns:
            raise ValueError(
                "wave_weights.csv must have either 'Weight %' or 'Weight' column."
            )

    # Fill missing wave IDs with a generic value if needed
    if "Wave" not in df.columns:
        df["Wave"] = "AI_Wave"

    # Clean ticker symbols (strip spaces)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    # Ensure Weight column exists in 0-1
    if "Weight" not in df.columns:
        # Fallback: if no dedicated Weight col, treat equally weighted
        df["Weight"] = 1.0 / len(df)

    return df


@st.cache_data(show_spinner=False)
def fetch_price_history(
    tickers: List[str],
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch adjusted close history for a basket of tickers.
    Returns a DataFrame: index=dates, columns=tickers
    """
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance shape handling: if multiple tickers, columns are (ticker, field)
    if isinstance(data.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Adj Close") in data.columns:
                closes[t] = data[(t, "Adj Close")]
            elif (t, "Close") in data.columns:
                closes[t] = data[(t, "Close")]
        prices = pd.DataFrame(closes)
    else:
        # Single ticker case
        if "Adj Close" in data.columns:
            prices = data["Adj Close"].to_frame(tickers[0])
        else:
            prices = data["Close"].to_frame(tickers[0])

    prices = prices.dropna(how="all")
    return prices


@st.cache_data(show_spinner=False)
def fetch_latest_quotes(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch last two daily prices for tickers to compute today's % change.
    Returns DataFrame with columns: ['last', 'prev', 'pct_change']
    """
    if not tickers:
        return pd.DataFrame()

    hist = fetch_price_history(tickers, period="5d", interval="1d")
    if hist.empty or len(hist) < 2:
        return pd.DataFrame()

    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    pct_change = (last / prev - 1.0) * 100.0

    quotes = pd.DataFrame(
        {
            "last": last,
            "prev": prev,
            "pct_change": pct_change,
        }
    )
    quotes.index.name = "Ticker"
    return quotes


# -----------------------------------------------------------------------------
# ANALYTICS
# -----------------------------------------------------------------------------
def build_portfolio_curve(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Compute normalized portfolio equity curve given price history and weights.
    prices: DataFrame[dates, tickers]
    weights: Series[ticker] in 0-1 that sum ~1
    """
    if prices.empty:
        return pd.Series(dtype=float)

    # Align weights to available columns
    weights = weights.reindex(prices.columns).fillna(0.0)
    if weights.sum() == 0:
        return pd.Series(dtype=float)
    weights = weights / weights.sum()

    # Normalize prices and aggregate
    norm = prices / prices.iloc[0]
    portfolio = (norm * weights).sum(axis=1)
    return portfolio


def compute_metrics(
    portfolio: pd.Series, bench: pd.Series
) -> Dict[str, float | None]:
    """
    Compute total return, today, max drawdown, alpha vs benchmark.
    Returns values in decimal (0.05 = 5%).
    """
    if portfolio.empty:
        return {
            "total": None,
            "today": None,
            "max_dd": None,
            "alpha": None,
        }

    # Total return
    total = portfolio.iloc[-1] / portfolio.iloc[0] - 1.0

    # Today's change vs previous day
    if len(portfolio) >= 2:
        today = portfolio.iloc[-1] / portfolio.iloc[-2] - 1.0
    else:
        today = None

    # Max drawdown
    cummax = portfolio.cummax()
    drawdown = portfolio / cummax - 1.0
    max_dd = drawdown.min()

    # Alpha vs benchmark (same dates)
    alpha = None
    if bench is not None and not bench.empty:
        aligned = pd.concat([portfolio, bench], axis=1, join="inner").dropna()
        if not aligned.empty:
            p = aligned.iloc[:, 0]
            b = aligned.iloc[:, 1]
            alpha = (p.iloc[-1] / p.iloc[0] - 1.0) - (b.iloc[-1] / b.iloc[0] - 1.0)

    return {
        "total": total,
        "today": today,
        "max_dd": max_dd,
        "alpha": alpha,
    }


def pct_to_str(x: float | None) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x * 100:,.2f}%"


# -----------------------------------------------------------------------------
# HOLDINGS TABLE – RED / GREEN STYLING + GOOGLE FINANCE LINKS
# -----------------------------------------------------------------------------
def google_finance_link(ticker: str) -> str:
    # This generic link works across exchanges and looks nice on mobile
    safe = ticker.upper().strip()
    return f"https://www.google.com/finance/quote/{safe}:NASDAQ"


def build_holdings_table(
    wave_df: pd.DataFrame, quotes: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a top 10 holdings table with weights and today's % change.
    """
    if wave_df.empty:
        return pd.DataFrame()

    df = wave_df.copy()
    df = df.sort_values("Weight", ascending=False).head(10)
    df["Weight %"] = df["Weight"] * 100.0

    # Join on quotes for pct_change
    if not quotes.empty:
        df = df.merge(
            quotes[["pct_change"]],
            how="left",
            left_on="Ticker",
            right_index=True,
        )
    else:
        df["pct_change"] = np.nan

    # Build Google links (Streamlit will render markdown)
    df["Ticker"] = df["Ticker"].apply(
        lambda t: f"[{t}]({google_finance_link(t)})"
    )

    pretty = df[["Ticker", "Weight %", "pct_change"]].rename(
        columns={
            "Ticker": "Ticker (Google Finance)",
            "Weight %": "Weight %",
            "pct_change": "Today %",
        }
    )
    return pretty


def style_holdings(df: pd.DataFrame):
    """
    Apply Bloomberg-style red/green formatting to Today % column.
    """
    def color_change(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #22c55e; font-weight: 600;"  # green
        elif val < 0:
            return "color: #ef4444; font-weight: 600;"  # red
        else:
            return "color: #e5e7eb;"

    styler = (
        df.style
        .format(
            {
                "Weight %": "{:,.2f}%",
                "Today %": "{:,.2f}%",
            },
            na_rep="—",
        )
        .applymap(color_change, subset=["Today %"])
    )
    return styler


# -----------------------------------------------------------------------------
# SIDEBAR – CONTROLS
# -----------------------------------------------------------------------------
weights_df = load_wave_weights()

available_waves = sorted(weights_df["Wave"].unique())

def friendly_wave_name(wave_id: str) -> str:
    if wave_id in WAVE_LABELS:
        return WAVE_LABELS[wave_id]
    # Fallback: make it readable
    return wave_id.replace("_", " ")

with st.sidebar:
    st.markdown("### 🌊 WAVES Intelligence™", unsafe_allow_html=True)
    st.caption("Desktop engine + cloud snapshot")

    st.markdown("---")
    selected_wave_id = st.selectbox(
        "Select Wave",
        options=available_waves,
        format_func=friendly_wave_name,
    )

    st.markdown("**Risk Mode** (label only)")
    risk_mode = st.radio(
        "",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    exposure = st.slider(
        "Equity Exposure",
        min_value=0,
        max_value=100,
        value=90,
        step=5,
        help="Visual only – engine keeps 10% cash buffer in Standard mode.",
    )

    st.markdown("---")
    st.caption(
        "Tip: keep this console open while the engine logic runs in the background "
        "(on cloud we fetch prices live via Yahoo / yfinance)."
    )

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
wave_friendly = friendly_wave_name(selected_wave_id)

st.markdown(
    f"""
    # WAVES Institutional Console  
    <span class="waves-subtitle">
    Live / demo console for <b>WAVES Intelligence™</b> — showing <b>{wave_friendly}</b>.
    </span>
    """,
    unsafe_allow_html=True,
)

# Banner row – SPX + VIX snapshot for feel (no need for extreme precision)
try:
    bench_quotes = fetch_latest_quotes([BENCH_TICKER, "^VIX"])
    spx_row = bench_quotes.loc[BENCH_TICKER] if BENCH_TICKER in bench_quotes.index else None
    vix_row = bench_quotes.loc["^VIX"] if "^VIX" in bench_quotes.index else None
except Exception:
    spx_row = None
    vix_row = None

cols_banner = st.columns(2)
with cols_banner[0]:
    if spx_row is not None:
        spx_change = spx_row["pct_change"]
        color = "🟢" if spx_change > 0 else "🔴" if spx_change < 0 else "⚪️"
        st.markdown(
            f"**SPY (Benchmark)** {color}  "
            f"{spx_change:+.2f}%",
        )
    else:
        st.markdown("**SPY (Benchmark)** —")

with cols_banner[1]:
    if vix_row is not None:
        vix_change = vix_row["pct_change"]
        color = "🟢" if vix_change > 0 else "🔴" if vix_change < 0 else "⚪️"
        st.markdown(
            f"**VIX** {color}  "
            f"{vix_change:+.2f}%",
        )
    else:
        st.markdown("**VIX** —")

st.markdown("---")

# -----------------------------------------------------------------------------
# SELECTED WAVE DATA
# -----------------------------------------------------------------------------
wave_positions = weights_df[weights_df["Wave"] == selected_wave_id].copy()
tickers = sorted(wave_positions["Ticker"].unique())

# Fetch histories
prices = fetch_price_history(tickers, period="6mo", interval="1d")
bench_prices = fetch_price_history([BENCH_TICKER], period="6mo", interval="1d")
bench_series = (
    bench_prices[BENCH_TICKER]
    if not bench_prices.empty and BENCH_TICKER in bench_prices.columns
    else pd.Series(dtype=float)
)

portfolio_curve = build_portfolio_curve(
    prices,
    wave_positions.set_index("Ticker")["Weight"],
)
metrics = compute_metrics(portfolio_curve, bench_series)

# -----------------------------------------------------------------------------
# METRIC CARDS ROW
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Return (6m lookback)", pct_to_str(metrics["total"]))
with m2:
    st.metric("Today", pct_to_str(metrics["today"]))
with m3:
    st.metric("Max Drawdown", pct_to_str(metrics["max_dd"]))
with m4:
    st.metric("Alpha vs SPY", pct_to_str(metrics["alpha"]))

st.markdown("")

# -----------------------------------------------------------------------------
# TWO-PANEL LAYOUT: PERFORMANCE CURVE + HOLDINGS
# -----------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Performance Curve")

    if portfolio_curve.empty:
        st.info(
            "No price history available for this Wave yet. "
            "Once market data is available, this chart will plot the equity curve."
        )
    else:
        chart_df = pd.DataFrame(
            {
                wave_friendly: portfolio_curve / portfolio_curve.iloc[0] * 100.0,
            }
        )

        if not bench_series.empty:
            aligned = bench_series.reindex(chart_df.index).dropna()
            if not aligned.empty:
                chart_df["SPY"] = aligned / aligned.iloc[0] * 100.0

        st.line_chart(chart_df)

        st.caption(
            "Curve is normalized to 100 at start of lookback window. "
            "Source: yfinance (Adj Close)."
        )

with right:
    st.subheader("Holdings, Weights & Risk")

    latest_quotes = fetch_latest_quotes(tickers)
    holdings_table = build_holdings_table(wave_positions, latest_quotes)

    if holdings_table.empty:
        st.info("No holdings found for this Wave in wave_weights.csv.")
    else:
        st.markdown("**Top 10 Positions — Google Finance Links (Bloomberg-style)**")
        styled = style_holdings(holdings_table)
        st.dataframe(
            styled,
            use_container_width=True,
            height=320,
        )

    with st.expander("Full Wave universe table"):
        full_table = wave_positions.copy()
        full_table["Weight %"] = full_table["Weight"] * 100.0
        full_pretty = full_table[["Ticker", "Weight %"]].sort_values(
            "Weight %", ascending=False
        )
        st.dataframe(
            full_pretty.style.format({"Weight %": "{:,.2f}%"}),
            use_container_width=True,
            height=360,
        )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "WAVES Institutional Console — demo view only. "
    "Returns & metrics are based on public market data via yfinance and "
    "do not represent live trading or an offer of advisory services."
)
