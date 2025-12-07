import os
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
    page_icon="🌊",
)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_weights(csv_path: str = "wave_weights.csv") -> pd.DataFrame:
    """Load wave universe / weights."""
    df = pd.read_csv(csv_path)
    # Try to normalize expected column names
    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker", list(df.columns)[0])
    wave_col = cols.get("wave", list(df.columns)[1])
    # allow either 'Weight %' or 'WeightPct' etc
    weight_col = None
    for c in df.columns:
        if "weight" in c.lower():
            weight_col = c
            break
    if weight_col is None:
        raise ValueError("No weight column found in wave_weights.csv")

    df = df[[ticker_col, wave_col, weight_col]].copy()
    df.columns = ["Ticker", "Wave", "WeightPct"]
    return df


def friendly_wave_name(raw: str) -> str:
    mapping = {
        "AI_Wave": "AI Leaders Wave",
        "Growth_Wave": "Growth Wave",
        "SP500_Wave": "S&P 500 Wave",
    }
    if raw in mapping:
        return mapping[raw]
    # Fallback: replace underscores and title-case
    return raw.replace("_", " ").strip().title()


def google_finance_link(ticker: str) -> str:
    # Generic search works well regardless of exchange
    return f"https://www.google.com/finance?q={ticker}"


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: List[str], lookback_days: int = 90) -> pd.DataFrame:
    """Download daily history for a list of tickers."""
    if not tickers:
        return pd.DataFrame()

    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days * 2)  # buffer for weekends/holidays

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end + dt.timedelta(days=1),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    # yfinance returns either:
    # - DataFrame with 'Adj Close' column (single ticker)
    # - MultiIndex columns ('Adj Close', 'TICKER') (multi-ticker)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Adj Close"].copy()
    else:
        close = data[["Adj Close"]].copy()
        close.columns = tickers

    close = close.dropna(how="all")
    return close


def build_portfolio_curve(
    prices: pd.DataFrame, weights: pd.Series
) -> Tuple[pd.Series, Dict[str, float]]:
    """Create an equity curve and summary stats from prices & weights."""
    if prices.empty or weights.empty:
        return pd.Series(dtype=float), {}

    # Align weights to available tickers; normalize to 1
    weights = weights.reindex(prices.columns).fillna(0.0)
    if weights.sum() == 0:
        return pd.Series(dtype=float), {}

    w = weights / weights.sum()

    # Normalize each ticker to 1.0 at first available price
    norm_prices = prices / prices.iloc[0]
    portfolio = (norm_prices * w.values).sum(axis=1)

    # Convert to cumulative return (%)
    cumret = portfolio / portfolio.iloc[0] - 1.0

    # Stats
    daily_ret = portfolio.pct_change().dropna()
    total_return = cumret.iloc[-1]
    today_ret = daily_ret.iloc[-1] if not daily_ret.empty else 0.0

    running_max = portfolio.cummax()
    drawdown = portfolio / running_max - 1.0
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    stats = {
        "total_return": float(total_return),
        "today_return": float(today_ret),
        "max_drawdown": float(max_drawdown),
    }
    return cumret, stats


@st.cache_data(show_spinner=False)
def fetch_ticker_moves(tickers: List[str]) -> pd.Series:
    """Get today's % move for tickers (for red/green in holdings table)."""
    if not tickers:
        return pd.Series(dtype=float)

    end = dt.date.today()
    start = end - dt.timedelta(days=7)

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end + dt.timedelta(days=1),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.Series(index=tickers, dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Adj Close"]
    else:
        close = data[["Adj Close"]]
        close.columns = tickers

    close = close.dropna(how="all")
    if close.shape[0] < 2:
        return pd.Series(index=tickers, dtype=float)

    last = close.iloc[-1]
    prev = close.iloc[-2]
    moves = (last / prev - 1.0) * 100.0
    return moves.reindex(tickers).round(2)


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

weights_df = load_weights()
wave_ids = sorted(weights_df["Wave"].unique())
wave_display_map = {raw: friendly_wave_name(raw) for raw in wave_ids}
display_to_raw = {v: k for k, v in wave_display_map.items()}

# ---------------------------------------------------------
# SIDEBAR – ENGINE + WAVE SELECTION
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("### 🌊 WAVES Intelligence™")
    st.markdown("**DESKTOP ENGINE + CLOUD SNAPSHOT**")

    st.caption(
        "Institutional console for WAVES Intelligence™ — select one of your "
        "locked Waves below. This version computes performance live in the cloud "
        "using current price history (no local CSV logs needed)."
    )

    selected_wave_display = st.selectbox(
        "Select Wave",
        options=list(display_to_raw.keys()),
        index=0,
    )
    selected_wave_id = display_to_raw[selected_wave_display]

    st.markdown("---")
    st.markdown("#### Risk Mode (label only)")
    mode = st.radio(
        "Risk Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
        label_visibility="collapsed",
    )
    st.caption(f"**{mode.upper()}**")

    st.markdown("---")
    st.caption(
        "This is a **cloud-only** engine: it pulls prices via yfinance on each "
        "refresh. No local terminal or CSV logs are required."
    )

# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------

st.markdown(
    "<h1 style='margin-bottom:0'>WAVES Institutional Console</h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Live / demo console for WAVES Intelligence™ — showing **{selected_wave_display}**."
)

col_spx, col_vix, col_mode = st.columns([2, 2, 1])
with col_spx:
    st.markdown("**SPX** 6899.07 *(demo)*")
with col_vix:
    st.markdown("**VIX** 15.40 *(demo)*")
with col_mode:
    st.markdown(f"**Mode:** {mode}")

st.markdown("---")

# ---------------------------------------------------------
# BUILD WAVE VIEW
# ---------------------------------------------------------

wave_slice = weights_df[weights_df["Wave"] == selected_wave_id].copy()
wave_slice = wave_slice.sort_values("WeightPct", ascending=False)

tickers = wave_slice["Ticker"].tolist()
weights_pct = wave_slice["WeightPct"]
weights_normalized = weights_pct / weights_pct.sum()

prices = fetch_prices(tickers)
curve, stats = build_portfolio_curve(prices, weights_normalized)

# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

if curve.empty:
    tr, td, mdd = 0.0, 0.0, 0.0
else:
    tr = stats["total_return"] * 100.0
    td = stats["today_return"] * 100.0
    mdd = stats["max_drawdown"] * 100.0

with m1:
    st.metric("Total Return (live)", f"{tr:+.2f}%")
with m2:
    st.metric("Today", f"{td:+.2f}%")
with m3:
    st.metric("Max Drawdown", f"{mdd:.2f}%")
with m4:
    st.metric("Alpha vs Benchmark", "—")  # placeholder until benchmark logic is added

st.markdown("---")

# ---------------------------------------------------------
# PERFORMANCE CURVE & HOLDINGS
# ---------------------------------------------------------

left, right = st.columns([2, 2])

with left:
    st.subheader("Performance Curve")
    if curve.empty:
        st.info(
            "No performance history could be built yet for this Wave. "
            "This usually means price data could not be fetched for its tickers "
            "or the Wave has no holdings defined."
        )
    else:
        curve_df = pd.DataFrame({"Cumulative Return": (curve * 100.0).round(2)})
        curve_df.index.name = "Date"
        st.line_chart(curve_df)

with right:
    st.subheader("Holdings, Weights & Risk")

    if wave_slice.empty:
        st.warning("No holdings found for this Wave in wave_weights.csv.")
    else:
        # Top 10 holdings
        top10 = wave_slice.head(10).copy()
        top10_moves = fetch_ticker_moves(top10["Ticker"].tolist())

        top10["Today %"] = top10["Ticker"].map(top10_moves).fillna(0.0)

        # Build a styled HTML table for Bloomberg-style red/green
        table_rows = []
        for _, row in top10.iterrows():
            t = row["Ticker"]
            w = row["WeightPct"]
            move = row["Today %"]

            color = "green" if move > 0 else "red" if move < 0 else "gray"
            move_html = f"<span style='color:{color}'>{fmt_pct(move)}</span>"

            link = google_finance_link(t)
            cell_ticker = f"<a href='{link}' target='_blank'>{t}</a>"

            table_rows.append(
                f"<tr>"
                f"<td style='padding:4px 8px'>{cell_ticker}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{w:.2f}%</td>"
                f"<td style='padding:4px 8px;text-align:right'>{move_html}</td>"
                f"</tr>"
            )

        table_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:0.9rem'>"
            "<thead>"
            "<tr>"
            "<th style='text-align:left;padding:4px 8px'>Ticker</th>"
            "<th style='text-align:right;padding:4px 8px'>Weight %</th>"
            "<th style='text-align:right;padding:4px 8px'>Today</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            + "".join(table_rows)
            + "</tbody></table>"
        )

        st.markdown("**Top 10 Positions — Google Finance Links (red/green moves)**")
        st.markdown(table_html, unsafe_allow_html=True)

        with st.expander("Full Wave universe table"):
            full_moves = fetch_ticker_moves(wave_slice["Ticker"].tolist())
            wave_slice["Today %"] = wave_slice["Ticker"].map(full_moves).fillna(0.0)
            st.dataframe(
                wave_slice[["Ticker", "WeightPct", "Today %"]]
                .rename(columns={"WeightPct": "Weight %"})
                .reset_index(drop=True)
            )

st.markdown("---")
st.caption(
    "This console is running entirely in the cloud. Performance curves and holdings "
    "are computed live from price history; no local CSV logs or background engine "
    "process is required."
)
