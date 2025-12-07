# app.py  — WAVES Institutional Console (Cloud + Desktop)

import datetime as dt
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_weights(csv_path: str = "wave_weights.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize columns a bit; support several possible schemas
    cols = {c.lower(): c for c in df.columns}

    # Ticker column
    if "ticker" in cols:
        ticker_col = cols["ticker"]
    else:
        raise ValueError("wave_weights.csv must have a 'Ticker' column")

    # Weight column (supports 'weight', 'weight %', etc)
    if "weight" in cols:
        weight_col = cols["weight"]
    elif "weight %" in cols:
        weight_col = cols["weight %"]
    else:
        raise ValueError("wave_weights.csv must have a 'Weight' or 'Weight %' column")

    # Wave id column
    wave_candidates = [
        "wave",
        "wave_name",
        "portfolio",
    ]
    wave_col = None
    for cand in wave_candidates:
        if cand in cols:
            wave_col = cols[cand]
            break

    if wave_col is None:
        raise ValueError(
            "wave_weights.csv must have a 'Wave' (or 'Wave_Name' / 'Portfolio') column"
        )

    df = df[[ticker_col, weight_col, wave_col]].copy()
    df.columns = ["Ticker", "Weight_raw", "Wave"]

    # If weights look like 0–1, keep; if they look like percents, scale
    if df["Weight_raw"].max() <= 1.0 + 1e-6:
        df["Weight"] = df["Weight_raw"].astype(float)
    else:
        df["Weight"] = (df["Weight_raw"].astype(float) / 100.0)

    # Uppercase tickers
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # Strip spaces from Wave ids
    df["Wave"] = df["Wave"].astype(str).str.strip()

    return df


def pretty_wave_name(wave_id: str) -> str:
    """Turn internal Wave IDs into nice display names."""
    overrides = {
        "AI_Wave": "AI Leaders Wave",
        "Growth_Wave": "Growth Wave",
        "Income_Wave": "Income Wave",
        "SmallCap_Growth_Wave": "Small Cap Growth Wave",
        "SMID_Growth_Wave": "Small–Mid Growth Wave",
        "Future_Energy_Wave": "Future Power & Energy Wave",
        "Crypto_Income_Wave": "Crypto Income Wave",
        "Quantum_Wave": "Quantum Computing Wave",
        "Clean_Transit_Wave": "Clean Transit & Infrastructure Wave",
    }

    if wave_id in overrides:
        return overrides[wave_id]

    # Fallback: AI_Wave -> "AI Wave", etc.
    base = wave_id.replace("_", " ").strip()
    if "wave" not in base.lower():
        base = base + " Wave"
    return base


@st.cache_data(show_spinner=False)
def fetch_prices(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch price data for given tickers via yfinance.
    Returns a DataFrame of closes (one column per ticker).
    """
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=list(set(tickers)),
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # yfinance returns:
    # - MultiIndex columns when multiple tickers
    # - Single Index when one ticker
    if isinstance(data.columns, pd.MultiIndex):
        closes = data.xs("Close", axis=1, level=1)
    else:
        closes = data[["Close"]] if "Close" in data.columns else data

    closes = closes.sort_index()
    return closes


def compute_portfolio_series(
    prices: pd.DataFrame, weights: Dict[str, float]
) -> pd.Series:
    """
    Compute portfolio index series (normalized to 100) from
    price history and a dict of ticker -> weight (summing to 1).
    """
    if prices.empty or not weights:
        return pd.Series(dtype=float)

    aligned = prices[list(weights.keys())].dropna(how="all")
    if aligned.empty:
        return pd.Series(dtype=float)

    w_vec = np.array([weights[t] for t in aligned.columns])
    rets = aligned.pct_change().dropna()
    port_rets = (rets * w_vec).sum(axis=1)
    port_idx = (1 + port_rets).cumprod() * 100.0
    return port_idx


def compute_metrics(series: pd.Series, bench: pd.Series) -> Dict[str, float]:
    if series.empty or bench.empty:
        return {
            "total_return": np.nan,
            "today_return": np.nan,
            "max_drawdown": np.nan,
            "alpha_vs_bench": np.nan,
        }

    series = series.dropna()
    bench = bench.reindex(series.index).dropna()
    if len(series) < 5 or len(bench) < 5:
        return {
            "total_return": np.nan,
            "today_return": np.nan,
            "max_drawdown": np.nan,
            "alpha_vs_bench": np.nan,
        }

    total_ret = series.iloc[-1] / series.iloc[0] - 1.0
    today_ret = series.pct_change().iloc[-1]

    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    max_dd = dd.min()

    bench_ret = bench.iloc[-1] / bench.iloc[0] - 1.0
    alpha = total_ret - bench_ret

    return {
        "total_return": float(total_ret),
        "today_return": float(today_ret),
        "max_drawdown": float(max_dd),
        "alpha_vs_bench": float(alpha),
    }


def format_pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:,.{digits}f}%"


def format_change_cell(x: float) -> str:
    if pd.isna(x):
        return ""
    color = "#00C853" if x >= 0 else "#FF5252"
    return f"<span style='color:{color}; font-weight:600;'>{x:+.2f}%</span>"


def make_gf_link(ticker: str) -> str:
    url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
    return (
        f"<a href='{url}' target='_blank' "
        f"style='text-decoration:none; color:#4DA3FF; font-weight:600;'>{ticker}</a>"
    )


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    weights_df = load_weights()

    # ----------------------------------------------------------------
    # Sidebar – Engine controls
    # ----------------------------------------------------------------
    st.sidebar.markdown(
        "<h3 style='color:#FFFFFF;'>⚡ WAVES Intelligence™</h3>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("DESKTOP ENGINE + CLOUD SNAPSHOT")

    st.sidebar.write(
        "Institutional console for **WAVES Intelligence™** — select one of your "
        "locked Waves below."
    )

    wave_ids = sorted(weights_df["Wave"].unique())
    default_wave = wave_ids[0] if wave_ids else None

    selected_wave_id = st.sidebar.selectbox(
        "Select Wave",
        options=wave_ids,
        index=0,
        format_func=pretty_wave_name,
    )

    st.sidebar.markdown("### Risk Mode (label only)")
    st.sidebar.radio(
        "Risk Mode",
        options=["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
        label_visibility="collapsed",
    )

    eq_exposure = st.sidebar.slider(
        "Equity Exposure (target)",
        min_value=0,
        max_value=100,
        value=90,
        step=5,
    )

    st.sidebar.caption(
        "Keep this console open while the desktop engine runs in the background, "
        "or let the cloud logic fetch prices live via Yahoo Finance."
    )

    # ----------------------------------------------------------------
    # Data prep for selected wave
    # ----------------------------------------------------------------
    wave_holdings = (
        weights_df[weights_df["Wave"] == selected_wave_id]
        .copy()
        .reset_index(drop=True)
    )

    # Normalize weights to 1.0
    if not wave_holdings.empty:
        wave_holdings["Weight"] = wave_holdings["Weight"] / wave_holdings["Weight"].sum()
    tickers = wave_holdings["Ticker"].tolist()

    # Always fetch SPY and VIX as well
    bench_ticker = "SPY"
    vix_ticker = "^VIX"
    all_tickers = tickers + [bench_ticker, vix_ticker]

    prices = fetch_prices(all_tickers, period="1y", interval="1d")

    # Split out
    bench_series = prices[bench_ticker] if bench_ticker in prices.columns else pd.Series()
    vix_series = prices[vix_ticker] if vix_ticker in prices.columns else pd.Series()
    wave_price_slice = prices[tickers] if tickers else pd.DataFrame()

    weight_dict = dict(zip(wave_holdings["Ticker"], wave_holdings["Weight"]))
    portfolio_series = compute_portfolio_series(wave_price_slice, weight_dict)

    if not bench_series.empty:
        bench_index = bench_series / bench_series.iloc[0] * 100.0
    else:
        bench_index = pd.Series(dtype=float)

    metrics = compute_metrics(portfolio_series, bench_index)

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    display_wave_name = pretty_wave_name(selected_wave_id)
    today = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    header_col1, header_col2 = st.columns([3, 2])

    with header_col1:
        st.markdown(
            f"""
            <h1 style="margin-bottom:0.25rem;">WAVES Institutional Console</h1>
            <p style="color:#BBBBBB; margin-top:0;">
                Live / demo console for <b>WAVES Intelligence™</b> — showing
                <b>{display_wave_name}</b>.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with header_col2:
        # SPY + VIX mini metrics
        spy_ret = np.nan
        if not bench_index.empty:
            spy_ret = bench_index.iloc[-1] / bench_index.iloc[0] - 1.0

        vix_last = vix_series.iloc[-1] if not vix_series.empty else np.nan

        m1, m2, m3 = st.columns(3)
        m1.metric("SPY (Benchmark)", format_pct(spy_ret), None)
        m2.metric("VIX", f"{vix_last:,.2f}" if not pd.isna(vix_last) else "—", None)
        m3.caption(f"Snapshot as of {today}")

    st.markdown("---")

    # ----------------------------------------------------------------
    # Top KPI row
    # ----------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total Return (lookback)", format_pct(metrics["total_return"]))
    kpi2.metric("Today", format_pct(metrics["today_return"]))
    kpi3.metric("Max Drawdown", format_pct(metrics["max_drawdown"]))
    kpi4.metric("Alpha vs SPY", format_pct(metrics["alpha_vs_bench"]))

    # ----------------------------------------------------------------
    # Performance & holdings layout
    # ----------------------------------------------------------------
    left, right = st.columns([3, 2])

    # ----- Left: Performance curve + SPY + mini charts
    with left:
        st.subheader("Performance Curve")

        if portfolio_series.empty or bench_index.empty:
            st.info(
                "No sufficient price history found for this Wave yet. "
                "Once prices are available, the equity curve will appear here."
            )
        else:
            chart_df = pd.DataFrame(
                {
                    display_wave_name: portfolio_series,
                    "SPY": bench_index,
                }
            )
            st.line_chart(chart_df)

            st.caption(
                "Curve is normalized to 100 at the start of the lookback window. "
                "Source: Yahoo Finance (Adj Close)."
            )

        # Mini SPX + VIX charts
        mini1, mini2 = st.columns(2)
        with mini1:
            st.caption("S&P 500 (SPY) — mini view")
            if not bench_index.empty:
                st.line_chart(bench_index.rename("SPY"))
            else:
                st.write("No SPY data available.")

        with mini2:
            st.caption("VIX — mini view")
            if not vix_series.empty:
                st.line_chart(vix_series.rename("VIX"))
            else:
                st.write("No VIX data available.")

    # ----- Right: Holdings, weights, risk
    with right:
        st.subheader("Holdings, Weights & Risk")

        if wave_holdings.empty:
            st.write("No holdings found for this Wave.")
        else:
            # Today's percent change per ticker
            today_changes = {}
            if not wave_price_slice.empty and len(wave_price_slice) > 1:
                pct = wave_price_slice.pct_change().iloc[-1]
                today_changes = pct.to_dict()

            top10 = (
                wave_holdings.sort_values("Weight", ascending=False)
                .head(10)
                .copy()
                .reset_index(drop=True)
            )

            top10["Weight %"] = top10["Weight"] * 100.0
            top10["Today %"] = [
                today_changes.get(t, np.nan) * 100.0 for t in top10["Ticker"]
            ]

            # Build HTML table with Google Finance links + colored today %
            df_display = pd.DataFrame(
                {
                    "Ticker": [make_gf_link(t) for t in top10["Ticker"]],
                    "Weight %": [f"{w:.2f}%" for w in top10["Weight %"]],
                    "Today %": [format_change_cell(x) for x in top10["Today %"]],
                }
            )

            st.markdown(
                "Top 10 Positions — "
                "<span style='color:#AAAAAA;'>Google Finance Links (Bloomberg-style)</span>",
                unsafe_allow_html=True,
            )

            st.markdown(
                df_display.to_html(escape=False, index=False),
                unsafe_allow_html=True,
            )

            with st.expander("Full Wave universe table"):
                full_df = wave_holdings.copy()
                full_df["Weight %"] = full_df["Weight"] * 100.0
                full_df = full_df[["Ticker", "Weight %"]]
                st.dataframe(full_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown(
            f"""
            **Target equity exposure:** {eq_exposure}%  
            **Cash buffer:** {100 - eq_exposure}% (SmartSafe™ placeholder)  
            **Risk mode:** Standard (for demo; actual engine logic lives in desktop)
            """
        )

    # ----------------------------------------------------------------
    # Footer / disclaimer
    # ----------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "WAVES Institutional Console — demo view only. Returns & metrics are based on "
        "public market data via Yahoo Finance and do not represent live trading or an offer "
        "of advisory services."
    )


if __name__ == "__main__":
    main()
