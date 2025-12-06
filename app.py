# app.py — WAVES Intelligence™ Console
# - Dedupes tickers per Wave
# - Upgraded header (logo-style)
# - Right panel: S&P 500 (SPY) + VIX charts

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

# -------------------------------
# CONFIG
# -------------------------------

UNIVERSE_CSV = "list.csv"
WAVE_WEIGHTS_CSV = "wave_weights.csv"

REFRESH_SECONDS = 60
VIX_SYMBOL = "^VIX"
SPY_SYMBOL = "SPY"
HISTORY_PERIOD = "6mo"
HISTORY_INTERVAL = "1d"

st.set_page_config(
    page_title="WAVES Intelligence Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# GLOBAL STYLE
# -------------------------------

def apply_global_style():
    st.markdown(
        """
        <style>
        .main {
            background: radial-gradient(circle at top left, #141726 0, #05060B 45%, #02030a 100%);
            color: #FFFFFF;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1.4rem;
        }
        .waves-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            margin-bottom: 0.4rem;
        }
        .waves-logo-pill {
            width: 40px;
            height: 40px;
            border-radius: 999px;
            background: conic-gradient(from 160deg, #18ffb2, #25b6ff, #18ffb2);
            box-shadow: 0 0 18px rgba(24,255,178,0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 20px;
            color: #05070f;
        }
        .waves-title-text {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        .waves-title-main {
            font-size: 26px;
            font-weight: 720;
            color: #e9fdfc;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .waves-title-sub {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #8ea0c2;
        }
        .waves-subtitle {
            font-size: 13px;
            text-align: center;
            color: #b3b9cc;
            margin-bottom: 0.8rem;
        }
        .metric-card {
            background: linear-gradient(145deg, #111421, #05070f);
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            border: 1px solid rgba(140, 255, 210, 0.08);
            box-shadow: 0 8px 24px rgba(0,0,0,0.55);
        }
        .metric-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #8e94a8;
        }
        .metric-value {
            font-size: 20px;
            font-weight: 650;
            color: #ffffff;
            margin-top: 0.25rem;
        }
        .section-header {
            font-size: 16px;
            font-weight: 600;
            color: #e2e6ff;
            margin-top: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #05060c 0, #050309 65%, #020309 100%);
            border-right: 1px solid #15192b;
        }
        .sidebar-title {
            color: #e5ebff;
            font-weight: 600;
            font-size: 15px;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# DATA LOADERS
# -------------------------------

@st.cache_data(show_spinner=True)
def load_universe():
    """Load security universe from list.csv and normalize column names."""
    if not os.path.exists(UNIVERSE_CSV):
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' not found."

    try:
        df = pd.read_csv(UNIVERSE_CSV)
    except Exception as e:
        return None, f"[UNIVERSE ERROR] Failed to read '{UNIVERSE_CSV}': {e}"

    if df.empty:
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' is empty."

    col_map = {}
    if "Company" in df.columns:
        col_map["Company"] = "Name"
    if "Weight" in df.columns:
        col_map["Weight"] = "IndexWeight"
    if "Market Value" in df.columns:
        col_map["Market Value"] = "MarketValue"

    df = df.rename(columns=col_map)

    required = ["Ticker", "Name", "Sector", "IndexWeight", "MarketValue", "Price"]
    for col in required:
        if col not in df.columns:
            if col == "Sector":
                df["Sector"] = "Unclassified"
            elif col == "MarketValue":
                df["MarketValue"] = 0.0
            elif col == "Price":
                df["Price"] = 0.0
            else:
                return None, (
                    f"[UNIVERSE ERROR] Missing column '{col}' in '{UNIVERSE_CSV}'. "
                    f"Found columns: {list(df.columns)}"
                )

    df["IndexWeight"] = pd.to_numeric(df["IndexWeight"], errors="coerce").fillna(0.0)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str)

    return df, None


@st.cache_data(show_spinner=True)
def load_wave_weights():
    """
    Load Wave allocations from wave_weights.csv.

    Expected columns:
        Ticker,Wave,Weight

    - Ignores lines starting with '#'
    - Dedupes (Wave, Ticker) pairs by summing Weight
    - Normalizes weights per Wave so they sum to 1.0
    """
    if not os.path.exists(WAVE_WEIGHTS_CSV):
        return None, f"[WAVE ERROR] '{WAVE_WEIGHTS_CSV}' not found."

    try:
        df = pd.read_csv(WAVE_WEIGHTS_CSV, comment="#")
    except Exception as e:
        return None, f"[WAVE ERROR] Failed to read '{WAVE_WEIGHTS_CSV}': {e}"

    if df.empty:
        return None, f"[WAVE ERROR] '{WAVE_WEIGHTS_CSV}' is empty."

    required_cols = {"Ticker", "Wave", "Weight"}
    missing = required_cols - set(df.columns)
    if missing:
        return None, (
            f"[WAVE ERROR] Missing column(s) {missing} in '{WAVE_WEIGHTS_CSV}'. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    df = df[df["Weight"] > 0]
    if df.empty:
        return None, f"[WAVE ERROR] All weights in '{WAVE_WEIGHTS_CSV}' are zero or invalid."

    # Deduplicate within each Wave / Ticker
    df = (
        df.groupby(["Wave", "Ticker"], as_index=False)["Weight"]
        .sum()
    )

    # Normalize per Wave
    totals = df.groupby("Wave")["Weight"].transform("sum").replace(0, 1.0)
    df["WaveWeight"] = df["Weight"] / totals
    df = df.drop(columns=["Weight"])

    return df, None


@st.cache_data(show_spinner=False)
def fetch_live_prices(tickers):
    """Fetch latest prices + intraday change via yfinance."""
    if not tickers:
        return pd.DataFrame()

    data = []
    for t in tickers:
        try:
            yt = yf.Ticker(t)
            hist = yt.history(period="1d")
            if hist.empty:
                continue
            last_row = hist.iloc[-1]
            price = float(last_row["Close"])
            open_price = float(last_row.get("Open", price))
            change = price - open_price
            change_pct = (change / open_price) * 100 if open_price != 0 else 0.0
            data.append(
                {
                    "Ticker": t,
                    "LivePrice": price,
                    "LiveChange": change,
                    "LiveChangePct": change_pct,
                }
            )
        except Exception:
            continue

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data).set_index("Ticker")


@st.cache_data(show_spinner=False)
def fetch_history(symbol: str):
    """Fetch historical close for SPY or VIX."""
    try:
        tkr = yf.Ticker(symbol)
        hist = tkr.history(period=HISTORY_PERIOD, interval=HISTORY_INTERVAL)
        if hist.empty:
            return pd.DataFrame()
        df = hist.reset_index()[["Date", "Close"]]
        df.rename(columns={"Close": "Value"}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------------
# UI HELPERS
# -------------------------------

def render_header(active_wave: str, active_mode: str, equity_exposure: float):
    st.markdown(
        """
        <div class="waves-header">
          <div class="waves-logo-pill">W</div>
          <div class="waves-title-text">
            <div class="waves-title-main">WAVES Intelligence™</div>
            <div class="waves-title-sub">Live Engine Console</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    subtitle = (
        f"Wave: <b>{active_wave}</b> &nbsp;•&nbsp; "
        f"Mode: <b>{active_mode}</b> &nbsp;•&nbsp; "
        "Alpha-Minus-Beta Discipline &nbsp;•&nbsp; Vector-Driven Allocation<br/>"
        f"Equity Exposure: <b>{equity_exposure:.0f}%</b> &nbsp;•&nbsp; "
        f"Cash Buffer: <b>{100 - equity_exposure:.0f}%</b>"
    )
    st.markdown(
        f'<div class="waves-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def show_wave_snapshot(df_wave: pd.DataFrame, equity_exposure: float):
    total_names = len(df_wave)
    top10_weight = (
        df_wave.sort_values("WaveWeight", ascending=False)["WaveWeight"].head(10).sum()
    )
    last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    col1, col2, col3, col4 = st.columns(4)

    for col, label, value in [
        (col1, "Total Holdings", f"{total_names:,}"),
        (col2, "Top 10 Weight", f"{top10_weight*100:0.1f}%"),
        (col3, "Equity Exposure", f"{equity_exposure:.0f}%"),
        (col4, "Last Refresh", last_refresh),
    ]:
        with col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-label">{label}</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="metric-value">{value}</div>', unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)


def show_top_holdings(df_wave: pd.DataFrame, prices_df: pd.DataFrame):
    st.markdown('<div class="section-header">Top 10 Positions</div>', unsafe_allow_html=True)
    try:
        top = df_wave.sort_values("WaveWeight", ascending=False).head(10).copy()
        if not prices_df.empty:
            top = top.merge(prices_df.reset_index(), on="Ticker", how="left")

        top["Weight %"] = (top["WaveWeight"] * 100).round(2)

        display_cols = ["Ticker", "Name", "Sector", "Weight %"]
        if "LivePrice" in top.columns:
            display_cols.append("LivePrice")
        if "LiveChangePct" in top.columns:
            top["LiveChangePct"] = top["LiveChangePct"].round(2)
            display_cols.append("LiveChangePct")

        st.dataframe(
            top[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        chart_df = df_wave.sort_values("WaveWeight", ascending=False).head(10).copy()
        chart_df["Weight %"] = chart_df["WaveWeight"] * 100

        fig = px.bar(
            chart_df,
            x="Weight %",
            y="Ticker",
            orientation="h",
            labels={"Weight %": "Weight (%)", "Ticker": "Ticker"},
        )
        fig.update_layout(
            title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            margin=dict(l=40, r=20, t=10, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Top holdings view unavailable: {e}")


def show_spy_chart(spy_df: pd.DataFrame):
    st.markdown('<div class="section-header">S&P 500 (SPY)</div>', unsafe_allow_html=True)
    if spy_df.empty:
        st.info("SPY data not available right now.")
        return

    fig = px.line(
        spy_df,
        x="Date",
        y="Value",
        labels={"Value": "Price", "Date": ""},
    )
    fig.update_traces(line_width=2)
    fig.update_layout(
        title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        margin=dict(l=40, r=20, t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_vix_chart(vix_df: pd.DataFrame):
    st.markdown('<div class="section-header">Volatility (VIX)</div>', unsafe_allow_html=True)
    if vix_df.empty:
        st.info("VIX data not available right now.")
        return

    fig = px.line(
        vix_df,
        x="Date",
        y="Value",
        labels={"Value": "Index Level", "Date": ""},
    )
    fig.update_traces(line_width=2)
    fig.update_layout(
        title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        margin=dict(l=40, r=20, t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_alpha_placeholder():
    st.markdown('<div class="section-header">Alpha Capture (Preview)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="color:#b3b9cc; font-size:13px;">
        This terminal is currently wired to the <b>universe &amp; exposure</b> layer.
        Live alpha capture (WaveScore™, benchmark-relative returns, stress periods)
        runs in the WAVES performance engine, which can be plugged into this console
        as the next step for Franklin or any institutional partner.
        </p>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# MAIN APP
# -------------------------------

def main():
    apply_global_style()

    universe_df, uni_err = load_universe()
    wave_weights_df, wave_err = load_wave_weights()

    if uni_err:
        st.error(uni_err)
        return
    if wave_err:
        st.error(wave_err)
        return

    # Wave list
    waves = sorted(wave_weights_df["Wave"].dropna().unique().tolist())
    if "SP500_Wave" not in waves:
        waves.insert(0, "SP500_Wave")
    else:
        waves.remove("SP500_Wave")
        waves.insert(0, "SP500_Wave")

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Engine Controls</div>', unsafe_allow_html=True)

        active_wave = st.selectbox("Wave", options=waves, index=0)

        mode = st.radio(
            "Wave Mode",
            options=["Standard", "Alpha-Minus-Beta", "Private Logic"],
            index=0,
        )

        equity_exposure = st.slider(
            "Equity Exposure (%)",
            min_value=0,
            max_value=100,
            value=90,
            step=1,
            help="Represents current net equity vs SmartSafe™ / cash.",
        )

        auto_refresh = st.checkbox("Auto-refresh console", value=True)
        st.caption(f"Console will rerun every {REFRESH_SECONDS} seconds when enabled.")

    # ----- Build active Wave dataframe -----
    if active_wave == "SP500_Wave":
        df_wave = universe_df.copy()

        # In case of any ticker duplication in the universe, collapse it
        df_wave = (
            df_wave.groupby("Ticker", as_index=False)
            .agg({
                "Name": "first",
                "Sector": "first",
                "IndexWeight": "sum",
                "MarketValue": "sum",
                "Price": "first",
            })
        )

        total_index_weight = df_wave["IndexWeight"].sum()
        if total_index_weight <= 0:
            df_wave["WaveWeight"] = 1.0 / len(df_wave)
        else:
            df_wave["WaveWeight"] = df_wave["IndexWeight"] / total_index_weight

        df_wave["Wave"] = "SP500_Wave"

    else:
        wave_slice = wave_weights_df[wave_weights_df["Wave"] == active_wave].copy()
        if wave_slice.empty:
            st.error(f"[WAVE ERROR] No holdings found for '{active_wave}' in wave_weights.csv.")
            return

        df_wave = wave_slice.merge(universe_df, on="Ticker", how="left")
        df_wave = df_wave.dropna(subset=["Name"])
        if df_wave.empty:
            st.error(
                f"[WAVE ERROR] After joining with list.csv, "
                f"no valid universe rows found for '{active_wave}'."
            )
            return

        # 🔹 FINAL DEDUPE LAYER: one row per Ticker in this wave
        df_wave = (
            df_wave.groupby("Ticker", as_index=False)
            .agg({
                "WaveWeight": "sum",
                "Name": "first",
                "Sector": "first",
                "IndexWeight": "first",
                "MarketValue": "first",
                "Price": "first",
            })
        )

        total = df_wave["WaveWeight"].sum()
        if total > 0:
            df_wave["WaveWeight"] = df_wave["WaveWeight"] / total

        df_wave["Wave"] = active_wave

    render_header(active_wave, mode, float(equity_exposure))
    show_wave_snapshot(df_wave, float(equity_exposure))

    tickers = df_wave["Ticker"].dropna().unique().tolist()
    prices_df = fetch_live_prices(tickers)
    spy_df = fetch_history(SPY_SYMBOL)
    vix_df = fetch_history(VIX_SYMBOL)

    # Layout: Top 10 + SPY/VIX
    st.markdown("<hr/>", unsafe_allow_html=True)
    col_left, col_right = st.columns([2.1, 1.0])

    with col_left:
        show_top_holdings(df_wave, prices_df)

    with col_right:
        tabs = st.tabs(["S&P 500 (SPY)", "VIX"])
        with tabs[0]:
            show_spy_chart(spy_df)
        with tabs[1]:
            show_vix_chart(vix_df)

    st.markdown("<hr/>", unsafe_allow_html=True)

    show_alpha_placeholder()

    with st.expander("View Full Wave Holdings (raw)"):
        display_cols = [
            "Ticker",
            "Name",
            "Sector",
            "Wave",
            "WaveWeight",
            "IndexWeight",
            "MarketValue",
            "Price",
        ]
        existing = [c for c in display_cols if c in df_wave.columns]
        st.dataframe(df_wave[existing], use_container_width=True)

    if auto_refresh:
        st.markdown(
            f"<p style='color:#8e94a8; font-size:11px; margin-top:6px;'>"
            f"Auto-refresh active — console will rerun every {REFRESH_SECONDS} seconds."
            f"</p>",
            unsafe_allow_html=True,
        )
        time.sleep(REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
