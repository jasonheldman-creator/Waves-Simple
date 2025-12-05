# app.py — WAVES Intelligence™ Multi-Wave Console (with VIX)
#
# Expected files in this folder:
#   - list.csv          (universe: tickers, names, sectors, etc.)
#   - wave_weights.csv  (long format: Ticker,Wave,Weight + optional # comments)
#
# Requirements (in requirements.txt):
#   streamlit
#   pandas
#   yfinance
#   plotly
#
# Run locally:
#   streamlit run app.py

import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence Console",
    layout="wide",
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

UNIVERSE_CSV = "list.csv"
WAVE_WEIGHTS_CSV = "wave_weights.csv"

# Likely column names in list.csv
DEFAULT_TICKER_COLS = ["Ticker", "ticker", "Symbol", "symbol"]
DEFAULT_NAME_COLS = ["Name", "name", "Company", "company"]
SECTOR_COL_NAMES = ["Sector", "sector", "SECTOR"]

REFRESH_SECONDS = 60           # console rerun cadence
VIX_SYMBOL = "^VIX"
VIX_PERIOD = "6mo"
VIX_INTERVAL = "1d"


# ------------------------------------------------------------
# DATA LOADERS
# ------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_universe():
    """
    Load security universe from list.csv and normalize to:
    columns: Ticker, Name, Sector (Sector optional).
    """
    if not os.path.exists(UNIVERSE_CSV):
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' not found."

    try:
        df = pd.read_csv(UNIVERSE_CSV)
    except Exception as e:
        return None, f"[UNIVERSE ERROR] Failed to read '{UNIVERSE_CSV}': {e}"

    if df.empty:
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' is empty."

    # Ticker
    ticker_col = next((c for c in DEFAULT_TICKER_COLS if c in df.columns), None)
    if ticker_col is None:
        return None, (
            f"[UNIVERSE ERROR] Could not find ticker column in '{UNIVERSE_CSV}'. "
            f"Columns: {list(df.columns)}"
        )
    df = df.copy()
    df.rename(columns={ticker_col: "Ticker"}, inplace=True)

    # Name
    name_col = next((c for c in DEFAULT_NAME_COLS if c in df.columns), None)
    if name_col:
        df.rename(columns={name_col: "Name"}, inplace=True)
    else:
        df["Name"] = df["Ticker"]

    # Sector
    sector_col = next((c for c in SECTOR_COL_NAMES if c in df.columns), None)
    if sector_col:
        df.rename(columns={sector_col: "Sector"}, inplace=True)
    else:
        df["Sector"] = "Unclassified"

    return df, None


@st.cache_data(show_spinner=True)
def load_wave_weights():
    """
    Load Wave allocations from wave_weights.csv.

    Expected format (long, with # comments allowed):

        Ticker,Wave,Weight
        NVDA,SP500_Wave,0.1486
        AAPL,SP500_Wave,0.1388
        ...

    Comment lines starting with '#' are ignored.
    """
    if not os.path.exists(WAVE_WEIGHTS_CSV):
        return None, f"[WAVE ERROR] '{WAVE_WEIGHTS_CSV}' not found."

    try:
        # 'comment' tells pandas to ignore lines starting with '#'
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

    df_long = df.copy()
    df_long["Weight"] = pd.to_numeric(df_long["Weight"], errors="coerce").fillna(0.0)

    # Drop zero/negative weights
    df_long = df_long[df_long["Weight"] > 0]

    if df_long.empty:
        return None, f"[WAVE ERROR] All weights in '{WAVE_WEIGHTS_CSV}' are zero."

    # Normalize weights within each Wave so each sums to 1.0
    totals = df_long.groupby("Wave")["Weight"].transform("sum").replace(0, 1.0)
    df_long["Weight"] = df_long["Weight"] / totals

    return df_long, None


@st.cache_data(show_spinner=False)
def fetch_live_prices(tickers):
    """Fetch latest price + daily change using yfinance."""
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
                    "Price": price,
                    "Change": change,
                    "Change %": change_pct,
                }
            )
        except Exception:
            continue

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data).set_index("Ticker")


@st.cache_data(show_spinner=False)
def fetch_vix_series():
    """Fetch VIX history for mini-volatility chart."""
    try:
        vix = yf.Ticker(VIX_SYMBOL)
        hist = vix.history(period=VIX_PERIOD, interval=VIX_INTERVAL)
        if hist.empty:
            return pd.DataFrame()
        df = hist.reset_index()[["Date", "Close"]]
        df.rename(columns={"Close": "VIX"}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------
# UI HELPERS
# ------------------------------------------------------------

def apply_global_style():
    st.markdown(
        """
        <style>
        .main {
            background-color: #050608;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .stMetric label, .stMetric span {
            font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono",
                         "Courier New", monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(active_wave: str, active_mode: str, equity_exposure: float):
    st.markdown(
        """
        <h1 style="color:#18FFB2; text-align:center; font-family:system-ui; margin-bottom:0;">
            WAVES Intelligence™ — Live Engine Console
        </h1>
        """,
        unsafe_allow_html=True,
    )

    subtitle = (
        f"Wave: <b>{active_wave}</b> • Mode: <b>{active_mode}</b> • "
        "Alpha-Minus-Beta Discipline • Vector-Driven Allocation"
    )
    st.markdown(
        f"""
        <p style="color:#BBBBBB; text-align:center; font-family:system-ui; margin-top:4px;">
            {subtitle}<br/>
            Equity Exposure: <b>{equity_exposure:.0f}%</b> • Cash Buffer: <b>{100-equity_exposure:.0f}%</b>
        </p>
        """,
        unsafe_allow_html=True,
    )


def show_wave_snapshot(df_wave: pd.DataFrame, equity_exposure: float):
    total_names = len(df_wave)
    top10_weight = df_wave.sort_values("Weight", ascending=False)["Weight"].head(10).sum()
    last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.subheader("Wave Snapshot")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Holdings", f"{total_names:,}")
    col2.metric("Top 10 Weight", f"{top10_weight*100:0.1f}%")
    col3.metric("Equity Exposure", f"{equity_exposure:.0f}%")
    col4.metric("Last Refresh", last_refresh)


def show_top_holdings(df_wave: pd.DataFrame, prices_df: pd.DataFrame):
    st.subheader("Top 10 Positions")

    top = df_wave.sort_values("Weight", ascending=False).head(10).copy()

    if not prices_df.empty:
        top = top.merge(prices_df.reset_index(), on="Ticker", how="left")

    top["Weight %"] = (top["Weight"] * 100).round(2)

    display_cols = ["Ticker", "Name", "Weight %"]
    if "Price" in top.columns:
        display_cols.append("Price")
    if "Change %" in top.columns:
        top["Change %"] = top["Change %"].round(2)
        display_cols.append("Change %")

    st.dataframe(
        top[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    chart_df = df_wave.sort_values("Weight", ascending=False).head(10).copy()
    chart_df["Weight %"] = chart_df["Weight"] * 100
    fig = px.bar(
        chart_df,
        x="Weight %",
        y="Ticker",
        orientation="h",
        title="Top 10 Weight Allocation",
        labels={"Weight %": "Weight (%)", "Ticker": "Ticker"},
    )
    fig.update_layout(
        title_x=0.0,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_sector_breakdown(df_wave: pd.DataFrame):
    if "Sector" not in df_wave.columns:
        st.info("No 'Sector' column found in list.csv — sector breakdown not available.")
        return

    sector_df = (
        df_wave.groupby("Sector")["Weight"]
        .sum()
        .reset_index()
        .sort_values("Weight", ascending=False)
    )
    sector_df["Weight %"] = sector_df["Weight"] * 100

    fig = px.bar(
        sector_df,
        x="Weight %",
        y="Sector",
        orientation="h",
        title="Sector Allocation",
        labels={"Weight %": "Weight (%)", "Sector": "Sector"},
    )
    fig.update_layout(
        title_x=0.0,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_vix_chart(vix_df: pd.DataFrame):
    if vix_df.empty:
        st.info("VIX data not available right now.")
        return

    fig = px.line(
        vix_df,
        x="Date",
        y="VIX",
        title=f"VIX ({VIX_PERIOD} • {VIX_INTERVAL})",
        labels={"VIX": "Index Level", "Date": ""},
    )
    fig.update_traces(line_width=2)
    fig.update_layout(
        title_x=0.0,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_alpha_placeholder():
    st.subheader("Alpha Capture (Preview)")
    st.markdown(
        """
        <p style="color:#AAAAAA; font-family:system-ui;">
        This terminal is wired to the <b>universe &amp; exposure</b> layer.
        Live alpha capture (WaveScore™, benchmark-relative returns, stress
        periods) runs in the WAVES performance engine, which can be plugged
        into this console as a next step for institutional partners.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------

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

    # All available Waves from the weights file
    waves = sorted(wave_weights_df["Wave"].unique().tolist())

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Engine Controls")

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

    # Build active Wave snapshot
    wave_slice = wave_weights_df[wave_weights_df["Wave"] == active_wave].copy()
    df_wave = wave_slice.merge(universe_df, on="Ticker", how="left")

    # Fallbacks if universe is missing metadata
    df_wave["Name"] = df_wave["Name"].fillna(df_wave["Ticker"])
    if "Sector" not in df_wave.columns:
        df_wave["Sector"] = "Unclassified"

    render_header(active_wave, mode, float(equity_exposure))
    show_wave_snapshot(df_wave, float(equity_exposure))

    tickers = df_wave["Ticker"].dropna().unique().tolist()
    prices_df = fetch_live_prices(tickers)
    vix_df = fetch_vix_series()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        show_top_holdings(df_wave, prices_df)

    with col_right:
        tabs = st.tabs(["Sectors", "VIX"])
        with tabs[0]:
            st.subheader("Sector View")
            show_sector_breakdown(df_wave)
        with tabs[1]:
            st.subheader("Volatility (VIX)")
            show_vix_chart(vix_df)

    show_alpha_placeholder()

    with st.expander("View Full Wave Holdings (raw)"):
        st.dataframe(
            df_wave[["Ticker", "Name", "Sector", "Weight"]],
            use_container_width=True,
        )

    if auto_refresh:
        st.markdown(
            f"<p style='color:#888888; font-size:12px;'>"
            f"Auto-refresh active — console will rerun every {REFRESH_SECONDS} seconds."
            f"</p>",
            unsafe_allow_html=True,
        )
        time.sleep(REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
