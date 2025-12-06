# app.py — WAVES Intelligence™ Console (Multi-Wave, Joined in Code)
#
# Files expected in this folder:
#   - list.csv          (universe: Ticker, Company, Weight, Sector, Market Value, Price)
#   - wave_weights.csv  (Ticker, Wave, Weight  + optional '#' comment lines)
#
# This app:
#   • Uses list.csv as the master universe
#   • Uses wave_weights.csv to define non-core Waves
#   • Treats SP500_Wave as "full index": all rows from list.csv
#   • Joins wave_weights.csv to list.csv INSIDE the app (you never touch list.csv)

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

UNIVERSE_CSV = "list.csv"
WAVE_WEIGHTS_CSV = "wave_weights.csv"

REFRESH_SECONDS = 60
VIX_SYMBOL = "^VIX"
VIX_PERIOD = "6mo"
VIX_INTERVAL = "1d"

st.set_page_config(page_title="WAVES Intelligence Console", layout="wide")


# ------------------------------------------------------------
# DATA LOADERS
# ------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_universe():
    """
    Load security universe from list.csv and normalize columns to:
        Ticker, Name, Sector, IndexWeight, MarketValue, Price
    """
    if not os.path.exists(UNIVERSE_CSV):
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' not found."

    try:
        df = pd.read_csv(UNIVERSE_CSV)
    except Exception as e:
        return None, f"[UNIVERSE ERROR] Failed to read '{UNIVERSE_CSV}': {e}"

    if df.empty:
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' is empty."

    # Expecting: Ticker,Company,Weight,Sector,Market Value,Price
    # Rename into cleaner names
    col_map = {}
    if "Company" in df.columns:
        col_map["Company"] = "Name"
    if "Weight" in df.columns:
        col_map["Weight"] = "IndexWeight"
    if "Market Value" in df.columns:
        col_map["Market Value"] = "MarketValue"

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = ["Ticker", "Name", "Sector", "IndexWeight", "MarketValue", "Price"]
    for col in required:
        if col not in df.columns:
            # Allow missing Sector / MarketValue / Price gracefully
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

    # Clean up datatypes
    df["IndexWeight"] = pd.to_numeric(df["IndexWeight"], errors="coerce").fillna(0.0)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

    return df, None


@st.cache_data(show_spinner=True)
def load_wave_weights():
    """
    Load Wave allocations from wave_weights.csv.

    Expected format (long):

        Ticker,Wave,Weight
        NVDA,SP500_Wave,0.01486
        AAPL,Growth_Wave,0.10
        ...

    Lines starting with '#' are treated as comments.
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
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    # Drop zero / negative weights
    df = df[df["Weight"] > 0]
    if df.empty:
        return None, f"[WAVE ERROR] All weights in '{WAVE_WEIGHTS_CSV}' are zero."

    # Normalize within each Wave so weights sum to 1.0
    totals = df.groupby("Wave")["Weight"].transform("sum").replace(0, 1.0)
    df["Weight"] = df["Weight"] / totals

    df = df.rename(columns={"Weight": "WaveWeight"})
    return df, None


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
def fetch_vix_series():
    """Fetch VIX timeseries for small volatility chart."""
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
        .main { background-color: #050608; }
        .block-container { padding-top: 1.4rem; padding-bottom: 1.4rem; }
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
    top10_weight = (
        df_wave.sort_values("WaveWeight", ascending=False)["WaveWeight"].head(10).sum()
    )
    last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.subheader("Wave Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Holdings", f"{total_names:,}")
    col2.metric("Top 10 Weight", f"{top10_weight*100:0.1f}%")
    col3.metric("Equity Exposure", f"{equity_exposure:.0f}%")
    col4.metric("Last Refresh", last_refresh)


def show_top_holdings(df_wave: pd.DataFrame, prices_df: pd.DataFrame):
    st.subheader("Top 10 Positions")

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

    st.dataframe(top[display_cols], use_container_width=True, hide_index=True)

    chart_df = df_wave.sort_values("WaveWeight", ascending=False).head(10).copy()
    chart_df["Weight %"] = chart_df["WaveWeight"] * 100

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
        df_wave.groupby("Sector")["WaveWeight"]
        .sum()
        .reset_index()
        .sort_values("WaveWeight", ascending=False)
    )
    sector_df["Weight %"] = sector_df["WaveWeight"] * 100

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

    # Build list of available Waves from weights file
    waves = sorted(wave_weights_df["Wave"].dropna().unique().tolist())

    # Ensure SP500_Wave is always present as core index
    if "SP500_Wave" not in waves:
        waves.insert(0, "SP500_Wave")

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Engine Controls")

        # Try to default to SP500_Wave if available
        default_index = 0
        if "SP500_Wave" in waves:
            default_index = waves.index("SP500_Wave")

        active_wave = st.selectbox("Wave", options=waves, index=default_index)

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

    # Build active Wave holdings
    if active_wave == "SP500_Wave":
        # Core index uses the full universe with IndexWeight
        df_wave = universe_df.copy()
        total_index_weight = df_wave["IndexWeight"].sum()
        if total_index_weight <= 0:
            df_wave["WaveWeight"] = 1.0 / len(df_wave)
        else:
            df_wave["WaveWeight"] = df_wave["IndexWeight"] / total_index_weight
        df_wave["Wave"] = "SP500_Wave"
    else:
        wave_slice = wave_weights_df[wave_weights_df["Wave"] == active_wave].copy()
        if wave_slice.empty:
            st.error(
                f"[WAVE ERROR] No holdings found for '{active_wave}' in wave_weights.csv."
            )
            return

        df_wave = wave_slice.merge(universe_df, on="Ticker", how="left")
        df_wave = df_wave.dropna(subset=["Name"])
        if df_wave.empty:
            st.error(
                f"[WAVE ERROR] After joining with list.csv, "
                f"no valid universe rows found for '{active_wave}'."
            )
            return

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
        display_cols = [
            "Ticker",
            "Name",
            "Sector",
            "WaveWeight",
            "IndexWeight",
            "MarketValue",
            "Price",
        ]
        display_cols = [c for c in display_cols if c in df_wave.columns]
        st.dataframe(df_wave[display_cols], use_container_width=True)

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
