# app.py  — WAVES Intelligence™ Live Engine Console (Enhanced)
#
# Requirements (in requirements.txt):
#   streamlit
#   pandas
#   yfinance
#   plotly   # already in your requirements.txt
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

# ============================================================
# CONFIG
# ============================================================

UNIVERSE_CSV = "list.csv"        # <-- matches your actual filename

DEFAULT_TICKER_COLS = ["symbol", "ticker", "Symbol", "Ticker"]
DEFAULT_NAME_COLS   = ["name", "Name", "company", "Company"]
DEFAULT_WEIGHT_COLS = ["weight", "Weight", "weight_pct", "Weight %", "Weight%"]
SECTOR_COL_NAMES    = ["sector", "Sector", "SECTOR"]

REFRESH_SECONDS = 60             # live price refresh cadence (console rerun)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(show_spinner=True)
def load_universe():
    """
    Load the universe from list.csv in the current folder.
    Intentionally simple: we only look for ONE file.
    """
    if not os.path.exists(UNIVERSE_CSV):
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' not found in current directory."

    try:
        df = pd.read_csv(UNIVERSE_CSV)
    except Exception as e:
        return None, f"[UNIVERSE ERROR] Failed to read '{UNIVERSE_CSV}': {e}"

    if df.empty:
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' is empty."

    # Detect core columns flexibly
    ticker_col = next((c for c in DEFAULT_TICKER_COLS if c in df.columns), None)
    name_col   = next((c for c in DEFAULT_NAME_COLS   if c in df.columns), None)
    weight_col = next((c for c in DEFAULT_WEIGHT_COLS if c in df.columns), None)
    sector_col = next((c for c in SECTOR_COL_NAMES    if c in df.columns), None)

    if ticker_col is None:
        return None, (
            f"[UNIVERSE WARNING] Could not find a ticker/symbol column in '{UNIVERSE_CSV}'. "
            f"Columns found: {list(df.columns)}"
        )

    df = df.copy()

    # Normalize for internal use
    df.rename(columns={ticker_col: "Ticker"}, inplace=True)
    if name_col:
        df.rename(columns={name_col: "Name"}, inplace=True)
    else:
        df["Name"] = df["Ticker"]

    if weight_col:
        df.rename(columns={weight_col: "Weight"}, inplace=True)
    else:
        df["Weight"] = 1 / len(df)

    if sector_col:
        df.rename(columns={sector_col: "Sector"}, inplace=True)
    else:
        df["Sector"] = "Unclassified"

    # Ensure numeric weights and normalize to 1.0
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
    total_w = df["Weight"].sum()
    if total_w > 0:
        df["Weight"] = df["Weight"] / total_w

    return df, None


@st.cache_data(show_spinner=False)
def fetch_live_prices(tickers):
    """
    Fetch latest price + daily change using yfinance.
    Returns a DataFrame indexed by Ticker.
    """
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
            data.append({
                "Ticker": t,
                "Price": price,
                "Change": change,
                "Change %": change_pct
            })
        except Exception:
            # Skip tickers that fail
            continue

    if not data:
        return pd.DataFrame()

    prices_df = pd.DataFrame(data).set_index("Ticker")
    return prices_df


# ============================================================
# UI HELPERS
# ============================================================

def google_finance_link(ticker: str) -> str:
    """
    Build a Google Finance link for a given ticker.
    """
    base = "https://www.google.com/finance/quote/"
    return f"{base}{ticker}"


def style_page():
    st.set_page_config(
        page_title="WAVES Intelligence Console",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main {
            background-color: #000000;
        }
        .metric-label, .metric-value {
            font-family: system-ui;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header(active_mode: str, equity_exposure: float):
    st.markdown(
        """
        <h1 style="color:#18FFB2; text-align:center; font-family:system-ui; margin-bottom:0;">
            WAVES Intelligence™ — Live Engine Console
        </h1>
        """,
        unsafe_allow_html=True
    )

    subtitle = (
        f"Mode: <b>{active_mode}</b> • "
        "Alpha-Minus-Beta Discipline • Vector-Driven Allocation"
    )
    st.markdown(
        f"""
        <p style="color:#CCCCCC; text-align:center; font-family:system-ui; margin-top:4px;">
            {subtitle}<br/>
            Equity Exposure: <b>{equity_exposure:.0f}%</b> • Cash Buffer: <b>{100-equity_exposure:.0f}%</b>
        </p>
        """,
        unsafe_allow_html=True
    )


def show_universe_summary(df: pd.DataFrame, equity_exposure: float):
    total_names = len(df)
    top10_weight = df.sort_values("Weight", ascending=False)["Weight"].head(10).sum()
    last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.subheader("Wave Snapshot")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Holdings", f"{total_names:,}")
    col2.metric("Top 10 Weight", f"{top10_weight*100:0.1f}%")
    col3.metric("Equity Exposure", f"{equity_exposure:.0f}%")
    col4.metric("Last Refresh", last_refresh)


def show_top_holdings(df: pd.DataFrame, prices_df: pd.DataFrame):
    st.subheader("Top 10 Positions")

    top = df.sort_values("Weight", ascending=False).head(10).copy()

    if not prices_df.empty:
        top = top.merge(
            prices_df.reset_index(),
            on="Ticker",
            how="left"
        )

    # Build clickable ticker links
    tick_links = []
    for t in top["Ticker"]:
        url = google_finance_link(t)
        tick_links.append(f"[{t}]({url})")
    top["Ticker"] = tick_links

    top["Weight %"] = (top["Weight"] * 100).round(2)

    display_cols = ["Ticker", "Name", "Weight %"]
    if "Price" in top.columns:
        display_cols += ["Price"]
    if "Change %" in top.columns:
        top["Change %"] = top["Change %"].round(2)
        display_cols += ["Change %"]

    st.dataframe(
        top[display_cols],
        use_container_width=True,
        hide_index=True
    )

    # Bar chart of top 10 weights
    chart_df = df.sort_values("Weight", ascending=False).head(10).copy()
    chart_df["Weight %"] = chart_df["Weight"] * 100
    fig = px.bar(
        chart_df,
        x="Weight %", y="Ticker",
        orientation="h",
        title="Top 10 Weight Allocation",
        labels={"Weight %": "Weight (%)", "Ticker": "Ticker"}
    )
    fig.update_layout(
        title_x=0.0,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_sector_breakdown(df: pd.DataFrame):
    st.subheader("Sector Allocation")

    if "Sector" not in df.columns:
        st.info("No 'Sector' column found in list.csv — sector breakdown not available.")
        return

    sector_df = (
        df.groupby("Sector")["Weight"]
        .sum()
        .reset_index()
        .sort_values("Weight", ascending=False)
    )
    sector_df["Weight %"] = sector_df["Weight"] * 100

    fig = px.pie(
        sector_df,
        values="Weight %",
        names="Sector",
        title="Sector Weight (%)"
    )
    fig.update_layout(
        title_x=0.0,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF",
        legend_title_text="Sector"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_alpha_placeholder():
    st.subheader("Alpha Capture (Preview)")
    st.markdown(
        """
        <p style="color:#AAAAAA; font-family:system-ui;">
        This console is currently wired to the <b>universe & exposure</b> layer.
        Live alpha capture comes from the WAVES performance engine (WaveScore,
        benchmark-relative returns, and stress tests), which can be connected as
        the next step for Franklin or any institutional partner.
        </p>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    style_page()

    # ------------- Sidebar Controls -------------
    with st.sidebar:
        st.markdown("### Engine Controls")

        mode = st.radio(
            "Wave Mode",
            options=["Standard", "Alpha-Minus-Beta", "Private Logic"],
            index=0
        )

        equity_exposure = st.slider(
            "Equity Exposure (%)",
            min_value=0,
            max_value=100,
            value=90,
            step=1,
            help="Represents current net equity vs SmartSafe™ / cash."
        )

        auto_refresh = st.checkbox("Auto-refresh prices", value=True)
        st.caption(f"Console will rerun every {REFRESH_SECONDS} seconds when enabled.")

    # ------------- Data Load -------------
    df, error_msg = load_universe()

    if error_msg:
        st.error(error_msg)
        st.stop()

    if df is None or df.empty:
        st.error("[UNIVERSE ERROR] No data available from list.csv.")
        st.stop()

    # ------------- Header & Snapshot -------------
    render_header(mode, float(equity_exposure))
    show_universe_summary(df, float(equity_exposure))

    # ------------- Prices & Top Holdings -------------
    tickers = df["Ticker"].dropna().unique().tolist()
    prices_df = fetch_live_prices(tickers)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        show_top_holdings(df, prices_df)

    with col_right:
        show_sector_breakdown(df)

    # ------------- Alpha / Raw Data -------------
    show_alpha_placeholder()

    with st.expander("View Full Universe (raw)"):
        st.dataframe(
            df[["Ticker", "Name", "Sector", "Weight"]],
            use_container_width=True
        )

    # ------------- Auto-refresh Loop -------------
    if auto_refresh:
        st.markdown(
            f"<p style='color:#888888; font-size:12px;'>"
            f"Auto-refresh active — console will rerun every {REFRESH_SECONDS} seconds."
            f"</p>",
            unsafe_allow_html=True
        )
        time.sleep(REFRESH_SECONDS)
        st.experimental_rerun()


if __name__ == "__main__":
    main()