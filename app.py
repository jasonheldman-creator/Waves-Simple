# app.py  — WAVES Intelligence™ Console (Clean Rebuild)
#
# Requirements (in requirements.txt):
#   streamlit
#   pandas
#   yfinance
#
# Then run:  streamlit run app.py

import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

UNIVERSE_CSV = "List.csv"        # <-- THIS IS THE ONLY CSV WE USE
DEFAULT_TICKER_COLS = ["symbol", "ticker", "Symbol", "Ticker"]
DEFAULT_NAME_COLS = ["name", "Name", "company", "Company"]
DEFAULT_WEIGHT_COLS = ["weight", "Weight", "weight_pct", "Weight %"]

REFRESH_SECONDS = 60             # live price refresh cadence

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(show_spinner=True)
def load_universe():
    """
    Load the universe from List.csv in the current folder.
    This is intentionally simple and **only** checks one file.
    """
    if not os.path.exists(UNIVERSE_CSV):
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' not found in current directory."

    try:
        df = pd.read_csv(UNIVERSE_CSV)
    except Exception as e:
        return None, f"[UNIVERSE ERROR] Failed to read '{UNIVERSE_CSV}': {e}"

    if df.empty:
        return None, f"[UNIVERSE ERROR] '{UNIVERSE_CSV}' is empty."

    # Try to identify columns flexibly
    ticker_col = next((c for c in DEFAULT_TICKER_COLS if c in df.columns), None)
    name_col   = next((c for c in DEFAULT_NAME_COLS   if c in df.columns), None)
    weight_col = next((c for c in DEFAULT_WEIGHT_COLS if c in df.columns), None)

    if ticker_col is None:
        return None, (
            f"[UNIVERSE WARNING] Could not find a ticker/symbol column in '{UNIVERSE_CSV}'. "
            f"Columns found: {list(df.columns)}"
        )

    # Normalize column names for the app
    df = df.copy()
    df.rename(columns={
        ticker_col: "Ticker",
        name_col if name_col else ticker_col: "Name"
    }, inplace=True)

    if weight_col:
        df.rename(columns={weight_col: "Weight"}, inplace=True)
    else:
        # If no weight, assign equal weight
        df["Weight"] = 1 / len(df)

    # Ensure numeric weight
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
            open_price = float(last_row["Open"]) if "Open" in last_row else price
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

def google_finance_link(ticker):
    """
    Build a Google Finance link for a given ticker.
    We don't try to guess the exact exchange – this is just a clean lookup.
    """
    base = "https://www.google.com/finance/quote/"
    # Google will usually figure out the exchange from context
    return f"{base}{ticker}"


def style_title():
    st.markdown(
        """
        <style>
        .main {
            background-color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 style="color:#18FFB2; text-align:center; font-family:system-ui;">
            WAVES Intelligence™ — Live Engine Console
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style="color:#CCCCCC; text-align:center; font-family:system-ui;">
            Standard Mode • Alpha-Minus-Beta Discipline • Vector-Driven Allocation
        </p>
        """,
        unsafe_allow_html=True
    )


def show_universe_summary(df: pd.DataFrame):
    total_names = len(df)
    top10_weight = df.sort_values("Weight", ascending=False)["Weight"].head(10).sum()
    st.subheader("Wave Snapshot")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Holdings", f"{total_names:,}")
    col2.metric("Top 10 Weight", f"{top10_weight*100:0.1f}%")
    col3.metric("Last Refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def show_top_holdings(df: pd.DataFrame, prices_df: pd.DataFrame):
    st.subheader("Top 10 Positions")

    top = df.sort_values("Weight", ascending=False).head(10).copy()

    # Attach prices if available
    if not prices_df.empty:
        top = top.merge(
            prices_df.reset_index(),
            on="Ticker",
            how="left"
        )

    # Build clickable links
    links = []
    for t in top["Ticker"]:
        url = google_finance_link(t)
        links.append(f"[{t}]({url})")
    top["Ticker"] = links

    # Format weight
    top["Weight %"] = (top["Weight"] * 100).round(2)
    display_cols = ["Ticker", "Name", "Weight %"]
    if "Price" in top.columns:
        display_cols += ["Price", "Change %"]

    st.dataframe(
        top[display_cols],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="WAVES Intelligence Console",
        layout="wide",
    )

    style_title()

    with st.sidebar:
        st.markdown("### Engine Controls")
        auto_refresh = st.checkbox("Auto-refresh prices", value=True)
        st.write(f"Refresh every {REFRESH_SECONDS} seconds when enabled.")

    df, error_msg = load_universe()

    if error_msg:
        st.error(error_msg)
        st.stop()

    if df is None or df.empty:
        st.error("[UNIVERSE ERROR] No data available from List.csv.")
        st.stop()

    show_universe_summary(df)

    # Fetch live prices
    tickers = df["Ticker"].dropna().unique().tolist()
    prices_df = fetch_live_prices(tickers)

    show_top_holdings(df, prices_df)

    # Raw universe view
    with st.expander("View Full Universe (raw)"):

        st.dataframe(
            df[["Ticker", "Name", "Weight"]],
            use_container_width=True
        )

    if auto_refresh:
        # Simple auto-refresh note
        st.markdown(
            f"<p style='color:#888888; font-size:12px;'>Auto-refresh active — "
            f"prices will update every {REFRESH_SECONDS} seconds when the app reloads.</p>",
            unsafe_allow_html=True
        )
        # Streamlit workaround: this triggers a rerun
        time.sleep(REFRESH_SECONDS)
        st.experimental_rerun()


if __name__ == "__main__":
    main()