# streamlit_app.py

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG & GLOBAL STYLING
# --------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS: dark theme, compact spacing, neon accents, Bloomberg-like density
st.markdown(
    """
    <style>
    /* Overall page */
    body {
        background-color: #050514;
        color: #f5f5ff;
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 1400px;
    }

    header, footer {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #050514;
        border-right: 1px solid #202036;
    }

    /* Titles */
    h1, h2, h3, h4 {
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Metrics pill styles */
    .waves-metrics-box {
        background: radial-gradient(circle at top left, #1d4630 0, #050514 55%);
        border: 1px solid #22ff99;
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 0 18px rgba(34, 255, 153, 0.25);
        font-size: 0.85rem;
    }

    .waves-metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        grid-row-gap: 0.35rem;
        grid-column-gap: 0.75rem;
    }

    .waves-metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #a7acc2;
    }

    .waves-metric-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e6fff5;
    }

    .waves-metric-sub {
        font-size: 0.7rem;
        color: #7debb8;
    }

    /* Holdings table */
    .waves-table-container {
        border-radius: 10px;
        border: 1px solid #202036;
        overflow: hidden;
        margin-top: 0.4rem;
    }

    table.waves-holdings {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }

    table.waves-holdings thead {
        background: #09091f;
    }

    table.waves-holdings th,
    table.waves-holdings td {
        padding: 0.35rem 0.55rem;
        text-align: left;
        white-space: nowrap;
    }

    table.waves-holdings th {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #a7acc2;
        border-bottom: 1px solid #1b1b30;
    }

    table.waves-holdings tbody tr:nth-child(even) {
        background: #060618;
    }

    table.waves-holdings tbody tr:nth-child(odd) {
        background: #040413;
    }

    table.waves-holdings tbody tr:hover {
        background: #10102a;
    }

    .waves-link {
        color: #33ffe2;
        text-decoration: none;
    }

    .waves-link:hover {
        text-decoration: underline;
    }

    .waves-pos {
        color: #e6e9ff;
        font-weight: 500;
    }

    .waves-weight {
        font-variant-numeric: tabular-nums;
    }

    .waves-change-pos {
        color: #14ffb1;
    }

    .waves-change-neg {
        color: #ff4b81;
    }

    .waves-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.15rem;
        font-size: 0.7rem;
        padding: 0.08rem 0.35rem;
        border-radius: 999px;
        background: rgba(18, 255, 168, 0.08);
        border: 1px solid rgba(18, 255, 168, 0.4);
        color: #86ffcf;
    }

    /* Flash animations */
    @keyframes flashGreen {
        0% { background-color: rgba(34, 255, 153, 0.0); }
        30% { background-color: rgba(34, 255, 153, 0.17); }
        60% { background-color: rgba(34, 255, 153, 0.0); }
        100% { background-color: rgba(34, 255, 153, 0.0); }
    }
    @keyframes flashRed {
        0% { background-color: rgba(255, 77, 120, 0.0); }
        30% { background-color: rgba(255, 77, 120, 0.18); }
        60% { background-color: rgba(255, 77, 120, 0.0); }
        100% { background-color: rgba(255, 77, 120, 0.0); }
    }

    tr.flash-up {
        animation: flashGreen 1.3s ease-out;
    }

    tr.flash-down {
        animation: flashRed 1.3s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

DEFAULT_CSV_PATH = "live_snapshot.csv"


def load_snapshot(uploaded_file):
    """Load the live snapshot from upload or default path."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv(DEFAULT_CSV_PATH)
    except Exception:
        return None


def compute_core_metrics(df):
    """Compute Total holdings, Equity vs Cash, Largest Position, Alpha Capture."""
    if df is None or df.empty:
        return {
            "total_holdings": 0,
            "equity_value": 0.0,
            "cash_value": 0.0,
            "largest_name": "—",
            "largest_weight": None,
            "alpha_capture": None,
        }

    total_holdings = len(df)

    # Equity vs Cash – try multiple schema options
    equity_value = 0.0
    cash_value = 0.0

    if "is_cash" in df.columns and "market_value" in df.columns:
        cash_value = df.loc[df["is_cash"] == True, "market_value"].sum()
        equity_value = df.loc[df["is_cash"] == False, "market_value"].sum()
    elif "asset_type" in df.columns and "market_value" in df.columns:
        cash_mask = df["asset_type"].str.contains("cash", case=False, na=False)
        cash_value = df.loc[cash_mask, "market_value"].sum()
        equity_value = df.loc[~cash_mask, "market_value"].sum()
    elif "market_value" in df.columns:
        equity_value = df["market_value"].sum()

    # Largest position – use weight if present, else market_value
    if "weight" in df.columns:
        idx = df["weight"].idxmax()
        row = df.loc[idx]
        largest_name = row.get("ticker", row.get("name", "Largest"))
        largest_weight = float(row["weight"])
    elif "market_value" in df.columns:
        idx = df["market_value"].idxmax()
        row = df.loc[idx]
        largest_name = row.get("ticker", row.get("name", "Largest"))
        mv = float(row["market_value"])
        total_mv = df["market_value"].sum()
        largest_weight = mv / total_mv if total_mv > 0 else None
    else:
        largest_name = "—"
        largest_weight = None

    # Alpha capture – try common column names
    alpha_capture = None
    for col in ["alpha_capture", "AlphaCaptured", "alpha"]:
        if col in df.columns:
            # If per-position, aggregate; else, if scalar, keep scalar
            if df[col].ndim > 0:
                alpha_capture = df[col].sum()
            else:
                alpha_capture = float(df[col])
            break

    return {
        "total_holdings": total_holdings,
        "equity_value": equity_value,
        "cash_value": cash_value,
        "largest_name": largest_name,
        "largest_weight": largest_weight,
        "alpha_capture": alpha_capture,
    }


def format_dollar(x):
    if x is None:
        return "—"
    return f"${x:,.0f}"


def format_pct(x, decimals=1):
    if x is None:
        return "—"
    return f"{x * 100:.{decimals}f}%"


def get_top_holdings(df, n=10):
    if df is None or df.empty:
        return pd.DataFrame()
    if "weight" in df.columns:
        return df.sort_values("weight", ascending=False).head(n).copy()
    elif "market_value" in df.columns:
        return df.sort_values("market_value", ascending=False).head(n).copy()
    else:
        return df.head(n).copy()


def update_weight_history(top_df):
    """
    Track previous weights in session_state to detect direction of change.
    Returns a dict {ticker: "up"/"down"/None}.
    """
    if "weight_history" not in st.session_state:
        st.session_state.weight_history = {}

    prev = st.session_state.weight_history
    directions = {}

    for _, row in top_df.iterrows():
        ticker = str(row.get("ticker", row.get("name", "")))
        if not ticker:
            continue
        w = float(row["weight"]) if "weight" in row and not pd.isna(row["weight"]) else None
        if w is None:
            directions[ticker] = None
            continue

        if ticker in prev:
            old_w = prev[ticker]
            if old_w is not None:
                # Only flash if meaningful move (0.02% weight change)
                delta = w - old_w
                if delta > 0.0002:
                    directions[ticker] = "up"
                elif delta < -0.0002:
                    directions[ticker] = "down"
                else:
                    directions[ticker] = None
            else:
                directions[ticker] = None
        else:
            directions[ticker] = None

        # Update history baseline
        prev[ticker] = w

    st.session_state.weight_history = prev
    return directions


def render_holdings_table(top_df):
    if top_df.empty:
        st.info("No holdings available in the current snapshot.")
        return

    # Ensure required columns exist
    for col in ["ticker", "name", "weight", "price", "change_1d", "url"]:
        if col not in top_df.columns:
            top_df[col] = np.nan

    # Compute movement directions for flashing
    directions = update_weight_history(top_df)

    # Build HTML table manually for precise styling + flashing
    rows_html = []
    for _, row in top_df.iterrows():
        ticker = str(row.get("ticker", "—"))
        name = str(row.get("name", "—"))
        url = row.get("url", None)

        weight = row.get("weight", np.nan)
        weight_str = f"{weight * 100:.2f}%" if pd.notna(weight) else "—"

        price = row.get("price", np.nan)
        price_str = f"${price:,.2f}" if pd.notna(price) else "—"

        chg = row.get("change_1d", np.nan)
        if pd.isna(chg):
            chg_str = "—"
            chg_class = ""
        else:
            chg_str = f"{chg * 100:+.2f}%"
            chg_class = "waves-change-pos" if chg >= 0 else "waves-change-neg"

        link_html = f'<a href="{url}" class="waves-link" target="_blank">{ticker}</a>' if isinstance(url, str) and url else ticker

        direction = directions.get(ticker)
        flash_class = ""
        if direction == "up":
            flash_class = "flash-up"
        elif direction == "down":
            flash_class = "flash-down"

        row_html = f"""
        <tr class="{flash_class}">
            <td class="waves-pos">{link_html}</td>
            <td>{name}</td>
            <td class="waves-weight">{weight_str}</td>
            <td>{price_str}</td>
            <td class="{chg_class}">{chg_str}</td>
        </tr>
        """
        rows_html.append(row_html)

    table_html = f"""
    <div class="waves-table-container">
      <table class="waves-holdings">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Name</th>
            <th>Weight</th>
            <th>Last</th>
            <th>1D</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """

    st.markdown(table_html, unsafe_allow_html=True)


def render_allocation_chart(df):
    """Simple high-density bar chart for top holdings weights."""
    if df is None or df.empty:
        return

    if "ticker" not in df.columns or "weight" not in df.columns:
        return

    chart_df = get_top_holdings(df, 10)[["ticker", "weight"]].copy()

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title="Portfolio Weight", axis=alt.Axis(format=".0%")),
            y=alt.Y("ticker:N", sort="-x", title=""),
            tooltip=[alt.Tooltip("ticker:N"), alt.Tooltip("weight:Q", format=".2%")],
        )
        .properties(height=260)
    )

    st.altair_chart(chart, use_container_width=True)


# --------------------------------------------------
# SIDEBAR — WAVE SELECTOR, MODES, PROFILE
# --------------------------------------------------

st.sidebar.markdown(
    """
    <h2 style="margin-bottom:0.25rem;">WAVES Console</h2>
    <div style="font-size:0.78rem;color:#8b90a8;margin-bottom:0.8rem;">
      Bloomberg-style terminal for WAVES Intelligence™
    </div>
    """,
    unsafe_allow_html=True,
)

wave_name = st.sidebar.selectbox(
    "Wave",
    [
        "S&P Alpha Wave",
        "Growth Wave",
        "Income Wave",
        "Future Power & Energy Wave",
        "Crypto Income Wave",
        "RWA Income Wave",
    ],
)

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload live_snapshot CSV",
    type=["csv"],
    help="If omitted, the app will attempt to load live_snapshot.csv from the working directory.",
)

st.sidebar.markdown(
    """
    <div style="font-size:0.7rem;color:#7c819b;margin-top:0.5rem;">
      Tip: refresh every few seconds to see neon flashes when weights move.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# MAIN LAYOUT — HIGH-DENSITY GRID
# --------------------------------------------------

df = load_snapshot(uploaded_file)

# Top row: Branding/title left, metrics box right
top_left, top_right = st.columns([0.7, 0.3])

with top_left:
    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:0.6rem;margin-bottom:0.2rem;">
          <div style="font-size:1.4rem;font-weight:650;letter-spacing:0.12em;text-transform:uppercase;">
            WAVES <span style="color:#22ff99;">Console</span>
          </div>
          <div class="waves-badge">{wave_name}</div>
          <div class="waves-badge">{mode}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(
        f"""
        <div style="font-size:0.78rem;color:#8c91aa;margin-bottom:0.3rem;">
          Live snapshot • {now_str}
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    metrics = compute_core_metrics(df)

    equity_value = metrics["equity_value"]
    cash_value = metrics["cash_value"]
    total_value = (equity_value or 0) + (cash_value or 0)
    equity_pct = equity_value / total_value if total_value > 0 else None
    cash_pct = cash_value / total_value if total_value > 0 else None

    st.markdown(
        f"""
        <div class="waves-metrics-box">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.14em;color:#9aa0c0;">
              Wave Snapshot
            </div>
            <div style="font-size:0.75rem;color:#9aa0c0;">T+0</div>
          </div>
          <div class="waves-metrics-grid">
            <div>
              <div class="waves-metric-label">Total Holdings</div>
              <div class="waves-metric-value">{metrics['total_holdings']}</div>
            </div>
            <div>
              <div class="waves-metric-label">Largest Position</div>
              <div class="waves-metric-value">{metrics['largest_name']}</div>
              <div class="waves-metric-sub">{format_pct(metrics['largest_weight'])}</div>
            </div>
            <div>
              <div class="waves-metric-label">Equity Exposure</div>
              <div class="waves-metric-value">{format_pct(equity_pct)}</div>
              <div class="waves-metric-sub">{format_dollar(equity_value)}</div>
            </div>
            <div>
              <div class="waves-metric-label">Cash Buffer</div>
              <div class="waves-metric-value">{format_pct(cash_pct)}</div>
              <div class="waves-metric-sub">{format_dollar(cash_value)}</div>
            </div>
          </div>
          <div style="margin-top:0.45rem;border-top:1px solid rgba(111,255,197,0.14);padding-top:0.35rem;">
            <div class="waves-metric-label" style="margin-bottom:0.1rem;">Alpha Capture</div>
            <div style="display:flex;justify-content:space-between;align-items:baseline;">
              <div class="waves-metric-value">
                {format_pct(metrics['alpha_capture'], 2) if metrics['alpha_capture'] is not None else "—"}
              </div>
              <div style="font-size:0.7rem;color:#7debb8;">vs benchmark (WaveScore regime)</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")  # tiny spacer

# Second row: left panel chart(s), right panel quick view / future Step 2 upgrades
mid_left, mid_right = st.columns([0.6, 0.4])

with mid_left:
    st.markdown("#### Allocation Snapshot")
    render_allocation_chart(df)

with mid_right:
    st.markdown("#### Wave Diagnostics")
    if df is None:
        st.warning("No snapshot loaded. Upload a CSV or provide live_snapshot.csv.")
    else:
        # Simple diagnostics placeholder (can be upgraded in later steps)
        if "beta" in df.columns:
            avg_beta = df["beta"].mean()
        else:
            avg_beta = None

        if "vol_20d" in df.columns:
            avg_vol = df["vol_20d"].mean()
        else:
            avg_vol = None

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="Avg Beta vs Benchmark",
                value=f"{avg_beta:.2f}" if avg_beta is not None else "—",
            )
        with col_b:
            st.metric(
                label="Avg 20D Volatility",
                value=f"{avg_vol*100:.1f}%" if avg_vol is not None else "—",
            )

        st.caption(
            "These tiles will evolve into a full **Step 2** high-density grid "
            "with factor, sector, and regime diagnostics."
        )

# Third row: full-width Top 10 holdings with flashing
st.markdown("#### Top 10 Wave Holdings")

if df is None:
    st.info("Waiting for data…")
else:
    top_df = get_top_holdings(df, 10)
    render_holdings_table(top_df)