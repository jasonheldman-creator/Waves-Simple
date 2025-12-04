import streamlit as st
import pandas as pd

from waves_equity_universe_v2 import (
    WAVES_CONFIG,
    load_holdings_from_csv,
    compute_wave_nav,
)


# ----------------- Helpers ----------------- #

def get_wave_config_by_code(code: str):
    for w in WAVES_CONFIG:
        if w.code == code:
            return w
    return None


def add_quote_links(df: pd.DataFrame, ticker_col: str = "Ticker") -> pd.DataFrame:
    """Add a Google Finance quote URL for each ticker (for display / clicking)."""
    df = df.copy()
    df["Quote"] = df[ticker_col].apply(
        lambda x: f"https://www.google.com/finance/quote/{x}:NASDAQ"
    )
    return df


# ----------------- Page setup ----------------- #

st.set_page_config(
    page_title="WAVES Intelligence – Portfolio Wave Console",
    layout="wide",
)

# Simple dark styling so it feels closer to your old Bloomberg-style console
st.markdown(
    """
    <style>
    body { background-color: #040816; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Sidebar ----------------- #

st.sidebar.title("🌊 WAVES Console")

configured_waves = [w for w in WAVES_CONFIG if w.holdings_csv_url]
if not configured_waves:
    st.sidebar.error("No Waves configured with holdings_csv_url yet.")
    st.stop()

wave_codes = [w.code for w in configured_waves]

selected_wave_code = st.sidebar.selectbox(
    "Select Wave",
    wave_codes,
    index=wave_codes.index("SPX") if "SPX" in wave_codes else 0,
)

mode = st.sidebar.radio("Mode", ["Standard", "Private Logic™"])

benchmark_override = st.sidebar.selectbox(
    "Benchmark",
    ["Use Wave benchmark"] + [w.benchmark for w in configured_waves],
    index=0,
)

style_profile = st.sidebar.selectbox(
    "Style",
    ["Core – Large Cap", "Growth", "Income", "Global"],
)

type_profile = st.sidebar.selectbox(
    "Type",
    ["AI-Managed Wave", "Hybrid", "Passive"],
)

st.sidebar.markdown("### Override snapshot CSV (optional)")
override_file = st.sidebar.file_uploader(
    "Drop CSV here to override holdings for this session only",
    type=["csv"],
    label_visibility="collapsed",
)

st.sidebar.caption("Console is read-only – no live trades are placed.")


# ----------------- Load data for selected Wave ----------------- #

wave = get_wave_config_by_code(selected_wave_code)
if wave is None:
    st.error(f"Wave {selected_wave_code} not found in config.")
    st.stop()

# Use uploaded CSV if provided, otherwise source from URL
if override_file is not None:
    raw = pd.read_csv(override_file)
    # Try to normalize CSV if user drops something slightly different
    cols = [c.strip() for c in raw.columns]
    raw.columns = cols

    # Find ticker column
    ticker_col = None
    for candidate in ("Ticker", "Symbol", "ticker", "symbol", "TICKER", "SYMBOL"):
        if candidate in raw.columns:
            ticker_col = candidate
            break
    if ticker_col is None:
        st.error("Override CSV must contain a Ticker or Symbol column.")
        st.stop()
    raw.rename(columns={ticker_col: "Ticker"}, inplace=True)

    # Find weight column
    weight_col = None
    for candidate in ("Weight", "Weight (%)", "Weight%", "Index Weight"):
        if candidate in raw.columns:
            weight_col = candidate
            break

    if weight_col is None:
        st.error("Override CSV must contain a Weight / Weight (%) column.")
        st.stop()

    weights = raw[weight_col].astype(str).str.replace("%", "", regex=False)
    weights = weights.replace("", "0").astype(float)
    if weights.max() > 1.5:
        weights = weights / 100.0
    raw["Weight"] = weights / weights.sum()
    holdings = raw[["Ticker", "Weight"] + [c for c in raw.columns if c not in ("Ticker", "Weight")]]
else:
    # Normal path: use engine loader (already normalizes Weight)
    holdings = load_holdings_from_csv(wave.holdings_csv_url)

# Normalize and sort
holdings["Ticker"] = holdings["Ticker"].astype(str).str.upper()
holdings = holdings.sort_values("Weight", ascending=False).reset_index(drop=True)

top10 = holdings.head(10)
top10 = add_quote_links(top10)

total_holdings = len(holdings)
largest_pos = float(top10["Weight"].iloc[0]) if not top10.empty else 0.0

# Assume fully invested for now; SmartSafe layer can override later
equity_pct = 1.0
cash_pct = 0.0

# Compute returns / alpha / NAV
try:
    stats = compute_wave_nav(wave, holdings)
    wave_return = float(stats.get("wave_return", float("nan")))
    bench_return = float(stats.get("benchmark_return", float("nan")))
    alpha = float(stats.get("alpha", float("nan")))
    nav = float(stats.get("nav", float("nan")))
except Exception as e:
    # If anything fails (e.g., yfinance hiccup), degrade gracefully
    wave_return = bench_return = alpha = float("nan")
    nav = wave.default_notional
    st.warning(f"Price/return computation issue: {e}")


# ----------------- Header ----------------- #

bench_label = benchmark_override if benchmark_override != "Use Wave benchmark" else wave.benchmark

title_text = f"{wave.name} (LIVE Demo)"
st.markdown(
    f"""
    <h2 style="color:#E2ECFF;margin-bottom:0.2rem;">WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE</h2>
    <h1 style="color:#4DB8FF;margin-top:0rem;">{title_text}</h1>
    <p style="color:#9BA7C7;">
        Mode: <b>{mode}</b> · Benchmark: <b>{bench_label}</b> · Style: <b>{style_profile}</b> · Type: <b>{type_profile}</b>
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ----------------- Top grid: Top 10 + Snapshot card ----------------- #

col_left, col_right = st.columns([2.2, 1])

with col_left:
    st.subheader("Top 10 holdings")
    display_top10 = top10[["Ticker", "Weight"]].copy()
    display_top10["Weight"] = (display_top10["Weight"] * 100).round(2).astype(str) + "%"
    st.dataframe(
        display_top10,
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Positive 1D moves will render in green, negatives in red once price overlay is wired in.")

with col_right:
    st.subheader("Wave snapshot")
    wave_ret_str = f"{wave_return*100:.2f}%" if pd.notna(wave_return) else "N/A"
    bench_ret_str = f"{bench_return*100:.2f}%" if pd.notna(bench_return) else "N/A"
    alpha_str = f"{alpha*100:.2f}%" if pd.notna(alpha) else "N/A"
    nav_str = f"${nav:,.0f}" if pd.notna(nav) else "N/A"

    alpha_color = "#4AE17C" if pd.notna(alpha) and alpha >= 0 else "#FF5C7B"

    snapshot_html = f"""
    <div style="background:#050C24;border-radius:12px;padding:16px;color:#E2ECFF;border:1px solid #1A2A4A;">
      <div style="font-size:0.85rem;text-transform:uppercase;color:#7C8BB5;">Wave Snapshot</div>
      <div style="font-size:1.1rem;font-weight:600;margin-bottom:4px;">{wave.code} – {wave.name}</div>
      <hr style="border-color:#1A2A4A;" />
      <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
        <div>
          <div style="color:#7C8BB5;">Total holdings</div>
          <div style="font-size:1.2rem;font-weight:600;">{total_holdings}</div>
        </div>
        <div>
          <div style="color:#7C8BB5;">Equity vs Cash</div>
          <div style="font-size:1.2rem;font-weight:600;">{equity_pct*100:.0f}% / {cash_pct*100:.0f}%</div>
        </div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.9rem;margin-top:12px;">
        <div>
          <div style="color:#7C8BB5;">Largest position</div>
          <div style="font-size:1.2rem;font-weight:600;">{largest_pos*100:.1f}%</div>
        </div>
        <div>
          <div style="color:#7C8BB5;">Wave NAV (sim)</div>
          <div style="font-size:1.2rem;font-weight:600;">{nav_str}</div>
        </div>
      </div>
      <hr style="border-color:#1A2A4A;margin-top:12px;margin-bottom:8px;" />
      <div style="font-size:0.9rem;">
        <span style="color:#7C8BB5;">1-day Wave / Benchmark / Alpha:</span><br/>
        <span style="font-weight:600;">{wave_ret_str}</span> /
        <span style="font-weight:600;">{bench_ret_str}</span> /
        <span style="font-weight:600;color:{alpha_color};">{alpha_str}</span>
      </div>
    </div>
    """
    st.markdown(snapshot_html, unsafe_allow_html=True)


# ----------------- Second grid: charts ----------------- #

col_a, col_b, col_c = st.columns([1.4, 1.4, 1.2])

with col_a:
    st.subheader("Top 10 profile – Wave weight distribution")
    if not top10.empty:
        chart_df = top10.set_index("Ticker")["Weight"] * 100
        st.bar_chart(chart_df)
    else:
        st.info("No holdings found for this Wave.")

with col_b:
    st.subheader("Sector allocation (if available)")
    if "Sector" in holdings.columns:
        sector_df = (
            holdings.groupby("Sector")["Weight"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            * 100
        )
        st.bar_chart(sector_df)
    else:
        st.info("No 'Sector' column detected – add one to see sector allocation.")

with col_c:
    st.subheader("Holding rank")
    weights_sorted = holdings["Weight"].sort_values(ascending=False).reset_index(drop=True)
    weights_sorted.index = weights_sorted.index + 1  # rank
    st.line_chart(weights_sorted)


# ----------------- Bottom text panels ----------------- #

st.markdown("---")

col_bottom_left, col_bottom_right = st.columns(2)

with col_bottom_left:
    st.subheader("Mode overview")
    st.markdown(
        """
        **Standard mode** keeps the Wave tightly aligned with its benchmark with controlled tracking error,
        strict beta discipline, and lower turnover.

        **Private Logic™ mode** layers on proprietary leadership, regime-switching, and SmartSafe™ overlays
        to push risk-adjusted alpha while staying within institutional guardrails.
        """.strip()
    )

with col_bottom_right:
    st.subheader("Wave internals (from holdings snapshot)")
    st.markdown(
        """
        Breadth & movement detection will plug into this panel:

        - 1-day gain/loss breadth (advancers vs decliners)
        - % of holdings above / below key moving-average bands
        - Concentration metrics (top 5 / top 10 weight share)
        - Regime flags (risk-on / risk-off overlays)

        Add an explicit 1-day change column (e.g., `Change_1D` / `Return_1D`)
        in the source sheet to unlock full breadth analytics here.
        """.strip()
    )

st.caption("Prototype console – not investment advice. For internal WAVES Intelligence™ use only. Console is read-only; no live trades are placed.")