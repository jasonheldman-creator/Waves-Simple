import streamlit as st
import pandas as pd
from pathlib import Path

from waves_equity_universe_v2 import (
    WAVES_CONFIG,
    load_holdings_from_csv,
    compute_wave_nav,
)


# --------- HELPERS --------- #

def get_wave_config_by_code(code: str):
    for w in WAVES_CONFIG:
        if w.code == code:
            return w
    return None


def add_quote_links(df: pd.DataFrame, ticker_col: str = "Ticker") -> pd.DataFrame:
    df = df.copy()
    df["Quote"] = df[ticker_col].apply(
        lambda x: f"https://www.google.com/finance/quote/{x}:NASDAQ"
    )
    return df


# --------- PAGE SETUP --------- #

st.set_page_config(
    page_title="WAVES Intelligence – Portfolio Wave Console",
    layout="wide",
)

# Custom dark background to feel closer to your old UI
st.markdown(
    """
    <style>
    body { background-color: #040816; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- SIDEBAR: WAVE + MODE CONTROLS --------- #

st.sidebar.title("🌊 WAVES Console")

wave_codes = [w.code for w in WAVES_CONFIG if w.holdings_csv_url]
if not wave_codes:
    st.sidebar.error("No Waves configured with holdings_csv_url yet.")
    st.stop()

selected_wave_code = st.sidebar.selectbox(
    "Select Wave",
    wave_codes,
    index=wave_codes.index("SPX") if "SPX" in wave_codes else 0,
)

mode = st.sidebar.radio("Mode", ["Standard", "Private Logic™"])

benchmark_override = st.sidebar.selectbox(
    "Benchmark",
    ["Use Wave benchmark"] + [w.benchmark for w in WAVES_CONFIG],
    index=0,
)

style_profile = st.sidebar.selectbox("Style", ["Core – Large Cap", "Growth", "Income", "Global"])
type_profile = st.sidebar.selectbox("Type", ["AI-Managed Wave", "Hybrid", "Passive"])

st.sidebar.markdown("### Override snapshot CSV (optional)")
override_file = st.sidebar.file_uploader(
    "Drop CSV here to override holdings for this session only", type=["csv"], label_visibility="collapsed"
)

st.sidebar.caption("Console is read-only – no live trades are placed.")


# --------- LOAD DATA FOR SELECTED WAVE --------- #

wave = get_wave_config_by_code(selected_wave_code)
if wave is None:
    st.error(f"Wave {selected_wave_code} not found in config.")
    st.stop()

if override_file is not None:
    holdings_raw = pd.read_csv(override_file)
else:
    holdings_raw = load_holdings_from_csv(wave.holdings_csv_url)

# holdings_raw already normalized to Weight in waves_equity_universe_v2
holdings = holdings_raw.copy()
holdings["Ticker"] = holdings["Ticker"].astype(str).str.upper()
holdings = holdings.sort_values("Weight", ascending=False).reset_index(drop=True)

top10 = holdings.head(10)
top10 = add_quote_links(top10)

# Compute snapshot stats (returns, alpha, nav)
stats = compute_wave_nav(wave, holdings)

wave_return = stats.get("wave_return", float("nan"))
bench_return = stats.get("benchmark_return", float("nan"))
alpha = stats.get("alpha", float("nan"))
nav = stats.get("nav", float("nan"))
total_holdings = len(holdings)
largest_pos = top10["Weight"].iloc[0] if not top10.empty else 0.0

# For now, assume fully invested (you can wire in SmartSafe later)
equity_pct = 1.0
cash_pct = 0.0


# --------- HEADER --------- #

title_text = f"{wave.name} (LIVE Demo)"
st.markdown(
    f"""
    <h2 style="color:#E2ECFF;margin-bottom:0.2rem;">WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE</h2>
    <h1 style="color:#4DB8FF;margin-top:0rem;">{title_text}</h1>
    <p style="color:#9BA7C7;">Benchmark-aware, AI-directed Wave – rendered in a single screen, Bloomberg-style.</p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# --------- TOP GRID: HOLDINGS TABLE + SNAPSHOT CARD --------- #

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
    st.caption("Positive 1D moves render in green, negatives in red. (Price-move overlay coming next.)")

with col_right:
    st.subheader("Wave snapshot")
    bench_label = benchmark_override if benchmark_override != "Use Wave benchmark" else wave.benchmark

    snapshot_html = f"""
    <div style="background:#050C24;border-radius:12px;padding:16px;color:#E2ECFF;border:1px solid #1A2A4A;">
      <div style="font-size:0.85rem;text-transform:uppercase;color:#7C8BB5;">Mode:</div>
      <div style="font-size:1.1rem;font-weight:600;margin-bottom:4px;">{mode} – Benchmark: {bench_label}</div>
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
          <div style="font-size:1.2rem;font-weight:600;">${nav:,.0f}</div>
        </div>
      </div>
      <hr style="border-color:#1A2A4A;margin-top:12px;margin-bottom:8px;" />
      <div style="font-size:0.9rem;">
        <span style="color:#7C8BB5;">1-day Wave / Benchmark / Alpha:</span><br/>
        <span style="font-weight:600;">{wave_return*100:.2f}%</span> /
        <span style="font-weight:600;">{bench_return*100:.2f}%</span> /
        <span style="font-weight:600;color:{'#4AE17C' if alpha>=0 else '#FF5C7B'};">{alpha*100:.2f}%</span>
      </div>
    </div>
    """
    st.markdown(snapshot_html, unsafe_allow_html=True)


# --------- SECOND GRID: WEIGHT BARS / SECTOR / HOLDING RANK --------- #

col_a, col_b, col_c = st.columns([1.4, 1.4, 1.2])

with col_a:
    st.subheader("Top 10 profile – Wave weight distribution")
    if not top10.empty:
        chart_df = top10.set_index("Ticker")["Weight"] * 100
        st.bar_chart(chart_df)
    else:
        st.info("No holdings found for this Wave.")

with col_b:
    st.subheader("Top 10 pre-sector view – Weight distribution")
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
        st.info("No 'Sector' column detected – add one to holdings to see sector allocation.")

with col_c:
    st.subheader("Holding rank")
    weights_sorted = holdings["Weight"].sort_values(ascending=False).reset_index(drop=True)
    weights_sorted.index = weights_sorted.index + 1  # rank
    st.line_chart(weights_sorted)


# --------- BOTTOM TEXT PANELS --------- #

st.markdown("---")

col_bottom_left, col_bottom_right = st.columns(2)

with col_bottom_left:
    st.subheader("Mode overview")
    st.markdown(
        """
        **Standard mode** steers the Wave tightly around its benchmark with controlled tracking error,
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

        Add an explicit 1-day change column (or Change_ID / Return_1D column)
        in the source sheet to unlock full breadth analytics here.
        """.strip()
    )

st.caption("Console is read-only – no live trades are placed. Prototype for internal WAVES Intelligence™ demos.")