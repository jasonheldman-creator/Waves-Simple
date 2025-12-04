import streamlit as st
import pandas as pd
import traceback
from dataclasses import dataclass

# ---------------------------------------------------------
# Attempt to import the Wave engine pieces that actually exist.
# We *do not* import WAVES_CONFIG from there anymore.
# ---------------------------------------------------------

IMPORT_ERROR = None

try:
    from waves_equity_universe_v2 import (
        load_holdings_from_csv,
        compute_wave_nav,
    )
except Exception as e:  # catches ImportError and any error inside that module
    IMPORT_ERROR = e

    def load_holdings_from_csv(*args, **kwargs):
        raise IMPORT_ERROR

    def compute_wave_nav(*args, **kwargs):
        raise IMPORT_ERROR


# ---------------------------------------------------------
# Local Wave config (instead of importing WAVES_CONFIG)
# ---------------------------------------------------------

@dataclass
class WaveConfig:
    code: str
    name: str
    benchmark: str
    holdings_csv_url: str
    default_notional: float = 100_000.0


# 🔗 You can change / add more Waves here
WAVES_CONFIG = [
    WaveConfig(
        code="SPX",
        name="S&P 500 Core Equity Wave",
        benchmark="SPY",
        holdings_csv_url=(
            "https://docs.google.com/spreadsheets/d/"
            "e/2PACX-1vT7VpPdWSUSyZP9CVXZwTgqx7a7mMD2aQMRqSESqZgiagh8wSeEm3RAWHvLlWmJtLqYrqj7UVjQIpq9"
            "/pub?gid=711820877&single=true&output=csv"
        ),
    ),
    # Example: add more Waves here later:
    # WaveConfig(code="R2K", name="Russell 2000 Wave", benchmark="IWM", holdings_csv_url="..."),
]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_wave_config_by_code(code: str):
    for w in WAVES_CONFIG:
        if getattr(w, "code", None) == code:
            return w
    return None


def add_quote_links(df: pd.DataFrame, ticker_col: str = "Ticker") -> pd.DataFrame:
    df = df.copy()
    if ticker_col in df.columns:
        df["Quote"] = df[ticker_col].astype(str).apply(
            lambda x: f"https://www.google.com/finance/quote/{x}:NASDAQ"
        )
    return df


# ---------------------------------------------------------
# Page config & light theming
# ---------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence – Equity Waves Console",
    layout="wide",
)

st.markdown(
    """
    <style>
    body { background-color: #040816; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# If the engine import failed, show a clear error and stop.
# ---------------------------------------------------------

if IMPORT_ERROR is not None:
    st.error("Wave engine import failed – see details below.")
    st.code(
        "".join(
            traceback.format_exception(
                type(IMPORT_ERROR), IMPORT_ERROR, IMPORT_ERROR.__traceback__
            )
        ),
        language="text",
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar – Wave selection & mode controls
# ---------------------------------------------------------

st.sidebar.title("🌊 WAVES Console")

configured_waves = [w for w in WAVES_CONFIG if getattr(w, "holdings_csv_url", None)]
if not configured_waves:
    st.sidebar.warning("No Waves configured with holdings_csv_url yet.")
    st.stop()

wave_codes = [w.code for w in configured_waves]
default_index = wave_codes.index("SPX") if "SPX" in wave_codes else 0

selected_wave_code = st.sidebar.selectbox(
    "Select Wave",
    wave_codes,
    index=default_index,
)

mode = st.sidebar.radio("Mode", ["Standard", "Private Logic™"], index=0)

benchmark_options = ["Use Wave benchmark"] + sorted(
    list({w.benchmark for w in WAVES_CONFIG})
)
benchmark_choice = st.sidebar.selectbox(
    "Benchmark",
    benchmark_options,
    index=0,
)

style_profile = st.sidebar.selectbox(
    "Style",
    ["Core – Large Cap", "Growth", "Income", "Global / Ex-US"],
    index=0,
)

type_profile = st.sidebar.selectbox(
    "Type",
    ["AI-Managed Wave", "Hybrid", "Passive overlay"],
    index=0,
)

st.sidebar.markdown("### Override snapshot CSV (optional)")
override_file = st.sidebar.file_uploader(
    "Drop CSV here to override holdings for this run",
    type=["csv"],
    label_visibility="collapsed",
)

st.sidebar.caption("Console is read-only – no live trades are placed.")


# ---------------------------------------------------------
# Load selected Wave config & holdings
# ---------------------------------------------------------

wave = get_wave_config_by_code(selected_wave_code)
if wave is None:
    st.error(f"Wave '{selected_wave_code}' not found in WAVES_CONFIG.")
    st.stop()

try:
    if override_file is not None:
        holdings_raw = pd.read_csv(override_file)
    else:
        holdings_raw = load_holdings_from_csv(wave.holdings_csv_url)
except Exception as e:
    st.error(f"Error loading holdings for Wave {wave.code}: {e}")
    st.code(
        "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        language="text",
    )
    st.stop()

if "Ticker" not in holdings_raw.columns:
    st.error("Holdings CSV must contain a 'Ticker' column.")
    st.write("Columns found:", list(holdings_raw.columns))
    st.stop()

if "Weight" not in holdings_raw.columns:
    st.error("Holdings CSV must contain a 'Weight' column.")
    st.write("Columns found:", list(holdings_raw.columns))
    st.stop()

holdings = holdings_raw.copy()
holdings["Ticker"] = holdings["Ticker"].astype(str).str.upper()
holdings = holdings.sort_values("Weight", ascending=False).reset_index(drop=True)

holdings = add_quote_links(holdings)
top10 = holdings.head(10)

total_holdings = len(holdings)
largest_pos = float(top10["Weight"].iloc[0]) if not top10.empty else 0.0

equity_pct = 1.0
cash_pct = 0.0

# ---------------------------------------------------------
# Compute NAV/returns/alpha via engine
# ---------------------------------------------------------

try:
    stats = compute_wave_nav(wave, holdings)
except Exception as e:
    st.warning("NAV/return computation failed; showing holdings only.")
    st.code(
        "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        language="text",
    )
    stats = {}

wave_return = float(stats.get("wave_return", float("nan")))
bench_return = float(stats.get("benchmark_return", float("nan")))
alpha = float(stats.get("alpha", float("nan")))
nav = float(stats.get("nav", float("nan")))

bench_label = (
    benchmark_choice
    if benchmark_choice != "Use Wave benchmark"
    else wave.benchmark
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    f"""
    <h2 style="color:#E2ECFF;margin-bottom:0.2rem;">
        WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE
    </h2>
    <h1 style="color:#4DB8FF;margin-top:0rem;">
        {wave.name} (LIVE Demo)
    </h1>
    <p style="color:#9BA7C7;">
        Mode: <b>{mode}</b> · Benchmark: <b>{bench_label}</b> ·
        Style: <b>{style_profile}</b> · Type: <b>{type_profile}</b>
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------
# Top grid – Top 10 table + snapshot
# ---------------------------------------------------------

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
    st.caption(
        "Positive 1D moves will render in green, negatives in red once price overlays are wired in."
    )

with col_right:
    st.subheader("Wave snapshot")

    def fmt_pct(x):
        return "N/A" if pd.isna(x) else f"{x*100:.2f}%"

    def fmt_nav(x):
        return "N/A" if pd.isna(x) else f"${x:,.0f}"

    alpha_color = "#4AE17C" if not pd.isna(alpha) and alpha >= 0 else "#FF5C7B"

    snapshot_html = f"""
    <div style="background:#050C24;border-radius:12px;padding:16px;
                color:#E2ECFF;border:1px solid #1A2A4A;">
      <div style="font-size:0.85rem;text-transform:uppercase;color:#7C8BB5;">
        Wave Snapshot
      </div>
      <div style="font-size:1.1rem;font-weight:600;margin-bottom:4px;">
        {wave.code} – {wave.name}
      </div>
      <hr style="border-color:#1A2A4A;" />
      <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
        <div>
          <div style="color:#7C8BB5;">Total holdings</div>
          <div style="font-size:1.2rem;font-weight:600;">{total_holdings}</div>
        </div>
        <div>
          <div style="color:#7C8BB5;">Equity vs Cash</div>
          <div style="font-size:1.2rem;font-weight:600;">
            {equity_pct*100:.0f}% / {cash_pct*100:.0f}%
          </div>
        </div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.9rem;
                  margin-top:12px;">
        <div>
          <div style="color:#7C8BB5;">Largest position</div>
          <div style="font-size:1.2rem;font-weight:600;">
            {largest_pos*100:.1f}%
          </div>
        </div>
        <div>
          <div style="color:#7C8BB5;">Wave NAV (sim)</div>
          <div style="font-size:1.2rem;font-weight:600;">
            {fmt_nav(nav)}
          </div>
        </div>
      </div>
      <hr style="border-color:#1A2A4A;margin-top:12px;margin-bottom:8px;" />
      <div style="font-size:0.9rem;">
        <span style="color:#7C8BB5;">1-day Wave / Benchmark / Alpha:</span><br/>
        <span style="font-weight:600;">{fmt_pct(wave_return)}</span> /
        <span style="font-weight:600;">{fmt_pct(bench_return)}</span> /
        <span style="font-weight:600;color:{alpha_color};">
          {fmt_pct(alpha)}
        </span>
      </div>
    </div>
    """
    st.markdown(snapshot_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# Second grid – weight bars, sector view, rank curve
# ---------------------------------------------------------

col_a, col_b, col_c = st.columns([1.4, 1.4, 1.2])

with col_a:
    st.subheader("Top 10 profile – Wave weight distribution")
    if not top10.empty:
        chart_df = (top10.set_index("Ticker")["Weight"] * 100).sort_values(ascending=True)
        st.bar_chart(chart_df)
    else:
        st.info("No holdings available to plot.")

with col_b:
    st.subheader("Sector allocation")
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
        st.info("No 'Sector' column detected – add one to unlock sector allocation.")

with col_c:
    st.subheader("Holding rank curve")
    weights_sorted = holdings["Weight"].sort_values(ascending=False).reset_index(drop=True)
    weights_sorted.index = weights_sorted.index + 1  # rank
    st.line_chart(weights_sorted)

# ---------------------------------------------------------
# Bottom panels – description / internals
# ---------------------------------------------------------

st.markdown("---")

col_bottom_left, col_bottom_right = st.columns(2)

with col_bottom_left:
    st.subheader("Mode overview")
    st.markdown(
        """
        **Standard mode** keeps the Wave tightly aligned with its benchmark with controlled
        tracking error, strict beta discipline, and lower turnover.

        **Private Logic™ mode** layers on proprietary leadership, regime-switching, and
        SmartSafe™ overlays so the Wave can push for risk-adjusted alpha while staying
        inside institutional guardrails.
        """
    )

with col_bottom_right:
    st.subheader("Wave internals (from holdings snapshot)")
    st.markdown(
        """
        This panel will evolve into a full breadth and regime dashboard, driven directly
        from the uploaded holdings snapshot:

        - 1-day gain/loss breadth (advancers vs decliners)  
        - % of holdings above / below key moving-average bands  
        - Concentration metrics (top-5 / top-10 share of Wave weight)  
        - Regime flags that drive VectorOS and SmartSafe™ overrides  

        Add an explicit 1-day change or return column in the source sheet to unlock
        deeper Wave internals here.
        """
    )

st.caption(
    "Prototype console – not investment advice. For internal WAVES Intelligence™ use only."
)