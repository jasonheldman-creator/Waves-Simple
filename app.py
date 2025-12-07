# app.py  -- WAVES Institutional Console (Cloud + Desktop single script)
# ----------------------------------------------------------------------
# - Uses wave_weights.csv for universes
# - Pulls live prices with yfinance (no separate engine script needed)
# - Computes portfolio & benchmark curves on the fly
# - Shows performance metrics + top holdings with Google Finance links
# - Colors holdings green/red based on latest % change

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ---------- CONFIG & CONSTANTS -------------------------------------------------

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mapping from internal wave codes (in CSV) to pretty names
WAVE_PRETTY_NAMES = {
    "AI_Wave": "AI Leaders Wave",
    "Growth_Wave": "Growth Wave",
    "Income_Wave": "Income Wave",
    "SmallCapGrowth_Wave": "Small-Cap Growth Wave",
    "SMIDGrowth_Wave": "Small-Mid Growth Wave",
    "FuturePower_Wave": "Future Power & Energy Wave",
    "CryptoIncome_Wave": "Crypto Income Wave",
    "Quantum_Wave": "Quantum Computing Wave",
    "CleanTransit_Wave": "Clean Transit & Infrastructure Wave",
}

# Reverse lookup: pretty -> internal
PRETTY_TO_INTERNAL = {v: k for k, v in WAVE_PRETTY_NAMES.items()}

# Benchmark mapping (can refine per-wave later)
WAVE_BENCHMARK = {
    "AI_Wave": "SPY",
    "Growth_Wave": "SPY",
    "Income_Wave": "SPY",
    "SmallCapGrowth_Wave": "IWM",
    "SMIDGrowth_Wave": "VO",
    "FuturePower_Wave": "XLE",
    "CryptoIncome_Wave": "BTC-USD",  # example; adjust as needed
    "Quantum_Wave": "QQQ",
    "CleanTransit_Wave": "XTN",
}


# ---------- STYLE --------------------------------------------------------------

DARK_CSS = """
<style>
/* Global dark theme tweaks */
body, .stApp {
    background-color: #050910;
    color: #F5F7FB;
}

/* Top hero bar */
.waves-hero {
    padding: 0.75rem 1.25rem;
    border-radius: 999px;
    border: 1px solid #1b2335;
    background: linear-gradient(90deg, #071424, #04101c);
    box-shadow: 0 0 20px rgba(0, 255, 170, 0.15);
}

/* KPI cards */
.waves-kpi {
    padding: 0.8rem 1.1rem;
    border-radius: 12px;
    background: #070c14;
    border: 1px solid #101827;
}
.waves-kpi-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9CA3AF;
}
.waves-kpi-value {
    font-size: 1.4rem;
    font-weight: 600;
}

/* Section titles */
h2, h3 {
    color: #F9FAFB;
}

/* Holdings table */
.waves-table thead tr th {
    background-color: #020617;
}

/* Sidebar header */
.sidebar-brand {
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.4rem;
}
.sidebar-sub {
    font-size: 0.75rem;
    color: #9CA3AF;
}

/* Little pill labels */
.badge-pill {
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border: 1px solid #10b98133;
    color: #A7F3D0;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------- HELPERS ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_universe(path: str = "wave_weights.csv") -> pd.DataFrame:
    """Load the master Wave universe."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"wave_weights.csv not found in repo root. Expected at: {os.path.abspath(path)}"
        )
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Expected minimal columns
    expected = {"Ticker", "Wave"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            "wave_weights.csv must contain at least columns: 'Ticker', 'Wave' "
            f"(found: {list(df.columns)})"
        )

    # Accept either Weight or Weight % or weight
    if "Weight" not in df.columns:
        for alt in ["Weight %", "WeightPct", "WeightPct%", "weight", "weight_pct"]:
            if alt in df.columns:
                df["Weight"] = df[alt]
                break
    if "Weight" not in df.columns:
        # If nothing, equal weight within each wave
        df["Weight"] = 1.0

    # Convert to fractions if looks like % values
    if df["Weight"].max() > 1.5:
        df["Weight"] = df["Weight"] / 100.0

    # Normalise per wave
    df["Weight"] = df.groupby("Wave")["Weight"].transform(
        lambda s: s / s.sum() if s.sum() != 0 else s
    )

    df["Ticker"] = df["Ticker"].str.upper().str.strip()

    return df


@st.cache_data(show_spinner=False)
def fetch_prices(tickers, period="6mo") -> pd.DataFrame:
    """Fetch OHLCV for tickers via yfinance and return a DataFrame of closes."""
    if isinstance(tickers, (list, tuple, set)):
        tickers = sorted(set(tickers))
    data = yf.download(
        tickers,
        period=period,
        interval="1d",
        group_by="column",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    # Handle both multi-index and single-index cases
    if isinstance(data.columns, pd.MultiIndex):
        # Try Adj Close first, then Close
        lvl1 = set(data.columns.get_level_values(0))
        if "Adj Close" in lvl1:
            close = data["Adj Close"].copy()
        elif "Close" in lvl1:
            close = data["Close"].copy()
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' level found in price data.")
    else:
        cols = list(data.columns)
        if "Adj Close" in cols:
            close = data["Adj Close"].to_frame()
        elif "Close" in cols:
            close = data["Close"].to_frame()
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' column found in price data.")
        close.columns = tickers  # single ticker case

    close = close.dropna(how="all")
    return close


def build_portfolio_series(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Create a normalised portfolio equity curve from holdings + price history."""
    tickers = [t for t in holdings["Ticker"] if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)

    weights = holdings.set_index("Ticker")["Weight"].loc[tickers]

    # Normalise each ticker to 1.0 at start, then weight & sum
    norm_prices = prices[tickers] / prices[tickers].iloc[0]
    port = (norm_prices * weights).sum(axis=1)
    return port


def compute_metrics(port: pd.Series, bench: pd.Series | None) -> dict:
    """Compute total return, today's move, max drawdown, alpha vs benchmark."""
    if port.empty:
        return {"total": None, "today": None, "maxdd": None, "alpha": None}

    total_ret = port.iloc[-1] / port.iloc[0] - 1.0

    if len(port) >= 2:
        today_ret = port.iloc[-1] / port.iloc[-2] - 1.0
    else:
        today_ret = None

    running_max = port.cummax()
    dd = port / running_max - 1.0
    max_dd = dd.min()

    alpha = None
    if bench is not None and not bench.empty:
        bench_norm = bench / bench.iloc[0]
        common = port.index.intersection(bench_norm.index)
        if len(common) > 1:
            port_c = port.loc[common] / port.loc[common].iloc[0]
            bench_c = bench_norm.loc[common]
            alpha = port_c.iloc[-1] - bench_c.iloc[-1]

    return {"total": total_ret, "today": today_ret, "maxdd": max_dd, "alpha": alpha}


def fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:0.2f}%"


def ticker_to_google_link(ticker: str) -> str:
    """
    Build a Google Finance link.
    We won't guess exchange perfectly; Google is forgiving.
    """
    return f"https://www.google.com/finance/quote/{ticker}:NASDAQ"


def style_top_holdings(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply red/green styling to the Change % column."""
    def color_change(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #22c55e; font-weight: 600;"
        elif val < 0:
            return "color: #ef4444; font-weight: 600;"
        return ""

    styler = (
        df.style.format({"Weight %": "{:0.2f}%", "Change %": "{:0.2f}%"})
        .applymap(color_change, subset=["Change %"])
    )
    return styler


# ---------- LOAD DATA ----------------------------------------------------------

try:
    universe = load_universe()
except Exception as e:
    st.error(
        f"⚠️ Could not load `wave_weights.csv`: {e}\n\n"
        "Place `wave_weights.csv` in the root of the repo and redeploy."
    )
    st.stop()

# Filter to only the waves we know / want
available_wave_codes = sorted(set(universe["Wave"]))
valid_codes = [w for w in available_wave_codes if w in WAVE_PRETTY_NAMES]
if not valid_codes:
    st.error(
        "No recognised waves found in `wave_weights.csv`.\n\n"
        f"Expected one of: {list(WAVE_PRETTY_NAMES.keys())}, "
        f"found: {available_wave_codes}"
    )
    st.stop()

pretty_names = [WAVE_PRETTY_NAMES[w] for w in valid_codes]


# ---------- SIDEBAR ------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'>WAVES Intelligence™</div>"
        "<div class='sidebar-sub'>Institutional console — live engine view</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Desktop Engine + Cloud Snapshot**")
    st.caption(
        "This console pulls live prices via yfinance and builds each Wave’s equity "
        "curve on the fly. No local CSV performance logs required."
    )

    selected_pretty = st.selectbox("Select Wave", pretty_names, index=0)
    selected_wave = PRETTY_TO_INTERNAL[selected_pretty]

    risk_mode = st.radio(
        "Risk Mode *(label only)*",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    exposure = st.slider("Equity Exposure", min_value=0, max_value=100, value=90, step=5)
    st.caption(
        f"Target β ≈ 0.90 | Cash buffer: {100 - exposure}%  \n"
        "These controls label the view — actual allocations are set in the engine."
    )

    st.markdown("---")
    st.caption(
        "Tip: For a **live demo**, leave this console open while the market is open. "
        "It will refresh pricing on each interaction."
    )


# ---------- TOP BAR / HERO -----------------------------------------------------

# Fetch SPX & VIX quick quotes
try:
    spx = yf.Ticker("^GSPC").history(period="1d")["Close"]
    vix = yf.Ticker("^VIX").history(period="1d")["Close"]
    spx_val = float(spx.iloc[-1]) if not spx.empty else None
    vix_val = float(vix.iloc[-1]) if not vix.empty else None
except Exception:
    spx_val = vix_val = None

utc_now = datetime.now(timezone.utc).replace(microsecond=0)

col_hero = st.container()
with col_hero:
    st.markdown(
        """
        <div class="waves-hero">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
            <div>
              <div style="font-size:0.75rem; letter-spacing:0.18em; text-transform:uppercase; color:#6B7280;">
                WAVES INSTITUTIONAL CONSOLE
              </div>
              <div style="font-size:1.15rem; font-weight:600; color:#E5E7EB;">
                Live / demo console for <span style="color:#22c55e;">WAVES Intelligence™</span>
              </div>
              <div style="font-size:0.8rem; color:#9CA3AF;">
                Adaptive Index Waves™ — {wave_name} | Mode: {mode} | UTC: {utc}
              </div>
            </div>
            <div style="display:flex; gap:0.75rem; align-items:center;">
              <span class="badge-pill">Live Engine</span>
              <span class="badge-pill">Multi-Wave</span>
              <span class="badge-pill">Adaptive Index Waves™</span>
            </div>
          </div>
        </div>
        """.format(
            wave_name=selected_pretty,
            mode=risk_mode,
            utc=utc_now.isoformat().replace("T", " "),
        ),
        unsafe_allow_html=True,
    )

    # Quick SPX / VIX strip
    if spx_val is not None and vix_val is not None:
        st.markdown(
            f"""
            <div style="margin-top:0.6rem; display:flex; gap:1rem; align-items:center;">
              <div style="font-size:0.8rem; color:#9CA3AF;">SPX</div>
              <div style="font-size:0.9rem; color:#22c55e; font-weight:600;">{spx_val:,.2f}</div>
              <div style="font-size:0.8rem; color:#9CA3AF; margin-left:1.5rem;">VIX</div>
              <div style="font-size:0.9rem; color:#f97316; font-weight:600;">{vix_val:0.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("")


# ---------- LOAD HOLDINGS & PRICES --------------------------------------------

wave_holdings = universe[universe["Wave"] == selected_wave].copy()
wave_holdings = wave_holdings.sort_values("Weight", ascending=False).reset_index(drop=True)

tickers = wave_holdings["Ticker"].tolist()
benchmark_ticker = WAVE_BENCHMARK.get(selected_wave, "SPY")

all_price_tickers = sorted(set(tickers + [benchmark_ticker]))

with st.spinner("Fetching live prices…"):
    prices = fetch_prices(all_price_tickers, period="6mo")

if prices.empty:
    st.error("No price data returned. Check ticker symbols or yfinance availability.")
    st.stop()

# Split portfolio vs benchmark
bench_series = prices[benchmark_ticker].dropna() if benchmark_ticker in prices.columns else None
port_series = build_portfolio_series(wave_holdings, prices)

metrics = compute_metrics(port_series, bench_series)


# ---------- KPI ROW -----------------------------------------------------------

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(
        "<div class='waves-kpi'><div class='waves-kpi-label'>Total Return (logs)</div>"
        f"<div class='waves-kpi-value'>{fmt_pct(metrics['total'])}</div></div>",
        unsafe_allow_html=True,
    )
with kpi_col2:
    st.markdown(
        "<div class='waves-kpi'><div class='waves-kpi-label'>Today</div>"
        f"<div class='waves-kpi-value'>{fmt_pct(metrics['today'])}</div></div>",
        unsafe_allow_html=True,
    )
with kpi_col3:
    st.markdown(
        "<div class='waves-kpi'><div class='waves-kpi-label'>Max Drawdown</div>"
        f"<div class='waves-kpi-value'>{fmt_pct(metrics['maxdd'])}</div></div>",
        unsafe_allow_html=True,
    )
with kpi_col4:
    st.markdown(
        "<div class='waves-kpi'><div class='waves-kpi-label'>Alpha vs Benchmark</div>"
        f"<div class='waves-kpi-value'>{fmt_pct(metrics['alpha'])}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------- MAIN LAYOUT: CHART + HOLDINGS -------------------------------------

left_col, right_col = st.columns((2.2, 1.8))

# --- Performance Curve ---
with left_col:
    st.subheader("Performance Curve")

    if port_series.empty:
        st.info(
            "No performance history available for this Wave yet. "
            "Once price history is available for its holdings, the equity curve will appear here."
        )
    else:
        chart_df = pd.DataFrame(
            {
                "Portfolio": port_series / port_series.iloc[0] * 100,
            }
        )
        if bench_series is not None and not bench_series.empty:
            bench_norm = bench_series / bench_series.iloc[0] * 100
            chart_df["Benchmark"] = bench_norm.reindex(chart_df.index, method="ffill")

        st.line_chart(chart_df, height=320, use_container_width=True)

# --- Holdings & Risk ---
with right_col:
    st.subheader("Holdings, Weights & Risk")

    # Compute latest price & daily change for each holding
    latest_closes = prices.iloc[-1]
    if len(prices) >= 2:
        prev_closes = prices.iloc[-2]
        daily_change = latest_closes / prev_closes - 1.0
    else:
        daily_change = pd.Series(index=prices.columns, dtype=float)

    top = wave_holdings.copy()
    top["Weight %"] = top["Weight"] * 100
    top["Live Price"] = top["Ticker"].map(latest_closes.to_dict())
    top["Change %"] = top["Ticker"].map(daily_change.to_dict()) * 100

    # Build Google links in a separate display column
    top_display = top.copy()
    top_display["Ticker"] = top_display["Ticker"].apply(
        lambda t: f"<a href='{ticker_to_google_link(t)}' target='_blank'>{t}</a>"
    )

    top10 = top_display.head(10)[["Ticker", "Weight %", "Change %"]].reset_index(drop=True)

    st.caption("Top 10 Positions — Google Finance Links (Bloomberg-style red/green)")
    styled = style_top_holdings(top10)
    styled = styled.hide(axis="index")
    styled = styled.set_table_attributes('class="waves-table"').format(escape=False)
    st.write(styled.to_html(), unsafe_allow_html=True)

    st.markdown("")
    with st.expander("Full Wave universe table"):
        full = top_display[["Ticker", "Weight %", "Change %", "Live Price"]].reset_index(
            drop=True
        )
        full_styled = style_top_holdings(full)
        full_styled = full_styled.hide(axis="index").set_table_attributes(
            'class="waves-table"'
        ).format(escape=False)
        st.write(full_styled.to_html(), unsafe_allow_html=True)

st.markdown("---")

st.caption(
    "This console is a **single self-contained app**: it reads `wave_weights.csv`, "
    "pulls prices via yfinance, and computes each Wave’s live equity curve on demand. "
    "No background engine or CSV performance logs required on Streamlit Cloud."
)
