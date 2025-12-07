import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# CONFIG & CONSTANTS
# ============================================================

LOOKBACK_MONTHS = 12          # performance window for curves & totals
BENCHMARK_TICKER = "SPY"
VIX_TICKER = "^VIX"

# For yfinance date range
TODAY = datetime.utcnow().date()
START_DATE = TODAY - timedelta(days=int(LOOKBACK_MONTHS * 31))

# Streamlit page config
st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=900)
def load_wave_weights(csv_path: str = "wave_weights.csv") -> pd.DataFrame:
    """Load wave weight definitions from CSV."""
    if not os.path.exists(csv_path):
        st.error(f"Missing weights file: {csv_path}")
        st.stop()
    df = pd.read_csv(csv_path)
    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]

    expected_cols = {"Wave", "Ticker", "Weight"}
    if not expected_cols.issubset(set(df.columns)):
        st.error(
            f"`wave_weights.csv` must contain columns: {expected_cols}. "
            f"Found: {set(df.columns)}"
        )
        st.stop()

    # Clean tickers and waves
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Wave"] = df["Wave"].astype(str).strip()

    # Make sure weights are numeric
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Weight"])
    return df


@st.cache_data(ttl=900)
def download_prices(tickers, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Download Adjusted Close prices for a list of tickers using yfinance.
    Returns a DataFrame indexed by date, columns tickers.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )

    if "Adj Close" not in data.columns:
        # yfinance sometimes returns a single series
        if isinstance(data, pd.Series):
            adj = data.to_frame(name=tickers[0])
        else:
            raise ValueError("Could not find 'Adj Close' in downloaded data.")
    else:
        adj = data["Adj Close"]

    if isinstance(adj, pd.Series):
        adj = adj.to_frame()

    # Ensure we only keep requested tickers (yfinance sometimes reorders)
    adj = adj[[t for t in tickers if t in adj.columns]]
    adj = adj.dropna(how="all")
    return adj


def compute_wave_performance(
    price_df: pd.DataFrame,
    weights: pd.Series,
    benchmark_prices: pd.Series,
) -> dict:
    """
    Compute daily returns, normalized NAV curves, and key metrics for a Wave.
    Returns dict with:
      - wave_nav (Series)
      - bench_nav (Series)
      - daily_wave_ret (Series)
      - daily_bench_ret (Series)
      - metrics (dict)
    """
    # Align weights to price columns
    weights = weights.reindex(price_df.columns).fillna(0.0)
    weights = weights / weights.sum()

    returns = price_df.pct_change().fillna(0.0)
    bench_ret = benchmark_prices.pct_change().fillna(0.0)

    # Wave daily return = sum_i w_i * r_i
    daily_wave_ret = (returns * weights).sum(axis=1)
    daily_bench_ret = bench_ret

    # NAV normalized to 100 at start
    wave_nav = (1 + daily_wave_ret).cumprod() * 100.0
    bench_nav = (1 + daily_bench_ret).cumprod() * 100.0

    # Compute metrics over full window
    total_ret_wave = wave_nav.iloc[-1] / wave_nav.iloc[0] - 1.0
    total_ret_bench = bench_nav.iloc[-1] / bench_nav.iloc[0] - 1.0
    alpha_vs_bench = total_ret_wave - total_ret_bench

    today_ret = float(daily_wave_ret.iloc[-1])

    # Max drawdown
    running_max = wave_nav.cummax()
    drawdown = wave_nav / running_max - 1.0
    max_drawdown = float(drawdown.min())

    metrics = {
        "total_return": float(total_ret_wave),
        "total_return_bench": float(total_ret_bench),
        "alpha_vs_bench": float(alpha_vs_bench),
        "today_return": today_ret,
        "max_drawdown": max_drawdown,
    }

    return {
        "wave_nav": wave_nav,
        "bench_nav": bench_nav,
        "daily_wave_ret": daily_wave_ret,
        "daily_bench_ret": daily_bench_ret,
        "metrics": metrics,
    }


@st.cache_data(ttl=900)
def fetch_vix_series(start_date: datetime, end_date: datetime) -> pd.Series:
    """Fetch VIX Adjusted Close series."""
    vix_prices = download_prices([VIX_TICKER], start_date, end_date)
    series = vix_prices.iloc[:, 0]
    return series


def vix_to_equity_exposure(vix_level: float) -> float:
    """
    Simplified VIX ladder -> equity exposure.
    You can tweak these cutoffs to match your production rules.
    """
    if vix_level is None or np.isnan(vix_level):
        return 0.90  # default if we can't read VIX

    # These brackets loosely approximate your risk ladder
    if vix_level <= 14:
        return 1.00
    if vix_level <= 20:
        return 0.90
    if vix_level <= 25:
        return 0.80
    if vix_level <= 30:
        return 0.70
    if vix_level <= 40:
        return 0.60
    return 0.50  # crisis


def render_top_holdings_html(df: pd.DataFrame) -> str:
    """
    Render top holdings as an HTML table with:
      - Google Finance links
      - Today % colored red/green
    """
    rows = []
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        w = row["Weight %"]
        chg = row["Today %"]

        url = f"https://www.google.com/finance/quote/{ticker}"
        color = "#29cc6a" if chg > 0 else "#ff4b4b" if chg < 0 else "#bbbbbb"

        rows.append(
            f"<tr>"
            f"<td style='padding:4px 8px'><a href='{url}' target='_blank' style='color:#5bc0ff;text-decoration:none'>{ticker}</a></td>"
            f"<td style='padding:4px 8px; text-align:right'>{w:0.2f}%</td>"
            f"<td style='padding:4px 8px; text-align:right; color:{color}'>{chg:+0.2f}%</td>"
            f"</tr>"
        )

    html = """
    <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
      <thead>
        <tr style="border-bottom:1px solid #444;">
          <th style="text-align:left; padding:4px 8px">Ticker (Google Finance)</th>
          <th style="text-align:right; padding:4px 8px">Weight %</th>
          <th style="text-align:right; padding:4px 8px">Today %</th>
        </tr>
      </thead>
      <tbody>
    """
    html += "\n".join(rows)
    html += "</tbody></table>"
    return html


def pretty_wave_name(wave_id: str) -> str:
    """Turn internal id like 'AI_Wave' into 'AI Wave'."""
    return wave_id.replace("_", " ").strip()


# ============================================================
# MAIN APP LOGIC
# ============================================================

weights_df = load_wave_weights()

available_waves = sorted(weights_df["Wave"].unique())
wave_id_to_pretty = {w: pretty_wave_name(w) for w in available_waves}
pretty_to_wave_id = {v: k for k, v in wave_id_to_pretty.items()}

# ----------------- SIDEBAR CONTROLS -------------------------

st.sidebar.markdown(
    "### 🌊 WAVES Intelligence™\n"
    "**Desktop engine + cloud snapshot**\n\n"
    "Institutional console – select one of your Waves."
)

selected_pretty = st.sidebar.selectbox(
    "Select Wave",
    [wave_id_to_pretty[w] for w in available_waves],
)

selected_wave = pretty_to_wave_id[selected_pretty]

st.sidebar.markdown("#### Risk Mode (label only)")
risk_mode = st.sidebar.radio(
    " ",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

target_beta = st.sidebar.slider(
    "Equity Exposure Target (%)",
    min_value=40,
    max_value=100,
    value=90,
    step=5,
)
target_beta_decimal = target_beta / 100.0

st.sidebar.markdown(
    f"<small>Target β ≈ {target_beta_decimal:.2f} • Remaining in SmartSafe™ cash.</small>",
    unsafe_allow_html=True,
)

# ----------------- HEADER / BADGES --------------------------

st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-row {
        display:flex;
        gap:0.4rem;
        margin-top:0.4rem;
        margin-bottom:1.1rem;
    }
    .badge {
        padding:0.18rem 0.5rem;
        border-radius:999px;
        font-size:0.70rem;
        font-weight:600;
        border:1px solid #2e7cff33;
        background:linear-gradient(90deg,#071726,#052a3e);
        color:#8fd3ff;
    }
    .live-pill {
        padding:0.18rem 0.6rem;
        border-radius:999px;
        font-size:0.70rem;
        font-weight:700;
        background:linear-gradient(90deg,#00ff99,#00b3ff);
        color:#020617;
        margin-left:0.4rem;
    }
    .metric-label {
        font-size:0.8rem;
        text-transform:uppercase;
        letter-spacing:0.08em;
        color:#9ca3af;
    }
    .metric-value {
        font-size:1.8rem;
        font-weight:600;
    }
    .metric-sub {
        font-size:0.8rem;
        color:#9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="big-title">WAVES Institutional Console</div>
    <div style="color:#9ca3af; margin-bottom:0.4rem;">
        Live / demo console for WAVES Intelligence™ — showing <b>{selected_pretty}</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

header_cols = st.columns(3)
with header_cols[0]:
    st.markdown(
        f"""
        <div class="badge-row">
          <div class="badge">LIVE ENGINE</div>
          <div class="badge">MULTI-WAVE</div>
          <div class="badge">ADAPTIVE INDEX WAVES™</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_cols[1]:
    st.markdown(
        f"<div style='text-align:center; margin-top:0.4rem; font-size:0.8rem; color:#9ca3af;'>"
        f"UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )
with header_cols[2]:
    st.markdown(
        "<div style='text-align:right; margin-top:0.4rem;'>"
        "<span class='live-pill'>LIVE ENGINE VIEW</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# DATA PIPELINE FOR SELECTED WAVE
# ============================================================

wave_weights = weights_df[weights_df["Wave"] == selected_wave].copy()
wave_weights = wave_weights.sort_values("Weight", ascending=False)

tickers = wave_weights["Ticker"].tolist()

# Make sure benchmark is included in price pulls
tickers_for_download = sorted(set(tickers + [BENCHMARK_TICKER]))

prices_all = download_prices(
    tickers_for_download,
    start_date=START_DATE,
    end_date=TODAY,
)

if BENCHMARK_TICKER not in prices_all.columns:
    st.error(f"Benchmark {BENCHMARK_TICKER} prices not found.")
    st.stop()

benchmark_prices = prices_all[BENCHMARK_TICKER].dropna()
wave_prices = prices_all[[t for t in tickers if t in prices_all.columns]].dropna(how="all")

if wave_prices.empty:
    st.error("No price data found for this Wave's universe.")
    st.stop()

# Align on common index
common_index = wave_prices.index.intersection(benchmark_prices.index)
wave_prices = wave_prices.loc[common_index]
benchmark_prices = benchmark_prices.loc[common_index]

perf = compute_wave_performance(
    price_df=wave_prices,
    weights=wave_weights.set_index("Ticker")["Weight"],
    benchmark_prices=benchmark_prices,
)

wave_nav = perf["wave_nav"]
bench_nav = perf["bench_nav"]
daily_wave_ret = perf["daily_wave_ret"]
metrics = perf["metrics"]

# VIX data & equity exposure
vix_series = fetch_vix_series(START_DATE, TODAY)
vix_level = float(vix_series.iloc[-1]) if len(vix_series) else float("nan")
equity_exposure = vix_to_equity_exposure(vix_level)
cash_buffer = 1.0 - equity_exposure

# Today's per-ticker move (last two closes)
returns_all = prices_all.pct_change().fillna(0.0)
returns_all = returns_all.loc[common_index]

today_ticker_rets = returns_all.iloc[-1].reindex(tickers).fillna(0.0)

top_df = pd.DataFrame(
    {
        "Ticker": tickers,
        "Weight %": wave_weights.set_index("Ticker")["Weight"].reindex(tickers).fillna(0.0)
        * equity_exposure
        * 100.0,
        "Today %": today_ticker_rets.values * 100.0,
    }
)
top_df = top_df.sort_values("Weight %", ascending=False).head(10)

# ============================================================
# TOP METRICS ROW
# ============================================================

m_cols = st.columns(4)
metric_fmt = lambda x: f"{x*100:0.2f}%"

with m_cols[0]:
    st.markdown(
        f"<div class='metric-label'>Total Return (lookback)</div>"
        f"<div class='metric-value'>{metric_fmt(metrics['total_return'])}</div>"
        f"<div class='metric-sub'>{LOOKBACK_MONTHS}m window • "
        f"SPY: {metric_fmt(metrics['total_return_bench'])}</div>",
        unsafe_allow_html=True,
    )
with m_cols[1]:
    st.markdown(
        f"<div class='metric-label'>Today</div>"
        f"<div class='metric-value'>{metric_fmt(metrics['today_return'])}</div>"
        f"<div class='metric-sub'>Daily move (Wave)</div>",
        unsafe_allow_html=True,
    )
with m_cols[2]:
    st.markdown(
        f"<div class='metric-label'>Max Drawdown</div>"
        f"<div class='metric-value'>{metric_fmt(metrics['max_drawdown'])}</div>"
        f"<div class='metric-sub'>Worst peak-to-trough over window</div>",
        unsafe_allow_html=True,
    )
with m_cols[3]:
    st.markdown(
        f"<div class='metric-label'>Alpha vs SPY</div>"
        f"<div class='metric-value'>{metric_fmt(metrics['alpha_vs_bench'])}</div>"
        f"<div class='metric-sub'>Benchmark: SPY (Adj Close)</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================================
# MAIN BODY: PERFORMANCE CURVE + HOLDINGS PANEL
# ============================================================

left, right = st.columns([2.3, 1.7])

with left:
    st.subheader("Performance Curve")

    perf_df = pd.DataFrame(
        {
            selected_pretty: wave_nav,
            "SPY": bench_nav,
        }
    )
    st.line_chart(perf_df)

    st.caption(
        f"Curve normalized to 100 at start of the {LOOKBACK_MONTHS}m window. "
        "Source: Yahoo Finance (Adj Close)."
    )

with right:
    st.subheader("Holdings, Weights & Risk")

    st.markdown("**Top 10 Positions — Google Finance links (Bloomberg-style)**")

    html_table = render_top_holdings_html(top_df)
    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("#### Exposure Snapshot")
    st.markdown(
        f"- **Equity exposure** (after VIX adjustment): **{equity_exposure*100:0.1f}%**  \n"
        f"- **SmartSafe™ cash buffer**: **{cash_buffer*100:0.1f}%**  \n"
        f"- **Target β slider:** {target_beta_decimal:0.2f}  \n"
        f"- **Risk mode (label):** {risk_mode}"
    )

    if not np.isnan(vix_level):
        st.markdown(
            f"**VIX level:** {vix_level:0.2f} (mapped via ladder → equity exposure)"
        )
    else:
        st.markdown("**VIX level:** n/a (fallback exposure in use)")

# ============================================================
# MINI SPX & VIX CHARTS
# ============================================================

st.markdown("---")
st.markdown("### Market Context — SPY & VIX Mini Charts")

ctx_cols = st.columns(2)

# Mini SPY nav (same bench_nav but last 6 months)
recent_window = bench_nav.last("180D") if hasattr(bench_nav, "last") else bench_nav.tail(
    180
)

with ctx_cols[0]:
    st.markdown("**SPY (Benchmark)**")
    st.line_chart(recent_window)
    if len(recent_window) > 1:
        spy_recent_ret = recent_window.iloc[-1] / recent_window.iloc[0] - 1.0
        st.caption(f"Last ~6m: {spy_recent_ret*100:0.2f}%")

with ctx_cols[1]:
    st.markdown("**VIX (Implied volatility)**")
    vix_recent = vix_series.tail(180) if len(vix_series) else vix_series
    if len(vix_recent):
        st.line_chart(vix_recent)
        st.caption(f"Latest VIX: {vix_level:0.2f}")
    else:
        st.info("VIX data not available for this window.")

# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.markdown(
    """
    <hr/>
    <small>
    WAVES Institutional Console — demo view only. Returns & metrics are based on public
    market data via Yahoo Finance and do not represent live trading or an offer of
    advisory services.
    </small>
    """,
    unsafe_allow_html=True,
)
