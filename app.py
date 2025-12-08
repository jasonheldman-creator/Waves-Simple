import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ================================
# Streamlit config & styling
# ================================
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #050816;
        color: #f5f6fa;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9fa6b2 !important;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e5f9ff !important;
    }
    .section-card {
        background: #0b1020;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #1f2937;
        box-shadow: 0 0 25px rgba(0, 0, 0, 0.45);
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 600;
        color: #e5f9ff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #9fa6b2;
    }
    .wave-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border: 1px solid #1f2937;
        background: rgba(15,23,42,0.9);
        color: #a5b4fc;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border: 1px solid #1f2937;
        margin-left: 0.4rem;
        color: #6ee7b7;
    }
    .alpha-positive {
        color: #22c55e !important;
        font-weight: 600;
    }
    .alpha-negative {
        color: #f97373 !important;
        font-weight: 600;
    }
    table.top10-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    table.top10-table thead tr {
        background-color: #111827;
    }
    table.top10-table th, table.top10-table td {
        padding: 0.35rem 0.55rem;
        border-bottom: 1px solid #1f2937;
        text-align: left;
    }
    table.top10-table th {
        font-weight: 600;
        color: #e5f9ff;
    }
    table.top10-table td {
        color: #f9fafb;
    }
    table.top10-table a {
        color: #38bdf8;
        text-decoration: none;
        font-weight: 500;
    }
    table.top10-table a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================
# Constants & paths
# ================================
BASE_DIR = Path(".")
WEIGHTS_FILE = BASE_DIR / "wave_weights.csv"

MODES = {
    "Standard": {"beta_target": 0.90, "drift_annual": 0.07},
    "Alpha-Minus-Beta": {"beta_target": 0.80, "drift_annual": 0.06},
    "Private Logic™": {"beta_target": 1.05, "drift_annual": 0.09},
}

BENCHMARK_MAP = {
    "S&P 500 Wave": "^GSPC",
    "Growth Wave": "QQQ",
    "Infinity Wave": "^GSPC",
    "Income Wave": "^GSPC",
    "Future Power & Energy Wave": "XLE",
    "Crypto Income Wave": "BTC-USD",
    "Quantum Computing Wave": "QQQ",
    "Clean Transit-Infrastructure Wave": "IGE",
    "Small/Mid Growth Wave": "IWM",
}


# ================================
# Helpers
# ================================
def format_pct(x: float | None, decimals: int = 2) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x:.{decimals}f}%"


def google_url(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if not t or t == "NAN":
        return ""
    return f"https://www.google.com/finance/quote/{t}"


@st.cache_data(show_spinner=False)
def get_spx_vix_tiles() -> dict:
    data = {
        "SPX": {"label": "S&P 500", "value": None, "change": None},
        "VIX": {"label": "VIX", "value": None, "change": None},
    }
    try:
        for symbol, key in [("^GSPC", "SPX"), ("^VIX", "VIX")]:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                continue
            last_close = float(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
                change_pct = (last_close / prev_close - 1.0) * 100.0
            else:
                change_pct = 0.0
            data[key]["value"] = last_close
            data[key]["change"] = change_pct
    except Exception:
        pass
    return data


@st.cache_data(show_spinner=False)
def load_weights() -> pd.DataFrame:
    """
    Load wave_weights.csv and normalize:
      - wave
      - ticker
      - weight (sums to 1 within each wave)
    """
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"Missing weights file: {WEIGHTS_FILE}")

    df = pd.read_csv(WEIGHTS_FILE)
    if df.empty:
        raise ValueError("wave_weights.csv is empty")

    cols = {c.lower(): c for c in df.columns}

    wave_col = cols.get("wave") or cols.get("portfolio")
    if wave_col is None:
        raise ValueError("wave_weights.csv must have a 'wave' or 'portfolio' column")

    ticker_col = cols.get("ticker") or cols.get("symbol")
    if ticker_col is None:
        raise ValueError("wave_weights.csv must have a 'ticker' or 'symbol' column")

    weight_candidates = [
        "weight",
        "weight_pct",
        "weight_percent",
        "target_weight",
        "portfolio_weight",
    ]
    weight_col = None
    for cand in weight_candidates:
        if cand in cols:
            weight_col = cols[cand]
            break

    if weight_col is None:
        df["__weight__"] = 1.0
        weight_col = "__weight__"

    df = df.rename(
        columns={wave_col: "wave", ticker_col: "ticker", weight_col: "raw_weight"}
    )
    df["wave"] = df["wave"].astype(str)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["raw_weight"] = df["raw_weight"].astype(float)

    df["weight"] = df.groupby("wave")["raw_weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else 1.0 / len(x)
    )

    return df[["wave", "ticker", "weight"]]


def get_benchmark_for_wave(wave_name: str) -> str:
    return BENCHMARK_MAP.get(wave_name, "^GSPC")


@st.cache_data(show_spinner=True)
def fetch_prices(tickers: tuple[str, ...], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers between start & end.
    Returns DataFrame: index=date, columns=tickers.
    """
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=list(tickers),
        start=start.date(),
        end=(end + timedelta(days=1)).date(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # MultiIndex vs single
    if isinstance(data.columns, pd.MultiIndex):
        cols = []
        for t in tickers:
            if (t, "Close") in data.columns:
                cols.append(data[(t, "Close")].rename(t))
        if not cols:
            return pd.DataFrame()
        prices = pd.concat(cols, axis=1)
    else:
        prices = data["Close"].to_frame()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all")
    return prices


def compute_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


@st.cache_data(show_spinner=True)
def build_wave_performance(
    wave_name: str,
    mode_label: str,
    history_days: int = 365,
) -> pd.DataFrame:
    """
    Full engine for one Wave, in-memory:
      - Uses full basket from wave_weights.csv
      - Fetches prices + benchmark prices
      - Computes daily portfolio_return & benchmark_return
      - Computes α_1d, α_30d, α_60d, rolling returns.
    """
    weights_df = load_weights()
    wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
    if wave_weights.empty:
        raise ValueError(f"No rows for wave: {wave_name} in wave_weights.csv")

    tickers = sorted(wave_weights["ticker"].unique().tolist())
    bench_ticker = get_benchmark_for_wave(wave_name)

    end = datetime.utcnow()
    start = end - timedelta(days=history_days)

    basket_prices = fetch_prices(tuple(tickers), start, end)
    bench_prices = fetch_prices((bench_ticker,), start, end)

    if basket_prices.empty or bench_prices.empty:
        raise ValueError(f"Missing price history for wave {wave_name} or benchmark {bench_ticker}")

    basket_rets = compute_returns_from_prices(basket_prices)
    bench_rets = compute_returns_from_prices(bench_prices)

    common_dates = basket_rets.index.intersection(bench_rets.index)
    basket_rets = basket_rets.loc[common_dates]
    bench_rets = bench_rets.loc[common_dates]

    weight_map = {row["ticker"]: row["weight"] for _, row in wave_weights.iterrows()}
    aligned_weights = np.array([weight_map[t] for t in basket_rets.columns])

    port_ret = basket_rets.values @ aligned_weights

    df = pd.DataFrame(
        {
            "date": common_dates,
            "portfolio_return": port_ret,
            "benchmark_return": bench_rets.iloc[:, 0].values,
        }
    )

    # Add alpha windows & rolling returns
    mode_cfg = MODES.get(mode_label, MODES["Standard"])
    beta_target = mode_cfg["beta_target"]

    df = df.sort_values("date").reset_index(drop=True)
    df["alpha_1d"] = df["portfolio_return"] - beta_target * df["benchmark_return"]
    df["alpha_30d"] = df["alpha_1d"].rolling(30, min_periods=1).sum()
    df["alpha_60d"] = df["alpha_1d"].rolling(60, min_periods=1).sum()

    df["ret_30d"] = (1.0 + df["portfolio_return"]).rolling(30, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )
    df["ret_60d"] = (1.0 + df["portfolio_return"]).rolling(60, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )
    df["bench_30d"] = (1.0 + df["benchmark_return"]).rolling(30, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )
    df["bench_60d"] = (1.0 + df["benchmark_return"]).rolling(60, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )

    return df


def compute_drawdown(df: pd.DataFrame) -> tuple[float, pd.Series]:
    df = df.copy().sort_values("date")
    cum = (1.0 + df["portfolio_return"]).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    max_dd = dd.min() * 100.0
    return float(max_dd), dd


def compute_standard_matrix_row(wave_name: str) -> dict:
    """
    Standard Mode snapshot for a Wave:
      - 1D Return, 1D Alpha, 30D Alpha, 60D Alpha
      - Realized β
      - Max Drawdown
      - Since-inception Return / Benchmark / Excess
    """
    perf_df = build_wave_performance(wave_name, "Standard", history_days=365)

    if perf_df.empty:
        return {
            "Wave": wave_name,
            "1D Return": np.nan,
            "1D Alpha": np.nan,
            "30D Alpha": np.nan,
            "60D Alpha": np.nan,
            "Realized β": np.nan,
            "Max Drawdown": np.nan,
            "SI Return": np.nan,
            "SI Benchmark": np.nan,
            "SI Excess": np.nan,
        }

    perf_df = perf_df.sort_values("date").reset_index(drop=True)
    latest = perf_df.iloc[-1]

    one_day_ret = latest["portfolio_return"] * 100.0
    one_day_alpha = latest["alpha_1d"] * 100.0
    alpha_30d = latest["alpha_30d"] * 100.0
    alpha_60d = latest["alpha_60d"] * 100.0

    if np.var(perf_df["benchmark_return"]) > 0:
        realized_beta = np.cov(
            perf_df["portfolio_return"], perf_df["benchmark_return"]
        )[0, 1] / np.var(perf_df["benchmark_return"])
    else:
        realized_beta = np.nan

    max_dd, _ = compute_drawdown(perf_df)

    cum_port = (1.0 + perf_df["portfolio_return"]).cumprod()
    cum_bench = (1.0 + perf_df["benchmark_return"]).cumprod()
    si_ret = (cum_port.iloc[-1] - 1.0) * 100.0
    si_bench = (cum_bench.iloc[-1] - 1.0) * 100.0
    si_excess = si_ret - si_bench

    return {
        "Wave": wave_name,
        "1D Return": one_day_ret,
        "1D Alpha": one_day_alpha,
        "30D Alpha": alpha_30d,
        "60D Alpha": alpha_60d,
        "Realized β": realized_beta,
        "Max Drawdown": max_dd,
        "SI Return": si_ret,
        "SI Benchmark": si_bench,
        "SI Excess": si_excess,
    }


def compute_alpha_summary_for_wave(wave_name: str, mode_label: str) -> dict:
    """
    For the Alpha Capture Matrix (selected mode).
    """
    perf_df = build_wave_performance(wave_name, mode_label, history_days=365)
    if perf_df.empty:
        return {
            "Wave": wave_name,
            "Intraday α": np.nan,
            "1D α": np.nan,
            "30D α": np.nan,
            "60D α": np.nan,
            "1Y α": np.nan,
            "SI Alpha": np.nan,
            "SI Wave Return": np.nan,
            "SI Benchmark Return": np.nan,
        }

    perf_df = perf_df.sort_values("date").reset_index(drop=True)
    latest = perf_df.iloc[-1]

    intraday_alpha = latest["alpha_1d"]
    one_day_alpha = perf_df["alpha_1d"].iloc[-2] if len(perf_df) >= 2 else intraday_alpha
    alpha_30d = latest["alpha_30d"]
    alpha_60d = latest["alpha_60d"]

    window = perf_df["alpha_1d"].tail(252)
    alpha_1y = window.sum()

    cum_port = (1.0 + perf_df["portfolio_return"]).cumprod()
    cum_bench = (1.0 + perf_df["benchmark_return"]).cumprod()
    si_wave_ret = (cum_port.iloc[-1] - 1.0) * 100.0
    si_bench_ret = (cum_bench.iloc[-1] - 1.0) * 100.0
    si_alpha = perf_df["alpha_1d"].sum() * 100.0

    return {
        "Wave": wave_name,
        "Intraday α": intraday_alpha * 100.0,
        "1D α": one_day_alpha * 100.0,
        "30D α": alpha_30d * 100.0,
        "60D α": alpha_60d * 100.0,
        "1Y α": alpha_1y * 100.0,
        "SI Alpha": si_alpha,
        "SI Wave Return": si_wave_ret,
        "SI Benchmark Return": si_bench_ret,
    }


def load_top10_holdings(wave_name: str) -> pd.DataFrame:
    """
    Top 10 by weight from wave_weights.csv, with optional yfinance metadata.
    """
    weights_df = load_weights()
    wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
    if wave_weights.empty:
        return pd.DataFrame()

    # Re-normalize
    wave_weights["weight"] = wave_weights["weight"] / wave_weights["weight"].sum()
    wave_weights["weight_pct"] = wave_weights["weight"] * 100.0

    # Enrich with names/sectors (best effort)
    names = {}
    sectors = {}
    tickers = sorted(wave_weights["ticker"].unique().tolist())
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names[t] = info.get("shortName") or info.get("longName") or ""
            sectors[t] = info.get("sector") or ""
        except Exception:
            names[t] = ""
            sectors[t] = ""

    wave_weights["Name"] = wave_weights["ticker"].map(names)
    wave_weights["Sector"] = wave_weights["ticker"].map(sectors)

    out = wave_weights.sort_values("weight_pct", ascending=False).head(10)
    out = out[["ticker", "Name", "Sector", "weight_pct"]].copy()
    out["Ticker"] = out["ticker"].astype(str).str.upper()
    out["Weight%"] = out["weight_pct"].round(2).astype(str) + "%"
    out["TickerLink"] = out["Ticker"].apply(
        lambda t: f'<a href="{google_url(t)}" target="_blank">{t}</a>'
    )
    return out[["Ticker", "TickerLink", "Name", "Sector", "Weight%"]]


# ================================
# Sidebar
# ================================
weights_df_global = load_weights()
waves = sorted(weights_df_global["wave"].unique().tolist())

with st.sidebar:
    st.markdown("### WAVES Intelligence™")
    st.markdown("#### Institutional Console")

    selected_wave = st.selectbox("Select Wave", options=waves, index=0)

    selected_mode = st.radio(
        "Mode",
        options=list(MODES.keys()),
        index=0,
        help="Display logic adapts to Standard, Alpha-Minus-Beta, and Private Logic™ targets.",
    )

    st.markdown("---")
    st.markdown(
        "Engine is running **live in-app**:\n"
        "- Uses full basket + secondary basket from `wave_weights.csv`\n"
        "- Pulls prices via yfinance on demand\n"
        "- No external logs required"
    )


# ================================
# Compute perf for selected Wave/mode
# ================================
perf_df = build_wave_performance(selected_wave, selected_mode, history_days=365)
perf_df = perf_df.sort_values("date").reset_index(drop=True)

latest = perf_df.iloc[-1]
max_dd_pct, dd_series = compute_drawdown(perf_df)

mode_cfg = MODES[selected_mode]
beta_target = mode_cfg["beta_target"]
drift_annual = mode_cfg["drift_annual"]
drift_daily = drift_annual / 252.0 * 100.0

one_day_ret = latest["portfolio_return"] * 100.0
one_day_alpha = latest["alpha_1d"] * 100.0
alpha_30d = latest["alpha_30d"] * 100.0
alpha_60d = latest["alpha_60d"] * 100.0
ret_30d = latest["ret_30d"] * 100.0
ret_60d = latest["ret_60d"] * 100.0
bench_30d = latest["bench_30d"] * 100.0
bench_60d = latest["bench_60d"] * 100.0

cum_port = (1.0 + perf_df["portfolio_return"]).cumprod()
cum_bench = (1.0 + perf_df["benchmark_return"]).cumprod()
since_inc_ret = (cum_port.iloc[-1] - 1.0) * 100.0
since_inc_bench = (cum_bench.iloc[-1] - 1.0) * 100.0


# ================================
# Hero & tiles
# ================================
spx_vix = get_spx_vix_tiles()

st.markdown(
    f"""
    <div class="section-card" style="margin-bottom: 0.8rem;">
        <div class="hero-title">WAVES Intelligence™ Institutional Console</div>
        <div class="hero-subtitle">
            Live multi-wave, mode-aware monitoring — engine output visualized for institutional use.
        </div>
        <div style="margin-top: 0.6rem;">
            <span class="wave-badge">{selected_wave}</span>
            <span class="mode-badge">{selected_mode}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_spx, col_vix, col_mode = st.columns([1, 1, 1.2])

with col_spx:
    tile = spx_vix["SPX"]
    label = tile["label"]
    val = tile["value"]
    chg = tile["change"]
    chg_str = format_pct(chg) if chg is not None else "—"
    chg_class = "alpha-positive" if (chg or 0) >= 0 else "alpha-negative"
    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{f"{val:,.0f}" if val is not None else "—"}</div>
            <div class="{chg_class}" style="font-size:0.85rem;margin-top:0.15rem;">{chg_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_vix:
    tile = spx_vix["VIX"]
    label = tile["label"]
    val = tile["value"]
    chg = tile["change"]
    chg_str = format_pct(chg) if chg is not None else "—"
    chg_class = "alpha-positive" if (chg or 0) <= 0 else "alpha-negative"
    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{f"{val:,.2f}" if val is not None else "—"}</div>
            <div class="{chg_class}" style="font-size:0.85rem;margin-top:0.15rem;">{chg_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_mode:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-label">Mode Parameters</div>
            <div style="display:flex;gap:1.5rem;margin-top:0.3rem;">
                <div>
                    <div style="font-size:0.75rem;color:#9fa6b2;">β Target</div>
                    <div class="metric-value">{beta_target:.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem;color:#9fa6b2;">Drift (Annual)</div>
                    <div class="metric-value">{drift_annual*100:.1f}%</div>
                </div>
                <div>
                    <div style="font-size:0.75rem;color:#9fa6b2;">Expected Daily Drift</div>
                    <div class="metric-value">{drift_daily:.2f}%</div>
                </div>
            </div>
            <div style="font-size:0.75rem;color:#6b7280;margin-top:0.4rem;">
                Internal targets only — used for beta-adjusted alpha and drift framing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================
# Tabs
# ================================
tab_overview, tab_alpha, tab_std_matrix, tab_alpha_matrix, tab_top10, tab_logs = st.tabs(
    [
        "Overview",
        "Alpha Dashboard",
        "Standard Mode Matrix",
        "Alpha Capture Matrix (All Waves)",
        "Top 10 Holdings",
        "Engine Logs",
    ]
)


# -------- Overview --------
with tab_overview:
    col_left, col_right = st.columns([1.6, 1.4])

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Return & Alpha Strip</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**1D Return**")
            st.markdown(format_pct(one_day_ret))
        with c2:
            st.markdown("**1D Alpha (β-adjusted)**")
            cls = "alpha-positive" if one_day_alpha >= 0 else "alpha-negative"
            st.markdown(
                f"<span class='{cls}'>{format_pct(one_day_alpha)}</span>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown("**30D Alpha**")
            cls = "alpha-positive" if alpha_30d >= 0 else "alpha-negative"
            st.markdown(
                f"<span class='{cls}'>{format_pct(alpha_30d)}</span>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown("**60D Alpha**")
            cls = "alpha-positive" if alpha_60d >= 0 else "alpha-negative"
            st.markdown(
                f"<span class='{cls}'>{format_pct(alpha_60d)}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---", unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown("**30D Return vs Benchmark**")
            st.markdown(
                f"{format_pct(ret_30d)} | "
                f"<span style='color:#9fa6b2;'>{format_pct(bench_30d)}</span>",
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown("**60D Return vs Benchmark**")
            st.markdown(
                f"{format_pct(ret_60d)} | "
                f"<span style='color:#9fa6b2;'>{format_pct(bench_60d)}</span>",
                unsafe_allow_html=True,
            )
        with c7:
            st.markdown("**Since Inception vs Benchmark**")
            st.markdown(
                f"{format_pct(since_inc_ret)} | "
                f"<span style='color:#9fa6b2;'>{format_pct(since_inc_bench)}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card" style="margin-top:0.85rem;">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Performance Curve</div>', unsafe_allow_html=True)

        perf_chart_df = pd.DataFrame(
            {
                "Date": perf_df["date"],
                "Wave": cum_port.values,
                "Benchmark": cum_bench.values,
            }
        ).set_index("Date")

        st.line_chart(perf_chart_df)
        st.markdown(
            "<span style='font-size:0.75rem;color:#9fa6b2;'>"
            "Cumulative performance of the Wave vs its benchmark using daily returns."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Risk & Drawdown</div>', unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Max Drawdown**")
            st.markdown(format_pct(max_dd_pct))
        with col_r2:
            if np.var(perf_df["benchmark_return"]) > 0:
                realized_beta_sel = np.cov(
                    perf_df["portfolio_return"], perf_df["benchmark_return"]
                )[0, 1] / np.var(perf_df["benchmark_return"])
            else:
                realized_beta_sel = np.nan
            st.markdown("**Realized β**")
            st.markdown(
                f"{realized_beta_sel:.2f}" if not np.isnan(realized_beta_sel) else "—"
            )

        st.markdown("---", unsafe_allow_html=True)

        dd_chart_df = pd.DataFrame(
            {"Date": perf_df["date"], "Drawdown": dd_series.values}
        ).set_index("Date")
        st.area_chart(dd_chart_df)
        st.markdown(
            "<span style='font-size:0.75rem;color:#9fa6b2;'>"
            "Drawdown is computed from cumulative portfolio value; minimum value is shown as Max DD."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card" style="margin-top:0.85rem;">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Data Regime</div>', unsafe_allow_html=True)
        st.markdown("**Regime:** LIVE (yfinance engine)")
        st.markdown(
            "- Full basket + secondary basket from `wave_weights.csv`\n"
            "- Prices pulled live via yfinance on each refresh\n"
            "- β targets and drift assumptions always come from the selected mode."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# -------- Alpha Dashboard --------
with tab_alpha:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Alpha Timelines</div>', unsafe_allow_html=True)

    alpha_chart_df = pd.DataFrame(
        {
            "Date": perf_df["date"],
            "Alpha 1D": perf_df["alpha_1d"],
            "Alpha 30D": perf_df["alpha_30d"],
            "Alpha 60D": perf_df["alpha_60d"],
        }
    ).set_index("Date")

    st.line_chart(alpha_chart_df)
    st.markdown(
        "<span style='font-size:0.8rem;color:#9fa6b2;'>"
        "All alphas are β-adjusted excess returns using the mode's β target. "
        "30D and 60D are rolling sums of daily alpha."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -------- Standard Mode Matrix --------
with tab_std_matrix:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-label">Standard Mode Matrix — All Waves</div>',
        unsafe_allow_html=True,
    )

    rows = [compute_standard_matrix_row(w) for w in waves]
    matrix_df = pd.DataFrame(rows)

    if not matrix_df.empty:
        matrix_df = matrix_df.sort_values("60D Alpha", ascending=False).reset_index(
            drop=True
        )
        display_df = matrix_df.copy()

        for col in [
            "1D Return",
            "1D Alpha",
            "30D Alpha",
            "60D Alpha",
            "Max Drawdown",
            "SI Return",
            "SI Benchmark",
            "SI Excess",
        ]:
            display_df[col] = display_df[col].apply(lambda x: format_pct(x))

        display_df["Realized β"] = display_df["Realized β"].apply(
            lambda x: f"{x:.2f}" if not np.isnan(x) else "—"
        )

        st.markdown(
            "<span style='font-size:0.8rem;color:#9fa6b2;'>"
            "All metrics are computed in Standard mode (β≈0.90). "
            "Since-Inception metrics use each Wave's first available price date."
            "</span>",
            unsafe_allow_html=True,
        )
        st.dataframe(display_df, use_container_width=True)

        bar_df = matrix_df[["Wave", "60D Alpha"]].set_index("Wave")
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            "<span class='metric-label'>60D Alpha (Standard Mode, All Waves)</span>",
            unsafe_allow_html=True,
        )
        st.bar_chart(bar_df)
    else:
        st.markdown("No data available yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------- Alpha Capture Matrix --------
with tab_alpha_matrix:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-label">Alpha Capture Matrix — All Waves</div>',
        unsafe_allow_html=True,
    )

    summaries = [compute_alpha_summary_for_wave(w, selected_mode) for w in waves]
    summary_df = pd.DataFrame(summaries)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("60D α", ascending=False).reset_index(
            drop=True
        )

        display_df = summary_df.copy()
        for col in [
            "Intraday α",
            "1D α",
            "30D α",
            "60D α",
            "1Y α",
            "SI Alpha",
            "SI Wave Return",
            "SI Benchmark Return",
        ]:
            display_df[col] = display_df[col].apply(lambda x: format_pct(x))

        st.markdown(
            f"<span style='font-size:0.8rem;color:#9fa6b2;'>"
            f"All values are β-adjusted alpha captures and cumulative returns for the "
            f"selected mode (<b>{selected_mode}</b>). "
            f"Since-Inception metrics use each Wave's first available price date."
            f"</span>",
            unsafe_allow_html=True,
        )
        st.dataframe(display_df, use_container_width=True)

        bar_df = summary_df[["Wave", "SI Alpha"]].set_index("Wave")
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            "<span class='metric-label'>Since-Inception Alpha (All Waves)</span>",
            unsafe_allow_html=True,
        )
        st.bar_chart(bar_df)
    else:
        st.markdown("No data available yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------- Top 10 Holdings --------
with tab_top10:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Top 10 Holdings</div>', unsafe_allow_html=True)

    top10_df = load_top10_holdings(selected_wave)

    if top10_df.empty:
        st.markdown("No holdings data found for this Wave in wave_weights.csv.")
    else:
        st.markdown(
            "<span style='font-size:0.8rem;color:#9fa6b2;'>"
            "Source: wave_weights.csv (full basket). Tickers link directly to Google Finance."
            "</span>",
            unsafe_allow_html=True,
        )
        html_rows = [
            "<tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Weight</th></tr>"
        ]
        for _, row in top10_df.iterrows():
            html_rows.append(
                "<tr>"
                f"<td>{row['TickerLink']}</td>"
                f"<td>{row['Name']}</td>"
                f"<td>{row['Sector']}</td>"
                f"<td>{row['Weight%']}</td>"
                "</tr>"
            )
        html_table = (
            "<table class='top10-table'><thead>{head}</thead>"
            "<tbody>{body}</tbody></table>"
        ).format(head=html_rows[0], body="".join(html_rows[1:]))
        st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -------- Engine Logs (in-memory) --------
with tab_logs:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Engine Performance Log</div>', unsafe_allow_html=True)

    st.markdown(
        "This view shows the **computed daily performance** for the selected Wave "
        "based on full-basket prices from yfinance."
    )

    display_cols = [
        "date",
        "portfolio_return",
        "benchmark_return",
        "alpha_1d",
        "alpha_30d",
        "alpha_60d",
    ]
    display_cols_existing = [c for c in display_cols if c in perf_df.columns]
    st.dataframe(perf_df[display_cols_existing].tail(75), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)