# app.py — WAVES Institutional Console (MAX engine + MAX stability, auto-detect CSVs)

import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None  # Fallback if yfinance isn't available


# =========================
# General configuration
# =========================

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "WAVES Institutional Console"
APP_SUBTITLE = (
    "Adaptive Portfolio Waves · AIWs/APWs · SmartSafe™ · VIX-gated risk · "
    "Alpha-Minus-Beta & Private Logic™"
)

TARGET_BETA = 0.90
DEFAULT_LOOKBACK_DAYS = 365

# How many holdings to show in top table
TOP_N_HOLDINGS = 10

# -------------------------
# Simple debug collector
# -------------------------

if "debug_msgs" not in st.session_state:
    st.session_state["debug_msgs"] = []


def debug(msg: str) -> None:
    """Store debug messages for optional display."""
    ts = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state["debug_msgs"].append(f"[{ts}] {msg}")


# =========================
# File discovery helpers
# =========================

def find_first_existing(filenames):
    """Return Path to the first existing filename in current dir, or None."""
    for name in filenames:
        p = Path(name)
        if p.exists():
            debug(f"Found file: {p}")
            return p
    debug(f"No files found from candidates: {filenames}")
    return None


def auto_detect_weights_file():
    candidates = [
        "wave_weights.csv",
        "waves_weights.csv",
        "weights.csv",
        "WAVE_WEIGHTS.csv",
    ]
    return find_first_existing(candidates)


def auto_detect_universe_file():
    candidates = [
        "universe.csv",
        "Universe.csv",
        "master-stock-sheet.csv",
        "Master_Stock_Sheet.csv",
        "stocks_universe.csv",
        "List.csv",
    ]
    return find_first_existing(candidates)


def auto_detect_prices_file():
    candidates = [
        "prices.csv",
        "historical_prices.csv",
        "price_history.csv",
    ]
    return find_first_existing(candidates)


# =========================
# CSV loaders (robust)
# =========================

@st.cache_data(show_spinner=False)
def load_weights() -> pd.DataFrame:
    """
    Load wave weights from an auto-detected CSV.
    Expected logical columns: Wave, Ticker, Weight (case-insensitive).
    Returns clean DataFrame with columns: wave, ticker, weight.
    """
    weights_path = auto_detect_weights_file()
    if weights_path is None:
        debug("load_weights: no weights file found.")
        return pd.DataFrame(columns=["wave", "ticker", "weight"])

    try:
        df = pd.read_csv(weights_path)
    except Exception as e:
        debug(f"load_weights: error reading {weights_path}: {e}")
        return pd.DataFrame(columns=["wave", "ticker", "weight"])

    debug(f"load_weights: raw columns = {list(df.columns)}")

    # Normalize column names
    col_map = {c.lower().strip(): c for c in df.columns}
    # Find best matches
    wave_col = None
    ticker_col = None
    weight_col = None

    for key, orig in col_map.items():
        if "wave" in key:
            wave_col = orig
        if "ticker" in key or "symbol" in key:
            ticker_col = orig
        if "weight" in key or key in ("w", "wt"):
            weight_col = orig

    if wave_col is None or ticker_col is None or weight_col is None:
        debug(
            f"load_weights: missing required columns in {weights_path}. "
            f"wave_col={wave_col}, ticker_col={ticker_col}, weight_col={weight_col}"
        )
        return pd.DataFrame(columns=["wave", "ticker", "weight"])

    df_clean = pd.DataFrame()
    df_clean["wave"] = df[wave_col].astype(str).str.strip()
    df_clean["ticker"] = df[ticker_col].astype(str).str.strip().str.upper()
    df_clean["weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)

    # Drop obviously invalid rows
    df_clean = df_clean[df_clean["ticker"] != ""]
    df_clean = df_clean[df_clean["weight"] != 0.0]

    if df_clean.empty:
        debug("load_weights: cleaned DataFrame is empty after filtering.")
        return df_clean

    # Normalize weights within each wave
    def _normalize(group):
        total = group["weight"].sum()
        if total <= 0:
            return group
        group["weight"] = group["weight"] / total
        return group

    df_clean = df_clean.groupby("wave", group_keys=False).apply(_normalize)

    debug(
        f"load_weights: cleaned rows={len(df_clean)}, "
        f"waves={df_clean['wave'].nunique()}, tickers={df_clean['ticker'].nunique()}"
    )
    return df_clean


@st.cache_data(show_spinner=False)
def load_universe() -> pd.DataFrame:
    """
    Optional universe file with additional info.
    Looks for columns containing ticker/symbol and optionally name/sector.
    """
    universe_path = auto_detect_universe_file()
    if universe_path is None:
        debug("load_universe: no universe file found (optional).")
        return pd.DataFrame(columns=["ticker", "name", "sector"])

    try:
        df = pd.read_csv(universe_path)
    except Exception as e:
        debug(f"load_universe: error reading {universe_path}: {e}")
        return pd.DataFrame(columns=["ticker", "name", "sector"])

    debug(f"load_universe: raw columns = {list(df.columns)}")

    col_map = {c.lower().strip(): c for c in df.columns}
    ticker_col = None
    name_col = None
    sector_col = None

    for key, orig in col_map.items():
        if "ticker" in key or "symbol" in key:
            ticker_col = orig
        if key in ("name", "company", "security", "description"):
            name_col = orig
        if "sector" in key:
            sector_col = orig

    if ticker_col is None:
        debug("load_universe: no ticker/symbol column found.")
        return pd.DataFrame(columns=["ticker", "name", "sector"])

    df_clean = pd.DataFrame()
    df_clean["ticker"] = df[ticker_col].astype(str).str.strip().str.upper()
    if name_col is not None:
        df_clean["name"] = df[name_col].astype(str).str.strip()
    else:
        df_clean["name"] = ""
    if sector_col is not None:
        df_clean["sector"] = df[sector_col].astype(str).str.strip()
    else:
        df_clean["sector"] = ""

    df_clean = df_clean[df_clean["ticker"] != ""]
    debug(
        f"load_universe: cleaned rows={len(df_clean)}, "
        f"tickers={df_clean['ticker'].nunique()}"
    )
    return df_clean


# =========================
# Price fetching (robust)
# =========================

@st.cache_data(show_spinner=False)
def load_prices_csv() -> pd.DataFrame:
    """
    Optional local prices CSV fallback.
    Expects columns: date, ticker, close OR wide format with date index.
    """
    prices_path = auto_detect_prices_file()
    if prices_path is None:
        debug("load_prices_csv: no prices CSV fallback found.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(prices_path)
    except Exception as e:
        debug(f"load_prices_csv: error reading {prices_path}: {e}")
        return pd.DataFrame()

    debug(f"load_prices_csv: raw columns = {list(df.columns)}")

    cols_lower = [c.lower().strip() for c in df.columns]
    if "date" in cols_lower and "ticker" in cols_lower:
        # long format: date, ticker, close
        date_col = df.columns[cols_lower.index("date")]
        ticker_col = df.columns[cols_lower.index("ticker")]
        # choose a price column: close, adj close, price
        price_col = None
        for candidate in ["close", "adj close", "adj_close", "price"]:
            if candidate in cols_lower:
                price_col = df.columns[cols_lower.index(candidate)]
                break
        if price_col is None:
            debug("load_prices_csv: couldn't find price column.")
            return pd.DataFrame()

        df[date_col] = pd.to_datetime(df[date_col])
        pivot = df.pivot(index=date_col, columns=ticker_col, values=price_col)
        pivot.sort_index(inplace=True)
        debug(
            f"load_prices_csv: long->wide rows={pivot.shape[0]}, cols={pivot.shape[1]}"
        )
        return pivot

    # If already wide with date column
    if "date" in cols_lower:
        date_col = df.columns[cols_lower.index("date")]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        debug(f"load_prices_csv: wide format rows={df.shape[0]}, cols={df.shape[1]}")
        return df

    debug("load_prices_csv: unrecognized format.")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_prices_yfinance(tickers, days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch historical prices from yfinance, wide DataFrame indexed by date."""
    if yf is None or not tickers:
        debug("fetch_prices_yfinance: yfinance not available or tickers empty.")
        return pd.DataFrame()

    end = datetime.utcnow()
    start = end - timedelta(days=days + 5)  # small buffer

    try:
        data = yf.download(
            tickers=list(set(tickers)),
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        debug(f"fetch_prices_yfinance: error calling yfinance: {e}")
        return pd.DataFrame()

    if data.empty:
        debug("fetch_prices_yfinance: yfinance returned empty DataFrame.")
        return pd.DataFrame()

    # Normalize to wide: date index, columns = tickers
    if isinstance(data.columns, pd.MultiIndex):
        wide = pd.DataFrame(index=data.index)
        for ticker in tickers:
            if (ticker, "Adj Close") in data.columns:
                col = data[(ticker, "Adj Close")]
            elif (ticker, "Close") in data.columns:
                col = data[(ticker, "Close")]
            else:
                continue
            wide[ticker] = col
    else:
        # Single ticker case
        price_col = None
        for c in ["Adj Close", "Close"]:
            if c in data.columns:
                price_col = c
                break
        if price_col is None:
            debug("fetch_prices_yfinance: couldn't find price column for single ticker.")
            return pd.DataFrame()
        wide = pd.DataFrame({tickers[0]: data[price_col]})

    wide = wide.dropna(how="all")
    debug(f"fetch_prices_yfinance: rows={wide.shape[0]}, cols={wide.shape[1]}")
    return wide


@st.cache_data(show_spinner=False)
def get_price_history(tickers, days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Main price loader:
    1) Try yfinance
    2) Fallback to local prices CSV
    3) Fallback to synthetic random-walk prices
    """
    tickers = [t for t in tickers if isinstance(t, str) and t.strip() != ""]
    tickers = sorted(list(set(tickers)))
    if not tickers:
        debug("get_price_history: no tickers provided.")
        return pd.DataFrame()

    # 1) yfinance
    prices = fetch_prices_yfinance(tickers, days=days)
    if not prices.empty:
        return prices

    # 2) prices CSV fallback
    csv_prices = load_prices_csv()
    if not csv_prices.empty:
        missing = [t for t in tickers if t not in csv_prices.columns]
        available = [t for t in tickers if t in csv_prices.columns]
        debug(
            f"get_price_history: using prices CSV fallback. "
            f"available={len(available)}, missing={len(missing)}"
        )
        if available:
            subset = csv_prices[available].copy()
            # clip lookback
            if subset.shape[0] > days:
                subset = subset.iloc[-days:]
            return subset

    # 3) synthetic
    debug("get_price_history: generating synthetic random-walk prices.")
    dates = pd.date_range(end=datetime.utcnow().date(), periods=days, freq="B")
    prices = pd.DataFrame(index=dates)
    for t in tickers:
        steps = np.random.normal(loc=0.0005, scale=0.02, size=len(dates))
        series = 100 * (1 + pd.Series(steps, index=dates)).cumprod()
        prices[t] = series
    return prices


@st.cache_data(show_spinner=False)
def get_benchmark_and_vix(days: int = DEFAULT_LOOKBACK_DAYS):
    """Fetch SPY (benchmark) and VIX history."""
    spy = get_price_history(["SPY"], days=days)
    vix = get_price_history(["^VIX"], days=days)
    spy_series = spy["SPY"] if "SPY" in spy.columns else pd.Series(dtype=float)
    vix_series = vix["^VIX"] if "^VIX" in vix.columns else pd.Series(dtype=float)
    return spy_series, vix_series


# =========================
# Portfolio engine
# =========================

def compute_portfolio_series(prices: pd.DataFrame, weights: pd.Series):
    """
    Compute portfolio price series given wide prices and weights (indexed by ticker).
    """
    cols = [t for t in weights.index if t in prices.columns]
    if not cols:
        debug("compute_portfolio_series: no overlapping tickers between prices and weights.")
        return pd.Series(dtype=float)
    aligned_prices = prices[cols].copy()
    w = weights[cols]
    w = w / w.sum()
    portfolio = (aligned_prices.pct_change().fillna(0) * w).sum(axis=1)
    portfolio_index = (1 + portfolio).cumprod()
    return portfolio_index


def estimate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series):
    """Simple beta estimate using covariance / variance."""
    joined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if joined.shape[0] < 30:
        debug("estimate_beta: insufficient overlap to estimate beta; using TARGET_BETA.")
        return TARGET_BETA
    rp = joined.iloc[:, 0]
    rb = joined.iloc[:, 1]
    var_b = rb.var()
    if var_b == 0 or np.isnan(var_b):
        debug("estimate_beta: benchmark variance zero/NaN; using TARGET_BETA.")
        return TARGET_BETA
    beta = rp.cov(rb) / var_b
    debug(f"estimate_beta: beta={beta:.3f}")
    return float(beta)


def vix_exposure_scale(vix_latest: float) -> float:
    """Simple VIX ladder for scaling exposure."""
    if np.isnan(vix_latest):
        return 1.0
    if vix_latest >= 35:
        return 0.4
    if vix_latest >= 25:
        return 0.7
    if vix_latest >= 18:
        return 1.0
    if vix_latest >= 12:
        return 1.05
    return 1.10


def apply_mode_exposure(base_exposure: float, beta_est: float, mode: str) -> float:
    """Adjust exposure based on selected mode."""
    exposure = base_exposure
    beta_safe = max(beta_est, 0.01)
    if mode == "Standard":
        pass
    elif mode == "Alpha-Minus-Beta":
        # Dial exposure closer to 0.85 while controlling beta
        exposure *= min(1.0, 0.85 / beta_safe)
    elif mode == "Private Logic™":
        # Allow slightly higher exposure within strict cap
        exposure *= 1.10
    exposure = max(0.0, min(1.20, exposure))
    return exposure


def compute_wave_engine(
    wave_name: str,
    mode: str,
    weights_df: pd.DataFrame,
    spy_series: pd.Series,
    vix_series: pd.Series,
    days: int = DEFAULT_LOOKBACK_DAYS,
):
    """
    Core engine: compute portfolio & benchmark curves, alpha, exposure, SmartSafe, etc.
    Returns dict with all series & metrics.
    """
    if weights_df.empty:
        return {"error": "No weights data loaded."}

    wdf = weights_df[weights_df["wave"] == wave_name].copy()
    if wdf.empty:
        return {"error": f"No weights found for wave '{wave_name}'."}

    tickers = list(wdf["ticker"].unique())
    prices = get_price_history(tickers, days=days)
    if prices.empty:
        return {"error": "No price data available for selected wave tickers."}

    # Align benchmark and VIX with prices index
    benchmark = spy_series.reindex(prices.index).dropna()
    if benchmark.empty:
        # try to align using common index
        common_index = prices.index.intersection(spy_series.index)
        benchmark = spy_series.reindex(common_index)

    if benchmark.empty:
        debug("compute_wave_engine: benchmark series empty.")
        return {"error": "Benchmark (SPY) data unavailable."}

    prices = prices.reindex(benchmark.index).dropna(how="all")
    if prices.empty:
        return {"error": "No overlapping price history between wave and benchmark."}

    # Re-normalize weights just in case
    weights = wdf.set_index("ticker")["weight"]
    weights = weights / weights.sum()

    portfolio_index = compute_portfolio_series(prices, weights)
    # Align again
    common = portfolio_index.index.intersection(benchmark.index)
    portfolio_index = portfolio_index.reindex(common).dropna()
    benchmark = benchmark.reindex(common).dropna()

    if portfolio_index.empty or benchmark.empty:
        return {"error": "Insufficient overlapping history for engine computation."}

    # Convert benchmark prices to index
    bench_returns = benchmark.pct_change().fillna(0)
    bench_index = (1 + bench_returns).cumprod()

    port_returns = portfolio_index.pct_change().fillna(0)

    beta_est = estimate_beta(port_returns, bench_returns)

    # Latest VIX
    vix_latest = np.nan
    if not vix_series.empty:
        vix_latest = float(vix_series.iloc[-1])

    vix_scale = vix_exposure_scale(vix_latest)
    base_exposure = min(1.0, TARGET_BETA / max(beta_est, 0.01)) * vix_scale
    exposure = apply_mode_exposure(base_exposure, beta_est, mode)
    exposure = max(0.0, min(1.20, exposure))
    smartsafe = max(0.0, 1.0 - exposure)

    # Alpha series: portfolio vs scaled benchmark
    scaled_bench_returns = bench_returns * exposure
    alpha_series = port_returns - scaled_bench_returns
    cum_alpha = alpha_series.cumsum()

    # Today numbers
    today_port = float(port_returns.iloc[-1])
    today_bench = float(bench_returns.iloc[-1])
    today_alpha = float(alpha_series.iloc[-1])

    # Top holdings (today % change)
    today_returns = prices.pct_change().iloc[-1].fillna(0)
    top_weights = weights.sort_values(ascending=False).head(TOP_N_HOLDINGS)
    top_rows = []
    for ticker, w in top_weights.items():
        ret_today = float(today_returns.get(ticker, np.nan))
        top_rows.append(
            {
                "ticker": ticker,
                "weight": w,
                "today_ret": ret_today,
            }
        )

    result = {
        "portfolio_index": portfolio_index,
        "benchmark_index": bench_index.reindex(portfolio_index.index),
        "alpha_series": cum_alpha.reindex(portfolio_index.index),
        "port_returns": port_returns,
        "bench_returns": bench_returns,
        "today_port": today_port,
        "today_bench": today_bench,
        "today_alpha": today_alpha,
        "exposure": exposure,
        "smartsafe": smartsafe,
        "beta_est": beta_est,
        "vix_latest": vix_latest,
        "top_holdings": top_rows,
    }
    return result


# =========================
# UI helpers
# =========================

def format_pct(x):
    if x is None or np.isnan(x):
        return "—"
    return f"{x*100:.2f}%"


def build_top_holdings_html(top_rows):
    if not top_rows:
        return "<p>No holdings available.</p>"

    html = [
        """
        <style>
        table.waves-top-holdings {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        table.waves-top-holdings th, table.waves-top-holdings td {
            border-bottom: 1px solid #333333;
            padding: 4px 6px;
            text-align: left;
        }
        table.waves-top-holdings th {
            font-weight: 600;
        }
        </style>
        """,
        "<table class='waves-top-holdings'>",
        "<thead><tr><th>Ticker</th><th>Weight</th><th>Today %</th></tr></thead>",
        "<tbody>",
    ]
    for row in top_rows:
        ticker = row["ticker"]
        weight = row["weight"]
        today_ret = row["today_ret"]
        if np.isnan(today_ret):
            color = "#CCCCCC"
            today_str = "—"
        else:
            color = "#00FF7F" if today_ret > 0 else "#FF4B4B" if today_ret < 0 else "#CCCCCC"
            today_str = f"{today_ret*100:.2f}%"
        url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
        html.append(
            "<tr>"
            f"<td><a href='{url}' target='_blank'>{ticker}</a></td>"
            f"<td>{weight*100:.2f}%</td>"
            f"<td style='color:{color}; font-weight:600;'>{today_str}</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    return "\n".join(html)


# =========================
# Main Streamlit layout
# =========================

def main():
    # Header
    col_left, col_right = st.columns([4, 1])

    with col_left:
        st.markdown(
            f"<h1 style='margin-bottom:0.1rem;'>{APP_TITLE}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='opacity:0.8; margin-bottom:0.5rem;'>{APP_SUBTITLE}</p>",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            "<div style='text-align:right; font-size:0.85rem;'>"
            "<div><span style='color:#00FF7F; font-weight:600;'>Engine Status:</span> "
            "<span style='color:#00FF7F;'>LIVE</span> / "
            "<span style='color:#FFD700;'>SANDBOX</span></div>"
            f"<div>Last refresh: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Sidebar controls
    st.sidebar.title("Wave & Mode")

    weights_df = load_weights()
    universe_df = load_universe()

    if weights_df.empty:
        st.error(
            "No wave weights loaded. Please ensure a CSV like "
            "`wave_weights.csv`, `weights.csv`, or `waves_weights.csv` "
            "is present in the app directory with columns Wave/Ticker/Weight."
        )
        show_debug = st.sidebar.checkbox("Show debug info")
        if show_debug:
            st.sidebar.write("\n".join(st.session_state["debug_msgs"]))
        return

    waves = sorted(weights_df["wave"].unique())
    default_wave = waves[0] if waves else ""
    selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)

    mode = st.sidebar.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    days = st.sidebar.slider(
        "Lookback (trading days)",
        min_value=60,
        max_value=730,
        value=DEFAULT_LOOKBACK_DAYS,
        step=10,
    )

    show_alpha_chart = st.sidebar.checkbox("Show alpha curve", value=True)
    show_drawdown = st.sidebar.checkbox("Show drawdown", value=False)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # Data / engine
    spy_series, vix_series = get_benchmark_and_vix(days=days)

    result = compute_wave_engine(
        wave_name=selected_wave,
        mode=mode,
        weights_df=weights_df,
        spy_series=spy_series,
        vix_series=vix_series,
        days=days,
    )

    if "error" in result:
        st.error(result["error"])
        if show_debug:
            with st.expander("Debug log", expanded=True):
                st.write("\n".join(st.session_state["debug_msgs"]))
        return

    # KPIs row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric(
            label="Wave Today",
            value=format_pct(result["today_port"]),
            delta=None,
        )
    with kpi2:
        st.metric(
            label="Benchmark Today (SPY)",
            value=format_pct(result["today_bench"]),
            delta=None,
        )
    with kpi3:
        st.metric(
            label="Today Alpha",
            value=format_pct(result["today_alpha"]),
            delta=None,
        )
    with kpi4:
        st.metric(
            label="Exposure",
            value=f"{result['exposure']*100:.1f}%",
            delta=None,
        )
    with kpi5:
        st.metric(
            label="SmartSafe™ Allocation",
            value=f"{result['smartsafe']*100:.1f}%",
            delta=None,
        )

    # Second row of KPIs: Beta & VIX
    kpi6, kpi7 = st.columns(2)
    with kpi6:
        st.metric(
            label="Estimated Beta vs SPY",
            value=f"{result['beta_est']:.2f}",
            delta=f"Target {TARGET_BETA:.2f}",
        )
    with kpi7:
        vix_val = result["vix_latest"]
        vix_txt = "—" if np.isnan(vix_val) else f"{vix_val:.1f}"
        st.metric(label="VIX (latest)", value=vix_txt)

    st.markdown("---")

    # Charts: portfolio vs benchmark + alpha/drawdown
    col_chart_main, col_chart_side = st.columns([2.2, 1.8])

    port_index = result["portfolio_index"]
    bench_index = result["benchmark_index"].reindex(port_index.index)
    alpha_series = result["alpha_series"]

    df_curve = pd.DataFrame(
        {
            "Wave": port_index / port_index.iloc[0] - 1,
            "Benchmark": bench_index / bench_index.iloc[0] - 1,
        }
    )

    with col_chart_main:
        st.subheader(f"{selected_wave} vs Benchmark")
        st.line_chart(df_curve)

        if show_alpha_chart:
            st.subheader("Cumulative Alpha (Wave – Scaled Benchmark)")
            st.line_chart(alpha_series)

        if show_drawdown:
            # simple drawdown
            peak = port_index.cummax()
            dd = (port_index / peak) - 1
            st.subheader("Drawdown (Wave)")
            st.line_chart(dd)

    with col_chart_side:
        st.subheader("Top Holdings (Live)")

        top_html = build_top_holdings_html(result["top_holdings"])
        st.markdown(top_html, unsafe_allow_html=True)

        # Show some extra info from universe if available
        if not universe_df.empty:
            tickers = [row["ticker"] for row in result["top_holdings"]]
            extra = universe_df[universe_df["ticker"].isin(tickers)].set_index("ticker")
            if not extra.empty:
                st.caption("Names / sectors (from universe file):")
                st.table(extra[["name", "sector"]])

    # SPY + VIX mini-charts
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("SPY (Benchmark) – Price")
        if not spy_series.empty:
            spy_norm = spy_series / spy_series.iloc[0] - 1
            st.line_chart(spy_norm)
        else:
            st.info("SPY data not available.")

    with c2:
        st.subheader("VIX – Level")
        if not vix_series.empty:
            st.line_chart(vix_series)
        else:
            st.info("VIX data not available.")

    # Debug info
    if show_debug:
        st.markdown("---")
        st.subheader("Debug log")
        if st.session_state["debug_msgs"]:
            st.text("\n".join(st.session_state["debug_msgs"]))
        else:
            st.text("No debug messages yet.")


if __name__ == "__main__":
    main()
