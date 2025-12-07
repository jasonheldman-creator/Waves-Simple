# app.py  — WAVES Institutional Console (Single Script)

import os
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st

# Try to import yfinance, but don't die if it's missing
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
)

WAVE_WEIGHTS_PATH = os.environ.get("WAVES_WEIGHTS_CSV", "wave_weights.csv")

# Benchmarks for each Wave (you can edit these tickers to match your lineup)
WAVE_BENCHMARKS: Dict[str, str] = {
    "S&P Wave": "SPY",
    "Growth Wave": "QQQ",
    "Income Wave": "VYM",
    "Small Cap Growth Wave": "IWM",
    "Small to Mid Cap Growth Wave": "IJH",
    "Future Power & Energy Wave": "XLE",
    "Crypto Income Wave": "BTC-USD",
    "Quantum Computing Wave": "BOTZ",
    "Clean Transit-Infrastructure Wave": "ICLN",
    # Add any missing waves here...
}

TARGET_BETA = 0.90

SP_MARKET_TICKER = "SPY"   # For S&P mini-chart
VIX_TICKER = "^VIX"        # For VIX mini-chart

HISTORY_LOOKBACK_DAYS = 365        # approx 1 year
BETA_LOOKBACK_DAYS = 60
MOM_LONG_DAYS = 63
MOM_SHORT_DAYS = 21


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WaveEngineResult:
    wave_name: str
    mode: str
    portfolio_series: pd.Series
    benchmark_series: pd.Series
    alpha_series: pd.Series
    last_portfolio_return: float
    last_benchmark_return: float
    last_alpha: float
    exposure: float
    smartsafe_alloc: float
    beta_estimate: Optional[float]
    regime_label: str
    holdings_today: pd.DataFrame  # includes today % change, weights, etc.


# =============================================================================
# UTILS: SAFE HELPERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_weights(csv_path: str) -> pd.DataFrame:
    """
    Load wave weights from CSV.
    Expected flexible columns: one for wave, one for ticker/symbol, one for weight.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not read weights file at `{csv_path}`: {e}")
        return pd.DataFrame()

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Try to identify required columns
    wave_col = None
    ticker_col = None
    weight_col = None

    for cand in ["wave", "portfolio", "wave_name"]:
        if cand in df.columns:
            wave_col = cand
            break

    for cand in ["ticker", "symbol", "asset", "security"]:
        if cand in df.columns:
            ticker_col = cand
            break

    for cand in ["weight", "target_weight", "w"]:
        if cand in df.columns:
            weight_col = cand
            break

    if wave_col is None or ticker_col is None or weight_col is None:
        st.error(
            "Weights CSV is missing required columns. "
            "Please include columns for wave, ticker, and weight "
            "(e.g., 'Wave', 'Ticker', 'Weight')."
        )
        return pd.DataFrame()

    df = df[[wave_col, ticker_col, weight_col]].copy()
    df.columns = ["wave", "ticker", "weight"]

    # Drop rows with missing tickers or weights
    df = df.dropna(subset=["ticker", "weight"])
    # Normalize wave names and tickers
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Normalize weights per wave
    df["weight"] = df["weight"].astype(float)
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )

    return df


@st.cache_data(show_spinner=False)
def fetch_price_history(
    tickers: List[str], days: int = HISTORY_LOOKBACK_DAYS
) -> pd.DataFrame:
    """
    Fetch adjusted close history for given tickers.
    Returns DataFrame indexed by date with tickers as columns.
    If yfinance is unavailable or fetch fails, returns empty DataFrame.
    """
    if not YF_AVAILABLE:
        st.warning("yfinance is not installed; running in offline mode.")
        return pd.DataFrame()

    if len(tickers) == 0:
        return pd.DataFrame()

    end = dt.date.today()
    start = end - dt.timedelta(days=days + 10)

    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            data = data["Adj Close"]
        # Ensure a 2D DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data = data.dropna(how="all")
        return data
    except Exception as e:
        st.error(f"Error fetching price history from yfinance: {e}")
        return pd.DataFrame()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns from price history."""
    if prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna(how="all")
    return returns


def estimate_beta(
    port_rets: pd.Series, bench_rets: pd.Series, lookback_days: int = BETA_LOOKBACK_DAYS
) -> Optional[float]:
    """Estimate beta over a trailing window."""
    if port_rets.empty or bench_rets.empty:
        return None

    aligned = pd.concat([port_rets, bench_rets], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return None

    aligned = aligned.tail(lookback_days)
    x = aligned.iloc[:, 1]  # benchmark
    y = aligned.iloc[:, 0]  # portfolio

    if x.var() == 0:
        return None

    cov = np.cov(x, y)[0, 1]
    beta = cov / x.var()
    return beta


def vix_risk_scaler(vix_level: float, mode: str) -> float:
    """
    Map VIX level + mode to an exposure scaler between 0 and 1.
    You can tweak these bands to match your live spec.
    """
    if np.isnan(vix_level):
        return 1.0

    # Base bands
    if vix_level < 15:
        base = 1.0
    elif vix_level < 25:
        base = 0.8
    elif vix_level < 35:
        base = 0.6
    else:
        base = 0.4

    # Mode tweaks
    if mode == "Alpha-Minus-Beta":
        base *= 0.9
    elif mode == "Private Logic™":
        base *= 1.1

    return float(max(0.0, min(1.2, base)))


def detect_regime(spy_price: pd.Series, vix_price: pd.Series) -> str:
    """
    Very simple regime label using S&P trend & VIX level.
    """
    if spy_price.empty or vix_price.empty:
        return "Unknown"

    vix_last = vix_price.iloc[-1]
    spy = spy_price.dropna()
    if len(spy) < 40:
        return "Unknown"

    short = spy.tail(20).mean()
    long = spy.tail(40).mean()

    if short > long and vix_last < 18:
        return "Risk-On"
    elif short < long and vix_last > 25:
        return "Risk-Off"
    else:
        return "Choppy"


def compute_momentum_scores(
    prices: pd.DataFrame,
    long_days: int = MOM_LONG_DAYS,
    short_days: int = MOM_SHORT_DAYS,
) -> pd.Series:
    """
    Momentum score for each ticker: blend of long + short lookback returns.
    """
    if prices.empty:
        return pd.Series(dtype=float)

    if len(prices) < long_days + 2:
        return pd.Series(dtype=float)

    long_ret = prices.pct_change(long_days).iloc[-1]
    short_ret = prices.pct_change(short_days).iloc[-1]

    score = 0.6 * long_ret + 0.4 * short_ret
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def tilt_weights_with_momentum(
    base_weights: pd.Series,
    momentum_scores: pd.Series,
    intensity: float = 0.3,
) -> pd.Series:
    """
    Tilt base weights toward higher-momentum names.
    intensity: 0 = no tilt, 0.3 = modest tilt, 0.5+ = stronger.
    """
    if base_weights.empty:
        return base_weights

    scores = momentum_scores.reindex(base_weights.index).fillna(0.0)
    if scores.std() == 0:
        return base_weights

    z = (scores - scores.mean()) / scores.std()
    tilt = 1.0 + intensity * z
    tilted = base_weights * tilt.clip(lower=0.1)
    if tilted.sum() > 0:
        tilted /= tilted.sum()
    return tilted


def style_top_holdings(df: pd.DataFrame):
    """
    Return a Styler for the top holdings table with red/green today % change.
    No type hints referencing pd.io to avoid AttributeError.
    """
    if df.empty or "today_pct" not in df.columns:
        return df

    def color_pct(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 0:
            return "color: rgb(0, 200, 0); font-weight: 600;"
        elif v < 0:
            return "color: rgb(220, 0, 0); font-weight: 600;"
        else:
            return ""

    styler = (
        df.style
        .format(
            {
                "weight": "{:.2%}",
                "today_pct": "{:+.2%}",
            }
        )
        .applymap(color_pct, subset=["today_pct"])
    )
    return styler


def google_finance_link(ticker: str) -> str:
    """
    Return a Google Finance URL for the ticker.
    (We keep it generic – Google will route to the right page.)
    """
    t = ticker.upper()
    return f"https://www.google.com/finance/quote/{t}"


# =============================================================================
# CORE ENGINE
# =============================================================================

def run_wave_engine(
    wave_name: str,
    mode: str,
    weights_df: pd.DataFrame,
    prices_all: pd.DataFrame,
    bench_prices: pd.Series,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
) -> Optional[WaveEngineResult]:
    """
    Build full engine result for one Wave & Mode.
    """
    wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
    if wave_weights.empty:
        st.warning(f"No holdings found in weights file for wave: {wave_name}")
        return None

    tickers = wave_weights["ticker"].unique().tolist()

    # Slice price matrix for this wave
    port_prices = prices_all[tickers].dropna(how="all", axis=0)
    if port_prices.empty or bench_prices.empty:
        st.warning("Price history is missing for this wave or benchmark.")
        return None

    # Align benchmark with portfolio dates
    bench_prices = bench_prices.reindex(port_prices.index).ffill().dropna()
    if bench_prices.empty:
        st.warning("Benchmark price series is empty after alignment.")
        return None

    # Compute returns
    port_rets_all = compute_returns(port_prices)
    bench_rets_all = compute_returns(bench_prices.to_frame()).iloc[:, 0]

    if port_rets_all.empty or bench_rets_all.empty:
        st.warning("Unable to compute returns for wave or benchmark.")
        return None

    # Base weights as a Series indexed by ticker
    base_weights = (
        wave_weights.set_index("ticker")["weight"]
        .reindex(port_prices.columns)
        .fillna(0.0)
    )

    # Momentum tilt
    momentum_scores = compute_momentum_scores(port_prices)
    tilted_weights = tilt_weights_with_momentum(base_weights, momentum_scores, intensity=0.3)

    # Compute raw portfolio returns (before risk scaling)
    port_rets_raw = (port_rets_all * tilted_weights).sum(axis=1)

    # Estimate beta and apply beta discipline scaling
    beta_est = estimate_beta(port_rets_raw, bench_rets_all, BETA_LOOKBACK_DAYS)
    beta_scale = 1.0
    if beta_est is not None and beta_est != 0:
        beta_scale = TARGET_BETA / beta_est
        # Clamp to keep from going crazy
        beta_scale = float(max(0.5, min(1.5, beta_scale)))

    # VIX risk scaler
    vix_last = vix_prices.iloc[-1] if not vix_prices.empty else np.nan
    risk_scale = vix_risk_scaler(float(vix_last) if not np.isnan(vix_last) else np.nan, mode)

    total_scale = beta_scale * risk_scale
    total_scale = float(max(0.0, min(1.5, total_scale)))

    exposure = min(1.0, total_scale)
    smartsafe_alloc = 1.0 - exposure

    # Final portfolio returns after scaling (exposure applied)
    port_rets_scaled = exposure * port_rets_raw

    # Build cumulative curves
    port_curve = (1.0 + port_rets_scaled).cumprod()
    bench_curve = (1.0 + bench_rets_all).cumprod()
    alpha_series = port_curve - bench_curve

    # Align curves
    common_index = port_curve.index.intersection(bench_curve.index)
    port_curve = port_curve.reindex(common_index)
    bench_curve = bench_curve.reindex(common_index)
    alpha_series = alpha_series.reindex(common_index)

    last_port_ret = port_rets_scaled.iloc[-1]
    last_bench_ret = bench_rets_all.reindex(port_rets_scaled.index).iloc[-1]
    last_alpha = last_port_ret - last_bench_ret

    # Regime detection
    regime = detect_regime(spy_prices, vix_prices)

    # Top holdings with today's change and links
    today_idx = port_prices.index[-1]
    if len(port_prices) >= 2:
        yesterday_idx = port_prices.index[-2]
        today_prices = port_prices.loc[today_idx]
        yday_prices = port_prices.loc[yesterday_idx]
        today_pct = (today_prices / yday_prices - 1.0).replace([np.inf, -np.inf], np.nan)
    else:
        today_prices = port_prices.loc[today_idx]
        today_pct = pd.Series(0.0, index=today_prices.index)

    holdings_today = pd.DataFrame(
        {
            "ticker": port_prices.columns,
            "weight": tilted_weights.reindex(port_prices.columns).fillna(0.0),
            "today_pct": today_pct.reindex(port_prices.columns).fillna(0.0),
        }
    )

    holdings_today["google_link"] = holdings_today["ticker"].apply(google_finance_link)
    holdings_today = holdings_today.sort_values("weight", ascending=False).head(10).reset_index(drop=True)

    return WaveEngineResult(
        wave_name=wave_name,
        mode=mode,
        portfolio_series=port_curve,
        benchmark_series=bench_curve,
        alpha_series=alpha_series,
        last_portfolio_return=float(last_port_ret),
        last_benchmark_return=float(last_bench_ret),
        last_alpha=float(last_alpha),
        exposure=exposure,
        smartsafe_alloc=smartsafe_alloc,
        beta_estimate=beta_est,
        regime_label=regime,
        holdings_today=holdings_today,
    )


# =============================================================================
# UI LAYOUT
# =============================================================================

def render_header():
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; padding:0.5rem 0;">
          <div>
            <h1 style="margin-bottom:0.2rem;">WAVES Institutional Console</h1>
            <p style="margin-top:0; color:#888;">
              Adaptive Portfolio Waves • AIWs/APWs • SmartSafe™ • VIX-gated risk • Alpha-Minus-Beta & Private Logic™
            </p>
          </div>
          <div style="text-align:right; font-size:0.9rem; color:#aaa;">
            <div>Engine Status: <span style="color:#0f0; font-weight:600;">LIVE / SANDBOX</span></div>
            <div>Last refresh: {now}</div>
          </div>
        </div>
        """.format(now=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True,
    )
    st.markdown("---")


def sidebar_controls(weights_df: pd.DataFrame) -> Tuple[str, str]:
    st.sidebar.title("Wave & Mode")

    waves = sorted(weights_df["wave"].unique()) if not weights_df.empty else []
    if not waves:
        waves = ["S&P Wave"]

    wave_name = st.sidebar.selectbox("Select Wave", options=waves, index=0)

    mode = st.sidebar.radio(
        "Mode",
        options=["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.write("**Display Options**")
    show_alpha = st.sidebar.checkbox("Show Alpha curve", value=True)
    show_drawdown = st.sidebar.checkbox("Show drawdown", value=False)

    st.session_state.setdefault("ui_show_alpha", show_alpha)
    st.session_state.setdefault("ui_show_drawdown", show_drawdown)

    st.sidebar.markdown("---")
    st.sidebar.caption("WAVES Intelligence™ • For demo/illustration only • Not investment advice")

    return wave_name, mode


def render_kpi_row(result: WaveEngineResult):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label=f"{result.wave_name} — Today",
        value=f"{result.last_portfolio_return:+.2%}",
        delta=f"Alpha {result.last_alpha:+.2%}",
    )

    c2.metric(
        label="Benchmark — Today",
        value=f"{result.last_benchmark_return:+.2%}",
    )

    c3.metric(
        label="Exposure (Equity)",
        value=f"{result.exposure:.0%}",
        delta=f"SmartSafe™ {result.smartsafe_alloc:.0%}",
    )

    beta_label = "β estimate"
    beta_value = f"{result.beta_estimate:.2f}" if result.beta_estimate is not None else "N/A"

    c4.metric(
        label=beta_label,
        value=beta_value,
        delta=f"Target β {TARGET_BETA:.2f}",
    )

    st.caption(f"Regime: **{result.regime_label}**  •  Mode: **{result.mode}**")


def render_performance_charts(result: WaveEngineResult):
    st.subheader("Wave vs Benchmark")

    df_perf = pd.DataFrame(
        {
            f"{result.wave_name}": result.portfolio_series,
            "Benchmark": result.benchmark_series,
        }
    )

    st.line_chart(df_perf)

    if st.session_state.get("ui_show_alpha", True):
        st.subheader("Alpha (Cumulative)")
        st.line_chart(result.alpha_series)

    if st.session_state.get("ui_show_drawdown", False):
        st.subheader("Drawdown (from peak)")
        curve = result.portfolio_series
        running_max = curve.cummax()
        drawdown = (curve / running_max - 1.0)
        st.line_chart(drawdown)


def render_top_holdings_table(result: WaveEngineResult):
    st.subheader("Top Holdings (Live)")

    df = result.holdings_today.copy()

    # Show link as text (user can click/open)
    df_display = df[["ticker", "weight", "today_pct", "google_link"]].rename(
        columns={
            "ticker": "Ticker",
            "weight": "Weight",
            "today_pct": "Today % Change",
            "google_link": "Google Finance",
        }
    )

    styler = style_top_holdings(df_display)

    st.dataframe(
        styler,
        use_container_width=True,
    )


def render_market_mini_charts(spy_prices: pd.Series, vix_prices: pd.Series):
    st.subheader("Market Context — S&P & VIX")

    c1, c2 = st.columns(2)

    if not spy_prices.empty:
        c1.markdown("**S&P (SPY) — Price**")
        c1.line_chart(spy_prices)
    else:
        c1.info("S&P data unavailable.")

    if not vix_prices.empty:
        c2.markdown("**VIX Index**")
        c2.line_chart(vix_prices)
    else:
        c2.info("VIX data unavailable.")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    render_header()

    # Load weights CSV
    weights_df = load_weights(WAVE_WEIGHTS_PATH)
    if weights_df.empty:
        st.stop()

    wave_name, mode = sidebar_controls(weights_df)

    # Build master ticker list (all waves + all benchmarks + SPY + VIX)
    all_wave_tickers = weights_df["ticker"].unique().tolist()

    benchmark_tickers = list(set(WAVE_BENCHMARKS.values()))
    extra_tickers = [SP_MARKET_TICKER]
    price_tickers = sorted(list(set(all_wave_tickers + benchmark_tickers + extra_tickers)))

    # Fetch all prices in one call
    prices_all = fetch_price_history(price_tickers, HISTORY_LOOKBACK_DAYS)

    if prices_all.empty:
        st.error("No price data available. Check network / yfinance / ticker symbols.")
        st.stop()

    # Extract SPY & VIX
    spy_prices = fetch_price_history([SP_MARKET_TICKER], HISTORY_LOOKBACK_DAYS)
    spy_series = spy_prices.iloc[:, 0] if not spy_prices.empty else pd.Series(dtype=float)

    vix_prices_df = fetch_price_history([VIX_TICKER], HISTORY_LOOKBACK_DAYS)
    vix_series = vix_prices_df.iloc[:, 0] if not vix_prices_df.empty else pd.Series(dtype=float)

    # Benchmark series for selected wave
    bench_ticker = WAVE_BENCHMARKS.get(wave_name, SP_MARKET_TICKER)
    if bench_ticker in prices_all.columns:
        bench_series = prices_all[bench_ticker]
    else:
        st.warning(f"Benchmark ticker {bench_ticker} not found in price data; using SPY instead.")
        bench_series = prices_all[SP_MARKET_TICKER]

    # Run engine
    result = run_wave_engine(
        wave_name=wave_name,
        mode=mode,
        weights_df=weights_df,
        prices_all=prices_all,
        bench_prices=bench_series,
        spy_prices=spy_series,
        vix_prices=vix_series,
    )

    if result is None:
        st.stop()

    # Layout sections
    render_kpi_row(result)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_performance_charts(result)

    with col_right:
        render_top_holdings_table(result)
        render_market_mini_charts(spy_series, vix_series)

    st.markdown("---")
    st.caption(
        "WAVES Intelligence™ — Mini Bloomberg-style console for demonstration. "
        "All outputs are simulated and for illustrative purposes only; not investment advice."
    )


if __name__ == "__main__":
    main()
