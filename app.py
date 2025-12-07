import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
)

# -------------------------------------------------------------------
# 1. Configuration: Waves, modes, defaults
# -------------------------------------------------------------------

BENCHMARK_TICKER = "SPY"
VIX_TICKER = "^VIX"

WAVES = {
    "AI_Wave": "AI_Wave",
    "Growth_Wave": "Growth_Wave",
    "FuturePower_Wave": "FuturePower_Wave",
    "Quantum_Wave": "Quantum_Wave",
    "CleanTransitInfra_Wave": "CleanTransitInfra_Wave",
    "SmallCapGrowth_Wave": "SmallCapGrowth_Wave",
    "SmallMidGrowth_Wave": "SmallMidGrowth_Wave",
    "Crypto_Wave": "Crypto_Wave",
    "SP500_Wave": "SP500_Wave",
}

# Fallback weights if CSV fails (toy defaults – CSV should overwrite these)
DEFAULT_WAVE_WEIGHTS = {
    "AI_Wave": [
        ("NVDA", 0.18),
        ("MSFT", 0.16),
        ("META", 0.14),
        ("GOOGL", 0.12),
        ("AMZN", 0.10),
        ("AVGO", 0.08),
        ("CRM", 0.08),
        ("PLTR", 0.07),
        ("AMD", 0.07),
    ],
    "Growth_Wave": [
        ("AAPL", 0.15),
        ("TSLA", 0.15),
        ("NFLX", 0.10),
        ("SHOP", 0.10),
        ("ABNB", 0.10),
        ("NOW", 0.10),
        ("ADBE", 0.10),
        ("SQ", 0.10),
        ("INTU", 0.10),
    ],
    "FuturePower_Wave": [
        ("ENPH", 0.15),
        ("FSLR", 0.15),
        ("NEE", 0.15),
        ("SEDG", 0.15),
        ("PLUG", 0.10),
        ("RUN", 0.10),
        ("DQ", 0.10),
        ("BLDP", 0.10),
    ],
    "Quantum_Wave": [
        ("IBM", 0.20),
        ("MSFT", 0.20),
        ("GOOGL", 0.15),
        ("AMZN", 0.15),
        ("AAPL", 0.10),
        ("NVDA", 0.10),
        ("QUBT", 0.10),
    ],
    "CleanTransitInfra_Wave": [
        ("TSLA", 0.18),
        ("NIO", 0.15),
        ("NEE", 0.15),
        ("F", 0.12),
        ("GM", 0.10),
        ("BLDR", 0.10),
        ("CAT", 0.10),
        ("VMC", 0.10),
    ],
    "SmallCapGrowth_Wave": [
        ("IWM", 1.0),  # placeholder small-cap ETF
    ],
    "SmallMidGrowth_Wave": [
        ("IJH", 1.0),  # placeholder mid-cap ETF
    ],
    "Crypto_Wave": [
        ("BTC-USD", 0.60),
        ("ETH-USD", 0.40),
    ],
    "SP500_Wave": [
        ("SPY", 1.0),
    ],
}

MODES = ["Standard", "Alpha-Minus-Beta", "Private Logic™"]


# -------------------------------------------------------------------
# 2. Data loading helpers
# -------------------------------------------------------------------

@st.cache_data
def load_weights(path: str = "wave_weights.csv") -> pd.DataFrame:
    """
    Load wave weights from CSV, clean them, and if anything goes wrong,
    fall back to DEFAULT_WAVE_WEIGHTS.
    CSV format required: wave,ticker,weight  (no extra columns).
    """
    # Build fallback df from defaults
    default_rows = []
    for wave, pairs in DEFAULT_WAVE_WEIGHTS.items():
        for ticker, w in pairs:
            default_rows.append({"wave": wave, "ticker": ticker, "weight": w})
    default_df = pd.DataFrame(default_rows, columns=["wave", "ticker", "weight"])

    try:
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except TypeError:
            # Older pandas without on_bad_lines
            df = pd.read_csv(path, error_bad_lines=False)
    except FileNotFoundError:
        st.warning("wave_weights.csv not found. Using code-managed default weights.")
        return default_df
    except Exception as e:
        st.warning(f"Could not parse wave_weights.csv. Using defaults. (Error: {e})")
        return default_df

    if df is None or df.empty:
        st.warning("wave_weights.csv is empty or malformed. Using default weights.")
        return default_df

    # Normalize column names (case / whitespace)
    col_map = {c.strip().lower(): c for c in df.columns}
    required = ["wave", "ticker", "weight"]
    for r in required:
        if r not in col_map:
            st.warning(
                "wave_weights.csv is missing required columns "
                "['wave', 'ticker', 'weight']. Using code-managed default weights."
            )
            return default_df

    wave_col = col_map["wave"]
    ticker_col = col_map["ticker"]
    weight_col = col_map["weight"]

    # Clean values
    df[wave_col] = df[wave_col].astype(str).str.strip()
    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")

    df = df.dropna(subset=[weight_col])

    # Normalize weights per wave so they sum to 1
    df["weight_norm"] = (
        df.groupby(wave_col)[weight_col].transform(lambda x: x / x.sum())
    )

    clean_df = df.rename(
        columns={wave_col: "wave", ticker_col: "ticker", "weight_norm": "weight"}
    )[["wave", "ticker", "weight"]]

    return clean_df


@st.cache_data
def fetch_price_history(tickers, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers using yfinance.
    Returns a DataFrame with Date index and tickers as columns.
    """
    if len(tickers) == 0:
        return pd.DataFrame()

    data = yf.download(
        tickers=list(set(tickers)),
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        data = data["Adj Close"]

    data = data.sort_index()
    return data


@st.cache_data
def fetch_vix_history(start: datetime, end: datetime) -> pd.DataFrame:
    data = yf.download(
        VIX_TICKER,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        data = data[["Close"]].rename(columns={"Close": "VIX"})
    return data.sort_index()


# -------------------------------------------------------------------
# 3. Portfolio & alpha math
# -------------------------------------------------------------------

def compute_wave_timeseries(
    wave_name: str,
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str = BENCHMARK_TICKER,
) -> dict:
    """
    Build equity curves and daily returns for a single wave vs benchmark.
    Returns dict with:
        equity_df: DataFrame [wave_equity, benchmark_equity]
        returns_df: DataFrame [wave_ret, bench_ret]
    """
    if prices is None or prices.empty:
        return {"equity_df": pd.DataFrame(), "returns_df": pd.DataFrame()}

    wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
    if wave_weights.empty:
        return {"equity_df": pd.DataFrame(), "returns_df": pd.DataFrame()}

    tickers = [t for t in wave_weights["ticker"].unique() if t in prices.columns]
    if len(tickers) == 0 or benchmark_ticker not in prices.columns:
        return {"equity_df": pd.DataFrame(), "returns_df": pd.DataFrame()}

    w = wave_weights.set_index("ticker")["weight"]
    w = w.loc[[t for t in tickers if t in w.index]]
    w = w / w.sum()

    wave_prices = prices[tickers]
    bench_prices = prices[[benchmark_ticker]]

    # Compute normalized equity curves (start at 1.0)
    wave_equity = (wave_prices.pct_change().fillna(0).dot(w) + 1.0).cumprod()
    bench_equity = (bench_prices[benchmark_ticker].pct_change().fillna(0) + 1.0).cumprod()

    equity_df = pd.DataFrame(
        {
            "wave": wave_equity,
            "benchmark": bench_equity,
        }
    )

    # Daily returns
    returns_df = equity_df.pct_change().dropna()
    returns_df.columns = ["wave_ret", "bench_ret"]

    return {"equity_df": equity_df, "returns_df": returns_df}


def compute_alpha_windows_from_equity(equity_df: pd.DataFrame, beta: float = 1.0) -> dict:
    """
    Given an equity curve DataFrame with columns [wave, benchmark],
    compute alpha over 30D, 60D, 6M, and 1Y windows for THIS wave only.
    Alpha = WaveReturn - beta * BenchmarkReturn for each window.
    Returns values in percent (e.g., 1.23 = +1.23% alpha).
    """
    if equity_df is None or equity_df.empty:
        return {"30D": np.nan, "60D": np.nan, "6M": np.nan, "1Y": np.nan}

    wave_col, bench_col = equity_df.columns[:2]

    def window_alpha(days: int) -> float:
        if len(equity_df) < 2:
            return np.nan
        window_len = min(days, len(equity_df))
        sub = equity_df.iloc[-window_len:]

        wave_ret = sub[wave_col].iloc[-1] / sub[wave_col].iloc[0] - 1.0
        bench_ret = sub[bench_col].iloc[-1] / sub[bench_col].iloc[0] - 1.0

        return (wave_ret - beta * bench_ret) * 100.0

    return {
        "30D": window_alpha(30),
        "60D": window_alpha(60),
        "6M": window_alpha(126),   # ~6 months trading days
        "1Y": window_alpha(252),   # ~1 year trading days
    }


def compute_realized_beta(returns_df: pd.DataFrame) -> float:
    if returns_df is None or returns_df.empty:
        return np.nan
    x = returns_df["bench_ret"].values
    y = returns_df["wave_ret"].values
    if len(x) < 2 or np.var(x) == 0:
        return np.nan
    cov = np.cov(y, x)[0, 1]
    var = np.var(x)
    return cov / var if var != 0 else np.nan


def mode_exposure(mode: str) -> float:
    """
    Simple exposure targets per mode.
    Returns equity exposure (0–1). SmartSafe = 1 - exposure.
    """
    if mode == "Alpha-Minus-Beta":
        return 0.70
    elif mode == "Private Logic™":
        return 0.85
    return 0.75  # Standard


# -------------------------------------------------------------------
# 4. Streamlit UI
# -------------------------------------------------------------------

def main():
    # Sidebar controls
    st.sidebar.header("Wave & Mode")

    wave_name = st.sidebar.selectbox(
        "Select Wave",
        options=list(WAVES.keys()),
        index=list(WAVES.keys()).index("SP500_Wave") if "SP500_Wave" in WAVES else 0,
    )

    mode = st.sidebar.selectbox("Mode", MODES, index=0)

    lookback_days = st.sidebar.slider(
        "Lookback (trading days)",
        min_value=60,
        max_value=365,
        value=365,
    )

    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # Date range
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=int(lookback_days * 1.5))

    # Load weights and prices
    weights_df = load_weights("wave_weights.csv")
    tickers_for_wave = weights_df[weights_df["wave"] == wave_name]["ticker"].tolist()
    all_tickers = list(set(tickers_for_wave + [BENCHMARK_TICKER]))

    prices = fetch_price_history(all_tickers, start_date, end_date)
    vix_df = fetch_vix_history(start_date, end_date)

    # Compute wave vs benchmark series
    series = compute_wave_timeseries(wave_name, weights_df, prices, BENCHMARK_TICKER)
    equity_df = series["equity_df"]
    returns_df = series["returns_df"]

    latest_vix = float(vix_df["VIX"].iloc[-1]) if not vix_df.empty else np.nan

    # Top layout
    st.markdown("## WAVES Institutional Console")
    st.caption(
        "Adaptive Portfolio Waves • AIWs/APWs • SmartSafe™ • VIX-gated risk • "
        "Alpha-Minus-Beta & Private Logic™"
    )

    cols_top = st.columns([1, 1, 1, 1, 1, 1])

    # Default values
    wave_today = np.nan
    bench_today = np.nan
    alpha_today = np.nan
    realized_beta = compute_realized_beta(returns_df)
    exposure_now = mode_exposure(mode)
    smartsafe_now = 1.0 - exposure_now

    if not returns_df.empty:
        last_ret = returns_df.iloc[-1]
        wave_today = last_ret["wave_ret"] * 100.0
        bench_today = last_ret["bench_ret"] * 100.0

        beta_for_alpha = realized_beta if not np.isnan(realized_beta) else 1.0
        alpha_today = (last_ret["wave_ret"] - beta_for_alpha * last_ret["bench_ret"]) * 100.0

    with cols_top[0]:
        st.metric("Wave Today", f"{wave_today:0.2f}%" if not np.isnan(wave_today) else "—")
    with cols_top[1]:
        st.metric(
            "Benchmark Today (SPY)",
            f"{bench_today:0.2f}%" if not np.isnan(bench_today) else "—",
        )
    with cols_top[2]:
        st.metric(
            "Today Alpha Captured",
            f"{alpha_today:0.2f}%" if not np.isnan(alpha_today) else "—",
        )
    with cols_top[3]:
        st.metric(
            "Realized Beta vs SPY",
            f"{realized_beta:0.2f}" if not np.isnan(realized_beta) else "—",
            help="Rolling beta computed from this wave's daily returns vs SPY.",
        )
    with cols_top[4]:
        st.metric("Current Exposure", f"{exposure_now * 100:0.1f}%")
    with cols_top[5]:
        st.metric(
            "SmartSafe™ Allocation Now",
            f"{smartsafe_now * 100:0.1f}%",
        )

    cols_vix = st.columns([1, 3])
    with cols_vix[0]:
        st.metric(
            "VIX (latest)",
            f"{latest_vix:0.1f}" if not np.isnan(latest_vix) else "—",
            help="CBOE Volatility Index (approximate fear gauge).",
        )
    with cols_vix[1]:
        st.caption(f"Engine Status: SANDBOX • Last refresh: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")

    st.markdown("---")

    # ----------------------------------------------------------------
    # Alpha Captured Windows (THIS WAVE ONLY) – NEW PER-WAVE LOGIC
    # ----------------------------------------------------------------
    alpha_windows = compute_alpha_windows_from_equity(
        equity_df, beta=realized_beta if not np.isnan(realized_beta) else 1.0
    )

    st.markdown("### Alpha Captured Windows (This Wave Only)")
    cols_alpha = st.columns(4)
    with cols_alpha[0]:
        val = alpha_windows["30D"]
        st.metric("30D", f"{val:0.2f}%" if not np.isnan(val) else "—")
    with cols_alpha[1]:
        val = alpha_windows["60D"]
        st.metric("60D", f"{val:0.2f}%" if not np.isnan(val) else "—")
    with cols_alpha[2]:
        val = alpha_windows["6M"]
        st.metric("6M", f"{val:0.2f}%" if not np.isnan(val) else "—")
    with cols_alpha[3]:
        val = alpha_windows["1Y"]
        st.metric("1Y", f"{val:0.2f}%" if not np.isnan(val) else "—")

    st.markdown("---")

    # ----------------------------------------------------------------
    # Charts: Equity curve and benchmark
    # ----------------------------------------------------------------
    col_chart, col_holdings = st.columns([2, 1])

    with col_chart:
        st.markdown(f"#### {wave_name} vs Benchmark (Equity Curves)")
        if equity_df.empty:
            st.info("No price data available for this wave / benchmark.")
        else:
            st.line_chart(equity_df.rename(columns={"wave": wave_name, "benchmark": "SPY"}))

    # ----------------------------------------------------------------
    # Top holdings (live)
    # ----------------------------------------------------------------
    with col_holdings:
        st.markdown("#### Top Holdings (Live)")
        if len(tickers_for_wave) == 0:
            st.info("No holdings configured for this wave.")
        else:
            # Fetch latest prices for just this wave's tickers
            latest_prices = yf.download(
                tickers=list(set(tickers_for_wave)),
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
            if isinstance(latest_prices, pd.DataFrame) and "Close" in latest_prices.columns:
                latest_prices = latest_prices["Close"]

            latest_row = latest_prices.iloc[-1] if isinstance(latest_prices, pd.DataFrame) and not latest_prices.empty else pd.Series(dtype=float)
            prev_row = latest_prices.iloc[-2] if isinstance(latest_prices, pd.DataFrame) and len(latest_prices) > 1 else None

            wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
            wave_weights = wave_weights.sort_values("weight", ascending=False)

            rows = []
            for _, r in wave_weights.iterrows():
                t = r["ticker"]
                w = r["weight"] * 100.0
                price_today = float(latest_row[t]) if t in latest_row.index else np.nan
                if prev_row is not None and t in prev_row.index and not np.isnan(price_today):
                    price_yday = float(prev_row[t])
                    todays_pct = (price_today / price_yday - 1.0) * 100.0 if price_yday != 0 else np.nan
                else:
                    todays_pct = np.nan

                google_url = f"https://www.google.com/finance/quote/{t}:NASDAQ"
                ticker_link = f"[{t}]({google_url})"

                rows.append(
                    {
                        "Ticker": ticker_link,
                        "Weight": f"{w:0.2f}%",
                        "Today%": f"{todays_pct:0.2f}%" if not np.isnan(todays_pct) else "—",
                    }
                )

            if rows:
                holdings_df = pd.DataFrame(rows)
                st.markdown(
                    holdings_df.to_markdown(index=False),
                    unsafe_allow_html=True,
                )
            else:
                st.info("No valid holdings found for this wave.")

    st.markdown("---")

    # ----------------------------------------------------------------
    # Benchmark & VIX charts
    # ----------------------------------------------------------------
    st.markdown("### Market Context")
    col_spy, col_vix = st.columns(2)

    if not prices.empty and BENCHMARK_TICKER in prices.columns:
        spy_equity = (prices[BENCHMARK_TICKER].pct_change().fillna(0) + 1.0).cumprod()
        with col_spy:
            st.markdown("#### SPY (Benchmark) – Price (Normalized)")
            st.line_chart(spy_equity.rename("SPY"))
    else:
        with col_spy:
            st.info("No SPY data available.")

    with col_vix:
        st.markdown("#### VIX – Level")
        if not vix_df.empty:
            st.line_chart(vix_df["VIX"])
        else:
            st.info("No VIX data available.")

    # ----------------------------------------------------------------
    # Debug section
    # ----------------------------------------------------------------
    if show_debug:
        st.markdown("### Debug Information")
        st.write("Selected wave:", wave_name)
        st.write("Mode:", mode)
        st.write("Lookback days:", lookback_days)
        st.write("Weights (this wave):", weights_df[weights_df["wave"] == wave_name])
        st.write("Equity DF (tail):", equity_df.tail())
        st.write("Returns DF (tail):", returns_df.tail())
        st.write("Alpha windows:", alpha_windows)


if __name__ == "__main__":
    main()
