"""
WAVES Institutional Console - Streamlit App

This application provides a comprehensive view of WAVES portfolio performance,
including holdings, returns, and performance metrics with robust fallback mechanisms.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAVE_WEIGHTS_FILE = os.path.join(BASE_DIR, "wave_weights.csv")
PRICES_CSV_FILE = os.path.join(BASE_DIR, "prices.csv")
LOGS_PERF_DIR = os.path.join(BASE_DIR, "logs", "performance")
LOGS_POS_DIR = os.path.join(BASE_DIR, "logs", "positions")

# Benchmark mapping
WAVE_BENCHMARKS = {
    "AI_Wave": "QQQ",
    "Growth_Wave": "SPY",
    "Income_Wave": "SPY",
    "Future_Energy_Wave": "SPY",
    "CleanTransitInfra_Wave": "SPY",
    "RWA_Income_Wave": "SPY",
    "SmallMidValue_Wave": "IWM",
    "Quantum_Wave": "QQQ",
}
DEFAULT_BENCHMARK = "SPY"

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class WaveEngineResult:
    """Result from portfolio engine calculations."""
    wave_name: str
    wave_return: float
    benchmark_return: float
    exposure: float
    smartsafe_allocation: float
    holdings: pd.DataFrame
    nav_series: Optional[pd.Series] = None
    benchmark_nav_series: Optional[pd.Series] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def load_weights(csv_path: str = WAVE_WEIGHTS_FILE) -> pd.DataFrame:
    """
    Load and normalize wave_weights.csv with robust error handling.
    
    Expected columns (case-insensitive): Wave, Ticker, Weight
    
    Returns:
        pd.DataFrame with columns: wave, ticker, weight
    """
    if not os.path.exists(csv_path):
        st.error(f"❌ Weights file not found: {csv_path}")
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"❌ Could not read weights file: {e}")
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    
    if df.empty:
        st.warning("⚠️ Weights file is empty")
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    
    # Build lowercase column mapping
    col_map = {c.strip().lower(): c for c in df.columns}
    
    required = ["wave", "ticker", "weight"]
    missing = [r for r in required if r not in col_map]
    
    if missing:
        st.error(f"❌ Missing required columns: {missing}. Found: {list(df.columns)}")
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    
    # Select and rename columns
    df = df[[col_map["wave"], col_map["ticker"], col_map["weight"]]].copy()
    df.columns = ["wave", "ticker", "weight"]
    
    # Clean data
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    
    # Normalize weights per wave
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )
    
    # Drop invalid rows
    df = df[(df["ticker"] != "") & (df["weight"] > 0)].copy()
    
    if df.empty:
        st.warning("⚠️ No valid rows after cleaning weights file")
    
    return df


def get_wave_holdings(weights_df: pd.DataFrame, wave_name: str) -> pd.DataFrame:
    """
    Extract holdings for a specific wave and normalize weights.
    
    Returns:
        pd.DataFrame with columns: ticker, weight
    """
    wave_df = weights_df[weights_df["wave"] == wave_name].copy()
    
    if wave_df.empty:
        return pd.DataFrame(columns=["ticker", "weight"])
    
    # Aggregate duplicate tickers
    wave_df = wave_df.groupby("ticker", as_index=False)["weight"].sum()
    
    # Normalize
    total = wave_df["weight"].sum()
    if total > 0:
        wave_df["weight"] = wave_df["weight"] / total
    
    return wave_df.sort_values("weight", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=300)
def fetch_yahoo_price_history(ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch price history from Yahoo Finance.
    
    Returns:
        pd.DataFrame with DatetimeIndex and 'Close' column, or None if failed
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            return None
        
        # Ensure we have Close column
        if "Close" not in data.columns:
            return None
        
        return data[["Close"]].copy()
    
    except Exception:
        return None


def load_prices_from_csv(ticker: str, csv_path: str = PRICES_CSV_FILE) -> Optional[pd.DataFrame]:
    """
    Load price history from local CSV file.
    
    Expected format: Date, Ticker, Close
    
    Returns:
        pd.DataFrame with DatetimeIndex and 'Close' column, or None if not found
    """
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        if "date" not in df.columns or "ticker" not in df.columns or "close" not in df.columns:
            return None
        
        # Filter for ticker
        ticker_df = df[df["ticker"].str.upper() == ticker.upper()].copy()
        
        if ticker_df.empty:
            return None
        
        # Parse dates and set index
        ticker_df["date"] = pd.to_datetime(ticker_df["date"])
        ticker_df = ticker_df.set_index("date").sort_index()
        ticker_df = ticker_df[["close"]].rename(columns={"close": "Close"})
        
        return ticker_df
    
    except Exception:
        return None


def generate_synthetic_prices(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Generate synthetic price data for demonstration purposes.
    
    Returns:
        pd.DataFrame with DatetimeIndex and 'Close' column
    """
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    
    # Generate random walk with slight upward drift
    np.random.seed(hash(ticker) % (2**32))
    returns = np.random.normal(0.0005, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({"Close": prices}, index=dates)
    return df


def fetch_price_history(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch price history with fallback mechanism:
    1. Try Yahoo Finance
    2. Try local prices.csv
    3. Generate synthetic data
    
    Returns:
        pd.DataFrame with DatetimeIndex and 'Close' column
    """
    # Try Yahoo Finance first
    df = fetch_yahoo_price_history(ticker, days)
    if df is not None and not df.empty:
        return df
    
    # Try local CSV
    df = load_prices_from_csv(ticker)
    if df is not None and not df.empty:
        return df
    
    # Fall back to synthetic data
    return generate_synthetic_prices(ticker, days)


def compute_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Compute daily returns from price series.
    
    Args:
        prices: DataFrame with 'Close' column
    
    Returns:
        pd.Series of daily returns
    """
    return prices["Close"].pct_change().fillna(0.0)


def align_dates(*dfs: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to common date index.
    
    Returns:
        List of aligned DataFrames
    """
    if not dfs:
        return []
    
    # Find common dates
    common_dates = dfs[0].index
    for df in dfs[1:]:
        common_dates = common_dates.intersection(df.index)
    
    # Align all DataFrames
    return [df.loc[common_dates].sort_index() for df in dfs]


def compute_portfolio_engine(
    wave_name: str,
    holdings_df: pd.DataFrame,
    mode: str = "Live"
) -> WaveEngineResult:
    """
    Compute portfolio performance metrics.
    
    Args:
        wave_name: Name of the wave
        holdings_df: DataFrame with columns: ticker, weight
        mode: "Live" or "Demo"
    
    Returns:
        WaveEngineResult with computed metrics
    """
    if holdings_df.empty:
        return WaveEngineResult(
            wave_name=wave_name,
            wave_return=0.0,
            benchmark_return=0.0,
            exposure=0.0,
            smartsafe_allocation=0.0,
            holdings=pd.DataFrame(columns=["ticker", "weight", "today_pct_change", "google_finance"])
        )
    
    # Fetch price histories for all holdings
    price_data = {}
    for ticker in holdings_df["ticker"]:
        prices = fetch_price_history(ticker, days=365)
        if prices is not None and not prices.empty:
            price_data[ticker] = prices
    
    # Get benchmark
    benchmark_ticker = WAVE_BENCHMARKS.get(wave_name, DEFAULT_BENCHMARK)
    benchmark_prices = fetch_price_history(benchmark_ticker, days=365)
    
    # Align all price series
    if price_data and benchmark_prices is not None:
        all_prices = list(price_data.values()) + [benchmark_prices]
        aligned = align_dates(*all_prices)
        
        if aligned:
            # Update price_data with aligned series
            for i, ticker in enumerate(price_data.keys()):
                price_data[ticker] = aligned[i]
            benchmark_prices = aligned[-1]
    
    # Compute returns
    returns_data = {}
    for ticker, prices in price_data.items():
        returns_data[ticker] = compute_returns(prices)
    
    # Compute portfolio returns
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        
        # Weight returns by portfolio weights
        weights_dict = dict(zip(holdings_df["ticker"], holdings_df["weight"]))
        weighted_returns = pd.Series(0.0, index=returns_df.index)
        
        for ticker in returns_df.columns:
            if ticker in weights_dict:
                weighted_returns += returns_df[ticker] * weights_dict[ticker]
        
        # Compute cumulative returns
        portfolio_nav = (1 + weighted_returns).cumprod()
        wave_return = portfolio_nav.iloc[-1] - 1.0 if len(portfolio_nav) > 0 else 0.0
    else:
        portfolio_nav = pd.Series([1.0])
        wave_return = 0.0
    
    # Compute benchmark returns
    if benchmark_prices is not None and not benchmark_prices.empty:
        benchmark_returns = compute_returns(benchmark_prices)
        benchmark_nav = (1 + benchmark_returns).cumprod()
        benchmark_return = benchmark_nav.iloc[-1] - 1.0 if len(benchmark_nav) > 0 else 0.0
    else:
        benchmark_nav = pd.Series([1.0])
        benchmark_return = 0.0
    
    # Compute today's price changes
    holdings_with_metrics = holdings_df.copy()
    holdings_with_metrics["today_pct_change"] = 0.0
    holdings_with_metrics["google_finance"] = ""
    
    for idx, row in holdings_with_metrics.iterrows():
        ticker = row["ticker"]
        
        # Get today's change
        if ticker in returns_data and len(returns_data[ticker]) > 0:
            today_return = returns_data[ticker].iloc[-1]
            holdings_with_metrics.at[idx, "today_pct_change"] = today_return
        
        # Google Finance link
        holdings_with_metrics.at[idx, "google_finance"] = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
    
    # Compute exposure (for demo, use 100% or based on mode)
    exposure = 1.0 if mode == "Live" else 0.5
    smartsafe_allocation = 1.0 - exposure
    
    return WaveEngineResult(
        wave_name=wave_name,
        wave_return=wave_return,
        benchmark_return=benchmark_return,
        exposure=exposure,
        smartsafe_allocation=smartsafe_allocation,
        holdings=holdings_with_metrics,
        nav_series=portfolio_nav,
        benchmark_nav_series=benchmark_nav
    )


# =============================================================================
# STREAMLIT UI
# =============================================================================

def render_sidebar() -> Tuple[str, str]:
    """
    Render sidebar and return selected wave and mode.
    
    Returns:
        (wave_name, mode)
    """
    st.sidebar.title("🌊 WAVES Console")
    st.sidebar.markdown("---")
    
    # Load available waves
    weights_df = load_weights()
    
    if weights_df.empty:
        st.sidebar.error("❌ No waves available")
        return "", "Live"
    
    available_waves = sorted(weights_df["wave"].unique().tolist())
    
    # Wave selection
    wave_name = st.sidebar.selectbox(
        "Select Wave",
        options=available_waves,
        index=0 if available_waves else None
    )
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Mode",
        options=["Live", "Demo"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Live Mode**: Real-time data with fallback to local CSV or synthetic data\n\n"
        "**Demo Mode**: Uses synthetic data for demonstration"
    )
    
    return wave_name, mode


def render_performance_metrics(result: WaveEngineResult):
    """Render performance metrics in columns."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Wave Return",
            f"{result.wave_return * 100:.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Benchmark Return",
            f"{result.benchmark_return * 100:.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Exposure",
            f"{result.exposure * 100:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            "SmartSafe",
            f"{result.smartsafe_allocation * 100:.1f}%",
            delta=None
        )


def render_holdings_table(holdings: pd.DataFrame):
    """Render top holdings table with styling."""
    if holdings.empty:
        st.warning("⚠️ No holdings data available")
        return
    
    # Prepare display DataFrame
    display_df = holdings.copy()
    
    # Format columns
    display_df["Weight"] = (display_df["weight"] * 100).map("{:.2f}%".format)
    display_df["Today % Change"] = (display_df["today_pct_change"] * 100).map("{:.2f}%".format)
    display_df["Ticker"] = display_df["ticker"]
    
    # Create clickable links for Google Finance
    display_df["Google Finance"] = display_df["google_finance"].apply(
        lambda x: f"[View]({x})" if x else ""
    )
    
    # Select and order columns
    display_df = display_df[["Ticker", "Weight", "Today % Change", "Google Finance"]]
    
    st.subheader("📊 Top Holdings")
    st.markdown(display_df.head(10).to_markdown(index=False))


def render_performance_chart(result: WaveEngineResult):
    """Render performance chart."""
    if result.nav_series is None or result.benchmark_nav_series is None:
        st.info("ℹ️ Performance chart not available")
        return
    
    st.subheader("📈 Performance Chart")
    
    # Create chart data
    chart_data = pd.DataFrame({
        "Wave": (result.nav_series - 1) * 100,
        "Benchmark": (result.benchmark_nav_series - 1) * 100
    })
    
    st.line_chart(chart_data)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="WAVES Institutional Console",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌊 WAVES Institutional Console")
    st.markdown("*Real-time portfolio analytics with intelligent data fallback*")
    st.markdown("---")
    
    # Sidebar
    wave_name, mode = render_sidebar()
    
    if not wave_name:
        st.error("❌ Please select a valid wave from the sidebar")
        return
    
    # Load weights and get holdings
    weights_df = load_weights()
    holdings_df = get_wave_holdings(weights_df, wave_name)
    
    if holdings_df.empty:
        st.error(f"❌ No holdings found for wave: {wave_name}")
        return
    
    # Display wave info
    st.header(f"Wave: {wave_name}")
    st.caption(f"Mode: {mode} | Holdings: {len(holdings_df)}")
    
    # Compute portfolio metrics
    with st.spinner("Computing portfolio metrics..."):
        result = compute_portfolio_engine(wave_name, holdings_df, mode)
    
    # Render UI sections
    render_performance_metrics(result)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_performance_chart(result)
    
    with col2:
        render_holdings_table(result.holdings)
    
    # Footer
    st.markdown("---")
    st.caption(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Data source: Yahoo Finance with fallback to local/synthetic data"
    )


if __name__ == "__main__":
    main()
