# Sandbox app demonstrating graceful handling of missing/delisted tickers
# This prevents infinite rerun loops caused by data fetch failures

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("Sandbox: Safe Price Fetching Demo")
st.markdown("Demonstrates graceful handling of missing/delisted tickers (ARB-USD, MNT-USD, IMX-USD)")

# Sample tickers including known problematic ones
SAMPLE_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT",  # Known good tickers
    "ARB-USD", "MNT-USD", "IMX-USD",  # Known problematic tickers
    "INVALID-TICKER"  # Definitely invalid
]

# Configuration
MAX_DISPLAY_TICKERS = 15  # Maximum number of missing tickers to show in warning message

# User controls
if st.button("Fetch Price Data"):
    st.info("Starting price data fetch...")
    
    # Track missing tickers
    missing_tickers = []
    successful_data = {}
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    st.write(f"Fetching data from {start_date.date()} to {end_date.date()}")
    
    # Fetch data with error handling
    for ticker in SAMPLE_TICKERS:
        try:
            st.write(f"Processing {ticker}...")
            
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Validate the returned dataframe is not empty
            if data is None or data.empty:
                st.write(f"  ⚠️ Empty response for {ticker}")
                missing_tickers.append(ticker)
                continue
            
            # Check if Close column exists and has valid data
            if "Close" not in data.columns:
                st.write(f"  ⚠️ No Close price column for {ticker}")
                missing_tickers.append(ticker)
                continue
            
            # Check if all values are NaN
            if data["Close"].isna().all():
                st.write(f"  ⚠️ All Close prices are NaN for {ticker}")
                missing_tickers.append(ticker)
                continue
            
            # If we get here, data is valid
            successful_data[ticker] = data
            st.write(f"  ✅ Successfully fetched {len(data)} rows for {ticker}")
            
        except Exception as e:
            # Catch any unexpected errors and skip ticker
            st.write(f"  ❌ Unexpected error for {ticker}: {str(e)}")
            missing_tickers.append(ticker)
            continue
    
    # Display results
    st.success(f"✅ Fetch complete! Successfully loaded {len(successful_data)} tickers")
    
    # Show warning for missing tickers
    if missing_tickers:
        display_tickers = missing_tickers[:MAX_DISPLAY_TICKERS]
        ticker_list = ', '.join(display_tickers)
        suffix = " ..." if len(missing_tickers) > MAX_DISPLAY_TICKERS else ""
        st.warning(f"Skipped unavailable tickers: {ticker_list}{suffix}")
    
    # Display summary table
    if successful_data:
        st.subheader("Successfully Loaded Tickers")
        summary_data = []
        for ticker, df in successful_data.items():
            # Safely get last close price (should always exist due to validation)
            last_close = df['Close'].dropna().iloc[-1] if not df['Close'].dropna().empty else 0
            summary_data.append({
                "Ticker": ticker,
                "Rows": len(df),
                "First Date": df.index[0].strftime("%Y-%m-%d"),
                "Last Date": df.index[-1].strftime("%Y-%m-%d"),
                "Last Close": f"${last_close:.2f}"
            })
        st.dataframe(pd.DataFrame(summary_data))
    
    # Display missing tickers summary
    if missing_tickers:
        st.subheader("Missing/Unavailable Tickers")
        st.write(f"Total: {len(missing_tickers)}")
        st.write(missing_tickers)

st.markdown("---")
st.markdown("**Key Features:**")
st.markdown("- ✅ No `raise KeyError` - all errors are caught and handled")
st.markdown("- ✅ Validates dataframe is not empty before processing")
st.markdown("- ✅ Checks for missing columns and all-NaN data")
st.markdown("- ✅ Continues processing remaining tickers on error")
st.markdown("- ✅ Displays clear warning for unavailable tickers")
st.markdown("- ✅ No auto-refresh - manual button click only")