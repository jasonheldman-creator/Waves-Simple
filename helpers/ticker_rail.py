"""
V3 ADD-ON: Bottom Ticker (Institutional Rail) - Rendering Logic
Main logic for ticker aggregation and HTML rendering.
"""

from typing import List, Dict
import streamlit as st
from .ticker_sources import (
    get_wave_holdings_tickers,
    get_ticker_price_data,
    get_earnings_date,
    get_fed_indicators,
    get_waves_status,
    update_cache_with_current_data
)


# ============================================================================
# SECTION 1: Ticker Item Formatting
# ============================================================================

def format_ticker_item(ticker: str, include_earnings: bool = True) -> str:
    """
    Format a single ticker item with price and % change.
    
    Args:
        ticker: Stock ticker symbol
        include_earnings: Whether to include earnings date
    
    Returns:
        Formatted ticker string
    """
    try:
        # Get price data
        price_data = get_ticker_price_data(ticker)
        
        if price_data['success'] and price_data['price'] is not None:
            price = price_data['price']
            change_pct = price_data['change_pct']
            
            # Format percentage with color indicator
            if change_pct > 0:
                pct_str = f"+{change_pct:.2f}%"
                color_symbol = "🟢"
            elif change_pct < 0:
                pct_str = f"{change_pct:.2f}%"
                color_symbol = "🔴"
            else:
                pct_str = "0.00%"
                color_symbol = "⚪"
            
            base_str = f"{ticker}: ${price:.2f} {color_symbol}{pct_str}"
            
            # Add earnings date if requested
            if include_earnings:
                earnings = get_earnings_date(ticker)
                if earnings:
                    base_str += f" (E:{earnings})"
            
            return base_str
        else:
            # Fallback: symbol only
            return f"{ticker}: --"
            
    except Exception:
        return f"{ticker}: --"


def format_fed_macro_items() -> List[str]:
    """
    Format Fed and macro indicator items.
    
    Returns:
        List of formatted strings for Fed/macro data
    """
    items = []
    
    try:
        fed_data = get_fed_indicators()
        
        # Fed Funds Rate
        if fed_data.get('fed_funds_rate'):
            items.append(f"FED FUNDS: {fed_data['fed_funds_rate']}")
        
        # Next FOMC Meeting
        if fed_data.get('next_fomc_date'):
            items.append(f"NEXT FOMC: {fed_data['next_fomc_date']}")
        
        # CPI placeholder
        if fed_data.get('cpi_latest'):
            items.append(f"CPI: {fed_data['cpi_latest']}")
        
        # Jobs placeholder
        if fed_data.get('jobs_latest'):
            items.append(f"JOBS: {fed_data['jobs_latest']}")
            
    except Exception:
        items.append("FED DATA: N/A")
    
    return items


def format_waves_status_items() -> List[str]:
    """
    Format WAVES internal status items.
    
    Returns:
        List of formatted strings for WAVES status
    """
    items = []
    
    try:
        status = get_waves_status()
        
        # System status
        if status.get('system_status'):
            items.append(f"WAVES: {status['system_status']}")
        
        # Last update time
        if status.get('last_update'):
            items.append(f"UPDATED: {status['last_update']}")
        
        # Waves loading status
        if status.get('waves_status'):
            items.append(f"UNIVERSE: {status['waves_status']}")
            
    except Exception:
        items.append("WAVES: ONLINE")
    
    return items


# ============================================================================
# SECTION 2: Ticker Universe Aggregation
# ============================================================================

def build_ticker_universe(
    max_tickers: int = 60,
    top_n_per_wave: int = 5,
    sample_size: int = 15
) -> List[str]:
    """
    Build the complete ticker universe from all sources.
    Enhanced with partial data handling - continues with available data even if some tickers fail.
    
    Args:
        max_tickers: Maximum unique tickers from holdings
        top_n_per_wave: Top holdings per wave
        sample_size: Number of tickers to include in the display rotation
    
    Returns:
        List of formatted ticker items
    """
    ticker_items = []
    successful_tickers = []
    failed_count = 0
    
    try:
        # Get ticker symbols from wave holdings
        tickers = get_wave_holdings_tickers(
            max_tickers=max_tickers,
            top_n_per_wave=top_n_per_wave
        )
        
        # Sample a subset for display (to avoid overwhelming the ticker)
        # Rotate through different tickers on each refresh
        display_tickers = tickers[:sample_size]
        
        # Format each ticker with price and % change
        # Continue even if some tickers fail
        for ticker in display_tickers:
            try:
                ticker_item = format_ticker_item(ticker, include_earnings=False)
                # Only add if we got meaningful data (not just "--")
                if ticker_item and ": --" not in ticker_item:
                    ticker_items.append(ticker_item)
                    successful_tickers.append(ticker)
                else:
                    failed_count += 1
            except Exception:
                # Skip this ticker and continue
                failed_count += 1
                continue
        
        # Add a few earnings highlights (first 3 successful tickers)
        for ticker in successful_tickers[:3]:
            try:
                earnings = get_earnings_date(ticker)
                if earnings:
                    ticker_items.append(f"{ticker} EARNINGS: {earnings}")
            except Exception:
                # Skip if earnings fetch fails
                continue
        
        # Add Fed/Macro indicators (always try to include)
        try:
            fed_items = format_fed_macro_items()
            ticker_items.extend(fed_items)
        except Exception:
            # Continue without Fed data if it fails
            pass
        
        # Add WAVES status
        try:
            status_items = format_waves_status_items()
            ticker_items.extend(status_items)
        except Exception:
            # Add minimal status if fetch fails
            ticker_items.append("WAVES: ONLINE")
        
        # If we have some failed tickers, add a status indicator
        if failed_count > 0:
            ticker_items.append(f"⚠️ {failed_count} TICKERS UNAVAILABLE")
        
    except Exception:
        # Minimal fallback
        ticker_items = [
            "WAVES: ONLINE",
            "MARKET DATA: LOADING"
        ]
    
    return ticker_items


# ============================================================================
# SECTION 3: HTML Ticker Bar Rendering
# ============================================================================

def render_ticker_rail_html(ticker_items: List[str]) -> str:
    """
    Generate HTML for the scrolling ticker bar.
    
    Args:
        ticker_items: List of formatted ticker strings
    
    Returns:
        HTML string for the ticker bar
    """
    # Join ticker items with separator
    ticker_text = " • ".join(ticker_items)
    
    # Duplicate ticker text for seamless loop (3x for smooth scrolling)
    ticker_text_full = f"{ticker_text} • {ticker_text} • {ticker_text}"
    
    # Generate HTML with CSS animation
    html = f"""
    <style>
        @keyframes scroll-left {{
            0% {{
                transform: translateX(0);
            }}
            100% {{
                transform: translateX(-33.333%);
            }}
        }}
        
        .ticker-bar-v3 {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
            color: white;
            padding: 12px 0;
            z-index: 9999;
            overflow: hidden;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.3);
            font-family: 'Courier New', monospace;
            font-size: 14px;
            font-weight: bold;
            border-top: 2px solid rgba(255, 255, 255, 0.2);
        }}
        
        .ticker-content-v3 {{
            display: inline-block;
            white-space: nowrap;
            animation: scroll-left 90s linear infinite;
            padding-left: 100%;
        }}
        
        .ticker-content-v3:hover {{
            animation-play-state: paused;
        }}
        
        /* Add padding to main content to prevent overlap */
        .main .block-container {{
            padding-bottom: 60px !important;
        }}
        
        /* Additional styling for better visibility */
        .ticker-bar-v3::before {{
            content: "📊 INSTITUTIONAL RAIL";
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(0, 0, 0, 0.3);
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 11px;
            letter-spacing: 1px;
        }}
    </style>
    
    <div class="ticker-bar-v3">
        <div class="ticker-content-v3">
            {ticker_text_full}
        </div>
    </div>
    """
    
    return html


# ============================================================================
# SECTION 4: Main Render Function
# ============================================================================

def render_bottom_ticker_v3(
    max_tickers: int = 60,
    top_n_per_wave: int = 5,
    sample_size: int = 15
) -> None:
    """
    Main function to render the V3 bottom ticker bar.
    Handles all aggregation, formatting, and rendering with graceful degradation.
    
    Args:
        max_tickers: Maximum unique tickers from holdings
        top_n_per_wave: Top holdings per wave
        sample_size: Number of tickers to display in rotation
    """
    try:
        # Update cache in background (non-blocking)
        try:
            update_cache_with_current_data()
        except Exception:
            pass  # Don't fail if cache update fails
        
        # Build ticker universe
        ticker_items = build_ticker_universe(
            max_tickers=max_tickers,
            top_n_per_wave=top_n_per_wave,
            sample_size=sample_size
        )
        
        # Generate HTML
        ticker_html = render_ticker_rail_html(ticker_items)
        
        # Render to Streamlit
        st.markdown(ticker_html, unsafe_allow_html=True)
        
    except Exception as e:
        # Fail silently - don't disrupt the app if ticker bar fails
        # Optionally log error for debugging
        try:
            # Minimal fallback ticker
            fallback_html = """
            <style>
                .ticker-bar-fallback {
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    background: #1e3a8a;
                    color: white;
                    padding: 12px;
                    z-index: 9999;
                    text-align: center;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                }
            </style>
            <div class="ticker-bar-fallback">
                WAVES INSTITUTIONAL CONSOLE • ONLINE
            </div>
            """
            st.markdown(fallback_html, unsafe_allow_html=True)
        except Exception:
            pass  # Ultimate fail-safe: do nothing
