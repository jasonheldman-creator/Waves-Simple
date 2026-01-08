#!/usr/bin/env python3
"""
Demonstration of Crypto Volatility Overlay (Phase 1B.2)

This script shows how to use the crypto volatility overlay module
to compute regime-based exposure scaling for crypto portfolios.

Usage:
    python demo_crypto_volatility_overlay.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from helpers.crypto_volatility_overlay import compute_crypto_overlay
    print("✓ Successfully imported crypto_volatility_overlay")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)


def create_demo_price_data():
    """Create synthetic demo price data for BTC and ETH."""
    print("\n" + "=" * 70)
    print("Creating synthetic demo price data...")
    print("=" * 70)
    
    # Generate 120 days of price data
    days = 120
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # BTC: Moderate volatility, minor drawdown
    np.random.seed(42)
    btc_returns = np.random.normal(0.001, 0.025, days)  # ~60% annualized vol
    btc_prices = 40000 * np.exp(np.cumsum(btc_returns))
    
    # ETH: Higher volatility, moderate drawdown
    np.random.seed(43)
    eth_returns = np.random.normal(0.0005, 0.030, days)  # ~75% annualized vol
    eth_prices = 2500 * np.exp(np.cumsum(eth_returns))
    
    price_data = pd.DataFrame({
        'BTC-USD': btc_prices,
        'ETH-USD': eth_prices
    }, index=dates)
    
    print(f"\nGenerated {days} days of price data:")
    print(f"  BTC-USD: ${price_data['BTC-USD'].iloc[-1]:,.2f} (latest)")
    print(f"  ETH-USD: ${price_data['ETH-USD'].iloc[-1]:,.2f} (latest)")
    
    return price_data


def demo_basic_usage():
    """Demonstrate basic usage of compute_crypto_overlay."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Usage")
    print("=" * 70)
    
    # Create synthetic price data
    price_data = create_demo_price_data()
    
    # Compute overlay
    print("\nComputing crypto volatility overlay...")
    overlay = compute_crypto_overlay(
        benchmarks=['BTC-USD', 'ETH-USD'],
        price_data=price_data,
        vol_window=30,
        dd_window=60
    )
    
    # Display results
    print("\n" + "-" * 70)
    print("OVERLAY RESULTS")
    print("-" * 70)
    print(f"  Label:          {overlay['overlay_label']}")
    print(f"  Regime:         {overlay['regime']}")
    print(f"  Exposure:       {overlay['exposure']:.1%}")
    print(f"  Volatility:     {overlay['volatility']:.1%} (annualized)")
    print(f"  Max Drawdown:   {overlay['max_drawdown']:.1%}")
    print(f"  Vol Regime:     {overlay['vol_regime']}")
    print(f"  DD Severity:    {overlay['dd_severity']}")
    print("-" * 70)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if overlay['regime'] == 'LOW':
        print("  ✓ Low volatility regime - Full exposure recommended (100%)")
    elif overlay['regime'] == 'MED':
        print("  ⚠ Medium volatility regime - Moderate exposure (75%)")
    elif overlay['regime'] == 'HIGH':
        print("  ⚠ High volatility regime - Reduced exposure (50%)")
    elif overlay['regime'] == 'CRISIS':
        print("  ✗ CRISIS regime - Minimum exposure (20%)")
    
    return overlay


def demo_different_scenarios():
    """Demonstrate different market scenarios."""
    print("\n" + "=" * 70)
    print("DEMO 2: Different Market Scenarios")
    print("=" * 70)
    
    scenarios = [
        ("Low Volatility Bull Market", 0.001, 0.015, 100),
        ("Normal Volatility Consolidation", 0.0, 0.025, 100),
        ("High Volatility Bear Market", -0.002, 0.040, 100),
    ]
    
    for scenario_name, drift, vol, days in scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 70)
        
        # Generate scenario-specific data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = np.random.normal(drift, vol, days)
        
        btc_prices = 40000 * np.exp(np.cumsum(returns))
        eth_prices = 2500 * np.exp(np.cumsum(returns * 1.2))  # ETH more volatile
        
        price_data = pd.DataFrame({
            'BTC-USD': btc_prices,
            'ETH-USD': eth_prices
        }, index=dates)
        
        # Compute overlay
        overlay = compute_crypto_overlay(
            benchmarks=['BTC-USD', 'ETH-USD'],
            price_data=price_data,
            vol_window=30,
            dd_window=60
        )
        
        print(f"  Regime:   {overlay['regime']:>8}")
        print(f"  Exposure: {overlay['exposure']:>7.1%}")
        print(f"  Vol:      {overlay['volatility']:>7.1%}")
        print(f"  DD:       {overlay['max_drawdown']:>7.1%}")


def demo_single_benchmark():
    """Demonstrate using a single benchmark (BTC only)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Single Benchmark (BTC only)")
    print("=" * 70)
    
    # Create BTC-only data
    days = 100
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(42)
    btc_returns = np.random.normal(0.001, 0.025, days)
    btc_prices = 40000 * np.exp(np.cumsum(btc_returns))
    
    price_data = pd.DataFrame({
        'BTC-USD': btc_prices
    }, index=dates)
    
    print(f"\nUsing BTC-USD only as benchmark")
    
    # Compute overlay
    overlay = compute_crypto_overlay(
        benchmarks=['BTC-USD'],  # Single benchmark
        price_data=price_data,
        vol_window=30,
        dd_window=60
    )
    
    print(f"\n  Regime:   {overlay['regime']}")
    print(f"  Exposure: {overlay['exposure']:.1%}")
    print(f"  Vol:      {overlay['volatility']:.1%}")


def demo_custom_windows():
    """Demonstrate using custom volatility and drawdown windows."""
    print("\n" + "=" * 70)
    print("DEMO 4: Custom Lookback Windows")
    print("=" * 70)
    
    price_data = create_demo_price_data()
    
    window_configs = [
        ("Short-term (14D/30D)", 14, 30),
        ("Default (30D/60D)", 30, 60),
        ("Long-term (60D/90D)", 60, 90),
    ]
    
    for config_name, vol_window, dd_window in window_configs:
        print(f"\n{config_name}:")
        
        overlay = compute_crypto_overlay(
            benchmarks=['BTC-USD', 'ETH-USD'],
            price_data=price_data,
            vol_window=vol_window,
            dd_window=dd_window
        )
        
        print(f"  Regime:   {overlay['regime']:>8}")
        print(f"  Exposure: {overlay['exposure']:>7.1%}")
        print(f"  Vol:      {overlay['volatility']:>7.1%}")


def demo_portfolio_integration():
    """Demonstrate how to integrate overlay into portfolio management."""
    print("\n" + "=" * 70)
    print("DEMO 5: Portfolio Integration Example")
    print("=" * 70)
    
    price_data = create_demo_price_data()
    
    # Compute overlay
    overlay = compute_crypto_overlay(
        benchmarks=['BTC-USD', 'ETH-USD'],
        price_data=price_data,
        vol_window=30,
        dd_window=60
    )
    
    # Example portfolio allocation
    base_allocation = {
        'BTC-USD': 0.40,
        'ETH-USD': 0.30,
        'SOL-USD': 0.20,
        'AVAX-USD': 0.10
    }
    
    print("\nBase Portfolio Allocation:")
    for ticker, weight in base_allocation.items():
        print(f"  {ticker:>10}: {weight:>6.1%}")
    
    # Apply overlay scaling
    scaled_allocation = {
        ticker: weight * overlay['exposure']
        for ticker, weight in base_allocation.items()
    }
    
    # Calculate remaining to cash
    total_scaled = sum(scaled_allocation.values())
    cash_allocation = 1.0 - total_scaled
    
    print(f"\nScaled Portfolio (Regime: {overlay['regime']}, Exposure: {overlay['exposure']:.1%}):")
    for ticker, weight in scaled_allocation.items():
        print(f"  {ticker:>10}: {weight:>6.1%}")
    print(f"  {'CASH':>10}: {cash_allocation:>6.1%}")
    
    print(f"\nExposure Reduction: {(1.0 - overlay['exposure']):.1%} moved to cash")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("CRYPTO VOLATILITY OVERLAY - DEMONSTRATION")
    print("Phase 1B.2 Implementation")
    print("=" * 70)
    
    try:
        demo_basic_usage()
        demo_different_scenarios()
        demo_single_benchmark()
        demo_custom_windows()
        demo_portfolio_integration()
        
        print("\n" + "=" * 70)
        print("✓ All demonstrations completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
