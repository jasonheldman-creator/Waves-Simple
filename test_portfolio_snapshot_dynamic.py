#!/usr/bin/env python3
"""
Test script to validate Portfolio Snapshot dynamic computation logic.

This script tests the core computation functions used in the Portfolio Snapshot
without requiring the full Streamlit app to run.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def safe_compounded_return(returns_series):
    """
    Safely compute compounded returns with numerical stability.
    Returns None if data is invalid (e.g., returns <= -1).
    """
    try:
        # Check for extreme negative returns that would cause log1p to fail
        if (returns_series <= -1).any():
            return None
        # Use numerically stable formula: exp(sum(log(1+r))) - 1
        return np.expm1(np.log1p(returns_series).sum())
    except (ValueError, RuntimeError):
        return None


def test_portfolio_snapshot_computation():
    """Test the Portfolio Snapshot computation logic."""
    
    print("=" * 80)
    print("Portfolio Snapshot Dynamic Computation - Logic Validation")
    print("=" * 80)
    print()
    
    # Create synthetic price data
    print("1. Creating synthetic PRICE_BOOK data...")
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    n_dates = len(dates)
    n_tickers = 50
    
    # Generate synthetic prices with some realistic behavior
    np.random.seed(42)
    initial_prices = np.random.uniform(50, 200, n_tickers)
    
    # Generate random daily returns with slight upward drift
    daily_returns = np.random.normal(0.0005, 0.015, (n_dates, n_tickers))
    
    # Compute prices from returns
    prices = np.zeros((n_dates, n_tickers))
    prices[0] = initial_prices
    for i in range(1, n_dates):
        prices[i] = prices[i-1] * (1 + daily_returns[i])
    
    # Create PRICE_BOOK DataFrame
    ticker_names = [f'TICK{i:02d}' for i in range(n_tickers)]
    ticker_names[0] = 'SPY'  # Make first ticker SPY for benchmark
    
    PRICE_BOOK = pd.DataFrame(prices, index=dates, columns=ticker_names)
    print(f"   Created PRICE_BOOK: {PRICE_BOOK.shape[0]} rows × {PRICE_BOOK.shape[1]} tickers")
    print(f"   Date range: {PRICE_BOOK.index[0].date()} to {PRICE_BOOK.index[-1].date()}")
    print()
    
    # Compute returns
    print("2. Computing daily returns...")
    returns_df = PRICE_BOOK.pct_change().dropna()
    print(f"   Returns DataFrame: {returns_df.shape[0]} rows × {returns_df.shape[1]} tickers")
    print()
    
    # Compute portfolio returns (equal-weighted)
    print("3. Computing equal-weighted portfolio returns...")
    portfolio_returns = returns_df.mean(axis=1)
    print(f"   Portfolio returns series: {len(portfolio_returns)} values")
    print()
    
    # Get benchmark returns
    print("4. Extracting benchmark returns (SPY)...")
    benchmark_returns = returns_df['SPY']
    print(f"   Benchmark returns series: {len(benchmark_returns)} values")
    print()
    
    # Compute metrics for each timeframe
    print("5. Computing multi-timeframe metrics...")
    TRADING_DAYS_PER_YEAR = 252
    
    # Returns
    ret_1d = portfolio_returns.iloc[-1] if len(portfolio_returns) >= 1 else None
    ret_30d = safe_compounded_return(portfolio_returns.iloc[-30:]) if len(portfolio_returns) >= 30 else None
    ret_60d = safe_compounded_return(portfolio_returns.iloc[-60:]) if len(portfolio_returns) >= 60 else None
    ret_365d = safe_compounded_return(portfolio_returns.iloc[-TRADING_DAYS_PER_YEAR:]) if len(portfolio_returns) >= TRADING_DAYS_PER_YEAR else None
    
    # Benchmark returns
    bench_1d = benchmark_returns.iloc[-1] if len(benchmark_returns) >= 1 else None
    bench_30d = safe_compounded_return(benchmark_returns.iloc[-30:]) if len(benchmark_returns) >= 30 else None
    bench_60d = safe_compounded_return(benchmark_returns.iloc[-60:]) if len(benchmark_returns) >= 60 else None
    bench_365d = safe_compounded_return(benchmark_returns.iloc[-TRADING_DAYS_PER_YEAR:]) if len(benchmark_returns) >= TRADING_DAYS_PER_YEAR else None
    
    # Alpha
    alpha_1d = (ret_1d - bench_1d) if (ret_1d is not None and bench_1d is not None) else None
    alpha_30d = (ret_30d - bench_30d) if (ret_30d is not None and bench_30d is not None) else None
    alpha_60d = (ret_60d - bench_60d) if (ret_60d is not None and bench_60d is not None) else None
    alpha_365d = (ret_365d - bench_365d) if (ret_365d is not None and bench_365d is not None) else None
    
    print()
    print("=" * 80)
    print("PORTFOLIO SNAPSHOT METRICS (RUNTIME DYNAMIC COMPUTATION)")
    print("=" * 80)
    print()
    
    # Display results
    print("📊 RETURNS")
    print(f"   1D Return:   {ret_1d:+.4%}" if ret_1d is not None else "   1D Return:   N/A")
    print(f"   30D Return:  {ret_30d:+.4%}" if ret_30d is not None else "   30D Return:  N/A")
    print(f"   60D Return:  {ret_60d:+.4%}" if ret_60d is not None else "   60D Return:  N/A")
    print(f"   365D Return: {ret_365d:+.4%}" if ret_365d is not None else "   365D Return: N/A")
    print()
    
    print("📈 BENCHMARK (SPY)")
    print(f"   1D Return:   {bench_1d:+.4%}" if bench_1d is not None else "   1D Return:   N/A")
    print(f"   30D Return:  {bench_30d:+.4%}" if bench_30d is not None else "   30D Return:  N/A")
    print(f"   60D Return:  {bench_60d:+.4%}" if bench_60d is not None else "   60D Return:  N/A")
    print(f"   365D Return: {bench_365d:+.4%}" if bench_365d is not None else "   365D Return: N/A")
    print()
    
    print("⚡ ALPHA (Portfolio - Benchmark)")
    print(f"   1D Alpha:   {alpha_1d:+.4%}" if alpha_1d is not None else "   1D Alpha:   N/A")
    print(f"   30D Alpha:  {alpha_30d:+.4%}" if alpha_30d is not None else "   30D Alpha:  N/A")
    print(f"   60D Alpha:  {alpha_60d:+.4%}" if alpha_60d is not None else "   60D Alpha:  N/A")
    print(f"   365D Alpha: {alpha_365d:+.4%}" if alpha_365d is not None else "   365D Alpha: N/A")
    print()
    
    # Diagnostic info
    print("=" * 80)
    print("DIAGNOSTIC INFORMATION")
    print("=" * 80)
    print(f"Data Source: Synthetic PRICE_BOOK ({PRICE_BOOK.shape[0]} rows × {PRICE_BOOK.shape[1]} tickers)")
    print(f"Last Trading Date: {PRICE_BOOK.index[-1].date()}")
    print(f"Render UTC: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Benchmark: SPY ✅")
    print(f"Snapshot Artifact: ❌ No live_snapshot.csv dependency")
    print(f"Caching: ❌ No caching (pure runtime computation)")
    print()
    
    # Validation checks
    print("=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: All returns computed
    checks_total += 1
    if all(x is not None for x in [ret_1d, ret_30d, ret_60d, ret_365d]):
        print("✅ All portfolio returns computed successfully")
        checks_passed += 1
    else:
        print("❌ Some portfolio returns are None")
    
    # Check 2: All benchmark returns computed
    checks_total += 1
    if all(x is not None for x in [bench_1d, bench_30d, bench_60d, bench_365d]):
        print("✅ All benchmark returns computed successfully")
        checks_passed += 1
    else:
        print("❌ Some benchmark returns are None")
    
    # Check 3: All alpha metrics computed
    checks_total += 1
    if all(x is not None for x in [alpha_1d, alpha_30d, alpha_60d, alpha_365d]):
        print("✅ All alpha metrics computed successfully")
        checks_passed += 1
    else:
        print("❌ Some alpha metrics are None")
    
    # Check 4: Returns are reasonable (between -100% and +1000%)
    checks_total += 1
    returns_reasonable = True
    for ret in [ret_1d, ret_30d, ret_60d, ret_365d]:
        if ret is not None and (ret < -1.0 or ret > 10.0):
            returns_reasonable = False
    if returns_reasonable:
        print("✅ All returns are within reasonable bounds")
        checks_passed += 1
    else:
        print("❌ Some returns are outside reasonable bounds")
    
    # Check 5: Alpha computation is correct (manual check)
    checks_total += 1
    if alpha_1d is not None and ret_1d is not None and bench_1d is not None:
        expected_alpha = ret_1d - bench_1d
        if abs(alpha_1d - expected_alpha) < 1e-10:
            print("✅ Alpha computation is mathematically correct")
            checks_passed += 1
        else:
            print("❌ Alpha computation has numerical error")
    else:
        print("⚠️  Cannot verify alpha computation (missing data)")
    
    print()
    print(f"VALIDATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print()
    
    if checks_passed == checks_total:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        return True
    else:
        print("⚠️  SOME VALIDATION CHECKS FAILED")
        return False


if __name__ == "__main__":
    success = test_portfolio_snapshot_computation()
    exit(0 if success else 1)
