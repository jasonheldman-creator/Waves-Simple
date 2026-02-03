"""
Demonstration of Return Pipeline Functionality

This script demonstrates the canonical data access helper, wave registry,
and return pipeline in action.
"""

import sys
import os
import importlib.util

# Helper function to load module without __init__.py dependencies
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
helpers_dir = os.path.join(os.path.dirname(__file__), "helpers")
canonical_data = load_module("canonical_data", os.path.join(helpers_dir, "canonical_data.py"))
wave_registry = load_module("wave_registry", os.path.join(helpers_dir, "wave_registry.py"))
return_pipeline = load_module("return_pipeline", os.path.join(helpers_dir, "return_pipeline.py"))

print("=" * 70)
print("Return Pipeline Demonstration")
print("=" * 70)

# 1. Demonstrate canonical data access
print("\n1. CANONICAL DATA ACCESS")
print("-" * 70)
prices = canonical_data.get_canonical_price_data(tickers=['SPY', 'QQQ', 'BTC-USD'])
print(f"Retrieved price data: {prices.shape[0]} days × {prices.shape[1]} tickers")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"\nSample prices (last 5 days):")
print(prices.tail(5))

# 2. Demonstrate wave registry
print("\n\n2. CANONICAL WAVE REGISTRY")
print("-" * 70)
registry = wave_registry.get_wave_registry()
print(f"Total waves in registry: {len(registry)}")
print(f"Active waves: {len(registry[registry['active'] == True])}")

# Show sample wave
sample_wave = wave_registry.get_wave_by_id('sp500_wave')
if sample_wave:
    print(f"\nSample Wave (S&P 500 Wave):")
    print(f"  wave_id: {sample_wave['wave_id']}")
    print(f"  wave_name: {sample_wave['wave_name']}")
    print(f"  category: {sample_wave['category']}")
    print(f"  benchmark_recipe: {sample_wave['benchmark_recipe']}")
    print(f"  active: {sample_wave['active']}")

# 3. Demonstrate return pipeline
print("\n\n3. RETURN PIPELINE")
print("-" * 70)

# Test with a few different waves
test_waves = [
    'sp500_wave',
    'gold_wave',
    'income_wave'
]

for wave_id in test_waves:
    wave = wave_registry.get_wave_by_id(wave_id)
    if wave:
        print(f"\nComputing returns for: {wave['wave_name']} ({wave_id})")
        returns = return_pipeline.compute_wave_returns_pipeline(wave_id)
        
        if not returns.empty:
            print(f"  Data points: {len(returns)}")
            print(f"  Columns: {list(returns.columns)}")
            
            # Show non-zero returns summary
            non_zero = returns[returns['wave_return'] != 0]
            if not non_zero.empty:
                print(f"\n  Summary (non-zero returns only, n={len(non_zero)}):")
                print(f"    Wave Return: mean={non_zero['wave_return'].mean():.4f}, std={non_zero['wave_return'].std():.4f}")
                print(f"    Benchmark Return: mean={non_zero['benchmark_return'].mean():.4f}, std={non_zero['benchmark_return'].std():.4f}")
                print(f"    Alpha: mean={non_zero['alpha'].mean():.4f}, std={non_zero['alpha'].std():.4f}")
            else:
                print("    No non-zero returns found")
        else:
            print("  No data available")

# 4. Show required columns are present
print("\n\n4. VERIFICATION OF REQUIRED COLUMNS")
print("-" * 70)
test_returns = return_pipeline.compute_wave_returns_pipeline('sp500_wave')
required_columns = [
    'wave_return',
    'benchmark_return',
    'alpha',
    'overlay_return_vix',
    'overlay_return_custom'
]

for col in required_columns:
    present = col in test_returns.columns
    status = "✓" if present else "✗"
    print(f"  {status} {col}")

print("\n" + "=" * 70)
print("Demonstration Complete!")
print("=" * 70)
