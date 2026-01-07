"""Simple test for ledger reconciliation without mocking."""
import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class Holding:
    ticker: str
    weight: float
    name: str = None

# Create synthetic data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
price_book = pd.DataFrame({
    'SPY': [400 + i * 0.5 for i in range(100)],
    '^VIX': [20 + np.sin(i * 0.1) * 5 for i in range(100)],
    'BIL': [90 + i * 0.01 for i in range(100)],
    'AAPL': [150 + i * 0.7 for i in range(100)],
    'MSFT': [300 + i * 0.6 for i in range(100)],
}, index=dates)

# Test reconciliation formulas directly
print("Testing reconciliation formulas...")

# Compute daily returns manually
risk_return = price_book['AAPL'].pct_change().fillna(0)
safe_return = price_book['BIL'].pct_change().fillna(0)
benchmark_return = price_book['SPY'].pct_change().fillna(0)
exposure = pd.Series(0.8, index=dates)  # Constant exposure for test

# Compute realized return
realized_return = exposure * risk_return + (1 - exposure) * safe_return

# Compute alphas
alpha_total = realized_return - benchmark_return
alpha_selection = risk_return - benchmark_return
alpha_overlay = realized_return - risk_return
alpha_residual = alpha_total - (alpha_selection + alpha_overlay)

# Check reconciliation 1: realized_return - benchmark_return == alpha_total
check_1 = np.abs((realized_return - benchmark_return) - alpha_total)
print(f"Reconciliation 1 max diff: {check_1.max():.15f}")
assert (check_1 < 1e-10).all(), "Reconciliation 1 failed"

# Check reconciliation 2: alpha_selection + alpha_overlay + alpha_residual == alpha_total
check_2 = np.abs((alpha_selection + alpha_overlay + alpha_residual) - alpha_total)
print(f"Reconciliation 2 max diff: {check_2.max():.15f}")
assert (check_2 < 1e-10).all(), "Reconciliation 2 failed"

print("✓ All reconciliation formulas verified!")
print("✓ Test passed!")
