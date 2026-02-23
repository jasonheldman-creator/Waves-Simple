"""
Helpers package initialization.

This file ensures the helpers directory is treated as a Python package
so imports such as:

from helpers.wave_performance import compute_portfolio_composite_benchmark

work correctly during pytest collection in CI environments.
"""