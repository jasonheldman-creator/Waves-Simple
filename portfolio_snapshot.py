import pandas as pd
import numpy as np

def calculate_portfolio_returns(price_book):
    # Ensure price_book is valid
    if price_book is None or price_book.empty:
        raise ValueError("PRICE_BOOK is empty. Cannot compute portfolio snapshot.")

    # Extract trading dates and symbols
    trading_dates = price_book.index
    symbols = price_book.columns

    # Calculate returns over specified windows
    lookbacks = {
        '1D': 1,
        '30D': 30,
        '60D': 60,
        '365D': 365
    }

    portfolio_returns = {}

    for label, days in lookbacks.items():
        if len(trading_dates) > days:
            recent_prices = price_book.iloc[-(days + 1):]
            daily_returns = recent_prices.pct_change().iloc[1:, :]
            mean_returns = daily_returns.mean(axis=1)
            portfolio_returns[label] = mean_returns.mean()

    # Compute alpha (if benchmark exists)
    if 'BENCHMARK' in price_book.columns:
        benchmark_returns = price_book['BENCHMARK'].pct_change().iloc[1:]
        portfolio_alpha = {
            label: portfolio_returns[label] - benchmark_returns.mean()
            for label in portfolio_returns.keys()
        }
        portfolio_returns['ALPHA'] = portfolio_alpha

    return portfolio_returns
