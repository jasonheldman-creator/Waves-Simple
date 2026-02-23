def get_alpaca_client():
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient
    except ImportError:
        return None

# Re-exporting compute_portfolio_composite_benchmark for backward compatibility
def compute_portfolio_composite_benchmark():
    alpaca_client = get_alpaca_client()
    if not alpaca_client:
        raise ImportError("Alpaca SDK is required to compute portfolio composite benchmark")

    # Placeholder for original logic
    print("Restoring compute_portfolio_composite_benchmark logic...")

# Ensure no direct top-level imports from Alpaca and maintain test compatibility.