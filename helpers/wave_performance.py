def get_alpaca_client():
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient
    except ImportError:
        return None

# Update all usages of StockHistoricalDataClient as follows:
# Replace direct instantiation with calls to get_alpaca_client() inside function bodies
