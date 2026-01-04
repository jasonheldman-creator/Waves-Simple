import logging
from typing import List, Dict

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_TICKERS = ['AAPL', 'GOOGL', 'AMZN']
OPTIONAL_TICKERS = ['MSFT', 'TSLA', 'META']

def fetch_data_for_ticker(ticker: str) -> Dict:
    """Mock function to fetch data for a given ticker."""
    # Here should be the actual logic for fetching ticker data
    logging.info(f"Fetching data for ticker: {ticker}")
    return {"ticker": ticker, "price": 150.0}  # Placeholder data

def handle_tickers(required_tickers: List[str], optional_tickers: List[str]):
    """Handle processing of required and optional tickers.

    Args:
        required_tickers (List[str]): List of required tickers to process.
        optional_tickers (List[str]): List of optional tickers to process.
    """
    failed_required_tickers = []
    processed_tickers = {}

    # Process required tickers
    for ticker in required_tickers:
        try:
            result = fetch_data_for_ticker(ticker)
            processed_tickers[ticker] = result
        except Exception as e:
            logging.error(f"Failed to process required ticker {ticker}: {str(e)}")
            failed_required_tickers.append(ticker)

    # Exit logic for required tickers
    if failed_required_tickers:
        logging.critical(f"Critical: Failed to process all required tickers. Aborting operation. Failed tickers: {failed_required_tickers}")
        raise RuntimeError("Failed to process all required tickers")

    # Process optional tickers
    for ticker in optional_tickers:
        try:
            result = fetch_data_for_ticker(ticker)
            processed_tickers[ticker] = result
        except Exception as e:
            logging.warning(f"Optional ticker {ticker} could not be processed: {str(e)}")

    logging.info(f"Successfully processed tickers: {list(processed_tickers.keys())}")

def main():
    """Main function to control the flow."""
    try:
        handle_tickers(REQUIRED_TICKERS, OPTIONAL_TICKERS)
    except Exception as error:
        logging.error(f"Exiting program due to a critical error: {str(error)}")

if __name__ == "__main__":
    main()