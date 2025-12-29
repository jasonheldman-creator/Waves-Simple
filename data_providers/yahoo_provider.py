"""
Yahoo Finance Provider
Implementation of BaseProvider using yfinance.
"""

from datetime import datetime
from typing import Optional
import pandas as pd
from .base_provider import BaseProvider


class YahooProvider(BaseProvider):
    """
    Yahoo Finance data provider using yfinance library.
    """
    
    def __init__(self):
        super().__init__("Yahoo Finance")
        self._test_url = "https://query1.finance.yahoo.com"
    
    def get_history(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with columns ['date', 'ticker', 'close'] or None if failed
        """
        try:
            import yfinance as yf
            
            # Fetch data using yfinance
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            # Convert to canonical format
            df = pd.DataFrame({
                'date': hist.index,
                'ticker': ticker,
                'close': hist['Close'].values
            })
            
            # Reset index to make date a column
            df = df.reset_index(drop=True)
            
            # Ensure date is datetime type
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker} from Yahoo Finance: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test if Yahoo Finance is accessible.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            import requests
            response = requests.get(self._test_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
