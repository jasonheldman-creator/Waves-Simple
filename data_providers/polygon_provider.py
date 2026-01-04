"""
Polygon.io Provider
Implementation of BaseProvider using Polygon.io API.
"""

import os
from datetime import datetime
from typing import Optional
import pandas as pd
from .base_provider import BaseProvider


class PolygonProvider(BaseProvider):
    """
    Polygon.io data provider.
    Requires POLYGON_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Polygon.io")
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        self._base_url = "https://api.polygon.io/v2/aggs/ticker"
    
    def get_history(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data from Polygon.io.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with columns ['date', 'ticker', 'close'] or None if failed
        """
        if not self.api_key:
            print(f"Error: POLYGON_API_KEY not set for {ticker}")
            return None
        
        try:
            import requests
            
            # Format dates for Polygon API (YYYY-MM-DD)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Construct API URL
            url = f"{self._base_url}/{ticker}/range/1/day/{start_str}/{end_str}"
            params = {
                'apiKey': self.api_key,
                'adjusted': 'true',
                'sort': 'asc'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"Error fetching {ticker} from Polygon: HTTP {response.status_code}")
                return None
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                return None
            
            # Convert to canonical format
            records = []
            for result in data['results']:
                records.append({
                    'date': pd.to_datetime(result['t'], unit='ms'),
                    'ticker': ticker,
                    'close': result['c']
                })
            
            df = pd.DataFrame(records)
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker} from Polygon: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test if Polygon.io API is accessible.
        
        Returns:
            True if connection is successful, False otherwise
        """
        if not self.api_key:
            return False
        
        try:
            import requests
            # Test with a simple reference data endpoint
            url = "https://api.polygon.io/v3/reference/tickers"
            params = {'apiKey': self.api_key, 'limit': 1}
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
