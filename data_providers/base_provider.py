"""
Base Provider Interface
Abstract base class for all data providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import pandas as pd


class BaseProvider(ABC):
    """
    Abstract base class for data providers.
    All data providers must implement the get_history method.
    """
    
    def __init__(self, name: str):
        """
        Initialize the provider.
        
        Args:
            name: Name of the provider (e.g., 'Yahoo', 'Polygon')
        """
        self.name = name
    
    @abstractmethod
    def get_history(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with columns ['date', 'ticker', 'close'] or None if failed
            - date: datetime, the trading date
            - ticker: str, the ticker symbol
            - close: float, the closing price
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the provider connection is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"
