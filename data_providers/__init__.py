"""
Data Providers Module
Provides abstracted data provider interfaces for fetching market data.
"""

from .base_provider import BaseProvider
from .yahoo_provider import YahooProvider

try:
    from .polygon_provider import PolygonProvider
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

__all__ = ['BaseProvider', 'YahooProvider']

if POLYGON_AVAILABLE:
    __all__.append('PolygonProvider')
