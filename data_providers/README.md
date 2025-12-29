# Data Providers Module

Abstracted data provider interfaces for fetching market data to enable full data readiness (28/28 waves operational).

## Overview

This module provides a clean, extensible abstraction for fetching historical price data from multiple data sources. It supports three independent paths for data enablement:

1. **Live Fetch** - Direct from Yahoo Finance (free, no API key required)
2. **Alternate Providers** - Polygon.io, IEX Cloud, Alpha Vantage (requires API keys)
3. **Offline CSV** - Manual upload of prices.csv

## Architecture

```
data_providers/
├── __init__.py           # Module exports
├── base_provider.py      # Abstract base class
├── yahoo_provider.py     # Yahoo Finance implementation
└── polygon_provider.py   # Polygon.io implementation
```

## Usage

### Basic Example

```python
from data_providers import YahooProvider
from datetime import datetime, timedelta

# Create provider
provider = YahooProvider()

# Test connection
if provider.test_connection():
    print("✅ Connection successful")

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

df = provider.get_history('AAPL', start_date, end_date)
print(f"Fetched {len(df)} rows")
```

### Using Polygon.io

```python
from data_providers import PolygonProvider
import os

# Set API key
os.environ['POLYGON_API_KEY'] = 'your_api_key'

# Create provider
provider = PolygonProvider()

# Fetch data (same interface as Yahoo)
df = provider.get_history('AAPL', start_date, end_date)
```

## Data Format

All providers return data in the same canonical format:

```
date,ticker,close
2024-01-01,AAPL,150.23
2024-01-02,AAPL,151.45
...
```

**Columns:**
- `date`: datetime - Trading date
- `ticker`: str - Stock ticker symbol
- `close`: float - Closing price

## Provider Interface

All providers must implement the `BaseProvider` interface:

```python
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class BaseProvider(ABC):
    @abstractmethod
    def get_history(self, ticker: str, start_date: datetime, 
                   end_date: datetime) -> pd.DataFrame:
        """Fetch historical price data."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the provider connection is working."""
        pass
```

## Available Providers

### YahooProvider

**Source:** Yahoo Finance via yfinance library  
**API Key:** Not required  
**Rate Limits:** Moderate (built-in delays)  
**Coverage:** Stocks, ETFs, some crypto

```python
from data_providers import YahooProvider

provider = YahooProvider()
```

### PolygonProvider

**Source:** Polygon.io  
**API Key:** Required (`POLYGON_API_KEY`)  
**Rate Limits:** Depends on plan  
**Coverage:** Stocks, ETFs, crypto, forex

```python
from data_providers import PolygonProvider

provider = PolygonProvider()  # Uses env var
# or
provider = PolygonProvider(api_key='your_key')
```

## Adding New Providers

To add a new data provider:

1. Create a new file (e.g., `iex_provider.py`)
2. Implement the `BaseProvider` interface
3. Add to `__init__.py`

Example:

```python
# iex_provider.py
from .base_provider import BaseProvider

class IEXProvider(BaseProvider):
    def __init__(self, token=None):
        super().__init__("IEX Cloud")
        self.token = token or os.environ.get('IEX_TOKEN')
    
    def get_history(self, ticker, start_date, end_date):
        # Implementation
        pass
    
    def test_connection(self):
        # Implementation
        pass
```

## Error Handling

All providers handle errors gracefully and return `None` or empty DataFrame on failure:

```python
df = provider.get_history('INVALID', start_date, end_date)

if df is None or df.empty:
    print("Failed to fetch data")
else:
    print(f"Success: {len(df)} rows")
```

## Environment Variables

Configure providers via environment variables:

```bash
# Polygon.io
export POLYGON_API_KEY=your_key

# IEX Cloud (future)
export IEX_TOKEN=your_token

# Alpha Vantage (future)
export ALPHAVANTAGE_KEY=your_key
```

## Testing

Run the provider tests:

```bash
python test_data_providers.py
```

This validates:
- Provider interface implementation
- Connection testing
- Data fetch functionality

## Integration

The providers are used by:

- `scripts/enable_full_data.py` - Main data enablement script
- `scripts/analyze_data_readiness.py` - Data analysis tool

## Troubleshooting

### Connection Failures

```python
provider = YahooProvider()

if not provider.test_connection():
    print("❌ Cannot connect to Yahoo Finance")
    print("Check network connectivity")
```

### API Key Issues

```python
provider = PolygonProvider()

if not provider.api_key:
    print("❌ POLYGON_API_KEY not set")
    print("Set: export POLYGON_API_KEY=your_key")
```

### Data Fetch Failures

Common causes:
- Invalid ticker symbol
- No data for date range
- Network issues
- API rate limits

Check logs for specific error messages.

## Best Practices

1. **Test Connection First**
   ```python
   if provider.test_connection():
       df = provider.get_history(ticker, start, end)
   ```

2. **Handle Failures Gracefully**
   ```python
   df = provider.get_history(ticker, start, end)
   if df is None or df.empty:
       # Fallback or log error
       pass
   ```

3. **Use Appropriate Provider**
   - Yahoo: Best for free tier, good coverage
   - Polygon: Best for production, comprehensive data
   - Manual CSV: Best for offline/restricted environments

4. **Respect Rate Limits**
   - Add delays between requests
   - Batch requests when possible
   - Use caching

## License

Part of WAVES Intelligence™ platform.
