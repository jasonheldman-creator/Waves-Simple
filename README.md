# WAVES Institutional Console

A robust Streamlit application for portfolio analytics and performance monitoring with intelligent data fallback mechanisms.

## Overview

The WAVES Institutional Console provides comprehensive portfolio analytics for multiple investment waves, including:

- Real-time performance metrics
- Holdings analysis with weight distribution
- Benchmark comparison (SPY, QQQ, IWM)
- Visual performance charts
- Today's price change tracking
- Google Finance integration

## Features

### 1. Intelligent Data Fallback
The application uses a three-tier fallback system for price data:
1. **Yahoo Finance API** (primary) - Live market data
2. **Local prices.csv** (secondary) - Cached historical data
3. **Synthetic Data** (fallback) - Generated for demonstration when live data unavailable

### 2. Robust Wave Configuration
- Supports multiple investment waves (AI, Growth, Income, etc.)
- Automatic weight normalization per wave
- Handles duplicate tickers by aggregating weights
- Case-insensitive column matching

### 3. Portfolio Analytics
- Daily return calculation
- Cumulative NAV tracking
- Benchmark-relative performance
- Exposure and SmartSafe allocation metrics

### 4. User-Friendly Interface
- Sidebar navigation for wave and mode selection
- Live/Demo mode toggle
- Performance metrics dashboard
- Interactive holdings table with Google Finance links
- Visual performance charts

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jasonheldman-creator/Waves-Simple.git
cd Waves-Simple
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure `wave_weights.csv` exists in the root directory with the following format:
```csv
Wave,Ticker,Weight
AI_Wave,NVDA,0.12
AI_Wave,MSFT,0.11
...
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Navigation

1. **Select Wave**: Choose from available investment waves in the sidebar
2. **Select Mode**: 
   - **Live**: Uses real-time data with fallback mechanisms
   - **Demo**: Uses synthetic data for demonstration
3. **View Analytics**: Main dashboard displays:
   - Performance metrics (Wave Return, Benchmark Return, Exposure, SmartSafe)
   - Performance chart comparing wave vs benchmark
   - Top holdings table with weights and daily changes

## Architecture

### Key Components

#### 1. `load_weights()`
- Loads and validates `wave_weights.csv`
- Case-insensitive column matching
- Automatic weight normalization per wave
- Removes invalid entries
- Cached with `@st.cache_data` (5-minute TTL)

#### 2. `fetch_price_history()`
- Three-tier fallback system:
  1. Yahoo Finance API
  2. Local `prices.csv`
  3. Synthetic data generation
- Handles network failures gracefully
- Returns consistent DataFrame format

#### 3. `compute_portfolio_engine()`
- Aligns price series across all holdings
- Computes weighted portfolio returns
- Calculates cumulative NAV
- Tracks today's price changes
- Generates benchmark comparison

#### 4. Streamlit UI Components
- `render_sidebar()`: Wave and mode selection
- `render_performance_metrics()`: Metrics dashboard
- `render_holdings_table()`: Holdings with Google Finance links
- `render_performance_chart()`: Visual performance comparison

### Data Flow

```
wave_weights.csv → load_weights() → get_wave_holdings()
                                          ↓
Yahoo Finance ─┐                          ↓
prices.csv ────┼→ fetch_price_history() → compute_portfolio_engine()
Synthetic data ─┘                         ↓
                                    WaveEngineResult
                                          ↓
                                    Streamlit UI
```

## Configuration

### Benchmark Mapping

Edit `WAVE_BENCHMARKS` in `app.py` to customize benchmark tickers:

```python
WAVE_BENCHMARKS = {
    "AI_Wave": "QQQ",
    "Growth_Wave": "SPY",
    "Income_Wave": "SPY",
    # Add more mappings...
}
```

### Cache TTL

Adjust cache duration in decorators:

```python
@st.cache_data(ttl=300)  # 5 minutes
def load_weights(...):
    ...
```

## File Structure

```
Waves-Simple/
├── app.py                  # Main Streamlit application
├── waves_engine.py         # Backend portfolio engine (optional)
├── load_universe.py        # Universe loader utility
├── wave_weights.csv        # Portfolio weights configuration
├── requirements.txt        # Python dependencies
├── logs/
│   ├── positions/         # Position snapshots
│   └── performance/       # Performance history
└── README.md              # This file
```

## Error Handling

The application is designed to handle various failure scenarios gracefully:

- **Missing wave_weights.csv**: Displays error message, empty wave list
- **Invalid CSV format**: Shows specific error about missing columns
- **Network failures**: Falls back to local/synthetic data automatically
- **Missing tickers**: Continues with available data
- **Empty holdings**: Shows warning, displays empty metrics

## Design Changes (December 2024)

### Improvements Made

1. **Simplified `load_weights()` function**:
   - Removed DEBUG output statements
   - Improved error messages with emojis
   - Fixed column normalization logic
   - Added proper DataFrame copying to avoid SettingWithCopyWarning

2. **Enhanced fallback mechanism**:
   - Yahoo Finance → prices.csv → synthetic data
   - Each fallback is tried silently
   - No crashes on API restrictions

3. **Robust portfolio engine**:
   - Date alignment across all price series
   - Handles missing tickers gracefully
   - Computes weighted returns correctly
   - Generates valid WaveEngineResult even with partial data

4. **Improved Streamlit layout**:
   - Fixed AttributeError issues with holdings table
   - Proper key usage for all widgets
   - Markdown-based table rendering for compatibility
   - Google Finance links as markdown links

5. **Added @st.cache_data decorators**:
   - 5-minute TTL for load_weights()
   - 5-minute TTL for fetch_yahoo_price_history()
   - Reduces API calls and improves performance

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **yfinance**: Yahoo Finance API client
- **requests**: HTTP library

## Troubleshooting

### Yahoo Finance API Issues

If you see warnings about Yahoo Finance:
- The app automatically falls back to synthetic data
- No action needed, the app will continue to function
- Consider adding a `prices.csv` file for offline operation

### Port Already in Use

If port 8501 is busy:
```bash
streamlit run app.py --server.port 8502
```

### Cache Issues

Clear Streamlit cache:
```bash
streamlit cache clear
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add license information]

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: [Add contact information]
