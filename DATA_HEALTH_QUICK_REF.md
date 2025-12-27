# Data Health Pipeline - Quick Reference

## Quick Start

### View System Health
1. Open sidebar → "📊 Data Health Status"
2. Check "Failed Ticker Diagnostics" section
3. View failure counts and types

### Export Failed Ticker Report
```
Sidebar → Data Health Status → Export Failed Tickers Report → Download Report
```

### Force Rebuild All Data
```
Sidebar → 🔨 Force Build Data for All Waves
```

## Common Failure Types & Fixes

| Failure Type | Meaning | Suggested Fix |
|--------------|---------|---------------|
| `RATE_LIMIT` | Too many requests to API | Automatically retried with backoff |
| `SYMBOL_NEEDS_NORMALIZATION` | Ticker format issue (e.g., BRK.B) | Automatically normalized to BRK-B |
| `PROVIDER_EMPTY` | No data from provider | Verify ticker is valid/not delisted |
| `NETWORK_TIMEOUT` | Connection timeout | Automatically retried up to 3 times |
| `INSUFFICIENT_HISTORY` | New ticker, limited history | Reduce lookback period |
| `SYMBOL_INVALID` | Invalid ticker symbol | Remove from configuration |

## Key Files

| File | Purpose |
|------|---------|
| `helpers/ticker_diagnostics.py` | Core diagnostics module |
| `analytics_pipeline.py` | Enhanced data fetching with retry logic |
| `helpers/data_health_panel.py` | UI for viewing diagnostics |
| `./reports/failed_tickers_report_*.csv` | Exported failure reports |

## Code Examples

### Python: Get Diagnostics
```python
from helpers.ticker_diagnostics import get_diagnostics_tracker

tracker = get_diagnostics_tracker()
stats = tracker.get_summary_stats()
print(f"Total failures: {stats['total_failures']}")

# Export to CSV
csv_path = tracker.export_to_csv()
```

### Python: Fetch with Diagnostics
```python
from analytics_pipeline import fetch_prices
from datetime import datetime, timedelta

prices_df, failures = fetch_prices(
    tickers=["AAPL", "MSFT"],
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    wave_id="sp500_wave",
    wave_name="S&P 500 Wave"
)
```

## Testing

```bash
# Unit tests
python test_ticker_diagnostics.py

# Integration tests
python test_analytics_integration.py
```

## Troubleshooting

### Many rate limit errors?
→ Already handled automatically with exponential backoff (1s, 2s, 4s)

### BRK.B or BF.B failing?
→ Automatically normalized to BRK-B, BF-B format

### Need detailed failure info?
→ Sidebar → Data Health Status → View "Recent Failures" section

### Too many old failures?
→ Click "Force Build Data for All Waves" to clear and rebuild

## Performance Notes

- **Batch delay**: 0.5s between individual ticker fetches
- **Retry attempts**: Up to 3 retries per ticker
- **Backoff timing**: 1s, 2s, 4s for retries
- **Diagnostics overhead**: ~1ms per failure (negligible)

## Report Format

CSV columns:
- `ticker_original` - Original ticker symbol
- `ticker_normalized` - Normalized symbol
- `wave_id` - Wave identifier
- `wave_name` - Wave display name
- `source` - Data source (yfinance)
- `failure_type` - Categorized error type
- `error_message` - Error details
- `first_seen` - First occurrence timestamp
- `last_seen` - Last occurrence timestamp
- `is_fatal` - Whether error is fatal
- `suggested_fix` - Remediation guidance

## Admin Checklist

Daily:
- [ ] Check Data Health Status in sidebar
- [ ] Review any new failures
- [ ] Export report if failures > 10

Weekly:
- [ ] Force rebuild all data
- [ ] Archive old reports
- [ ] Review failure trends

Monthly:
- [ ] Update ticker configurations based on persistent failures
- [ ] Review and optimize fetch settings
