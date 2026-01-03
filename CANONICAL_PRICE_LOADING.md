# Canonical Price Loading Mechanism

## Overview

The canonical price loading mechanism ensures that all waves in the WAVES Intelligence™ system have reliable, consistent price data. This mechanism dynamically fetches and caches prices for each wave's tickers, generating `data/waves/<wave_id>/prices.csv` files for all waves that need them.

## Key Features

1. **Automatic Price Generation**: Dynamically fetches and caches prices for all waves
2. **SmartSafe Exemption**: Automatically excludes SmartSafe cash waves (they don't need price data)
3. **Consistent Readiness**: Readiness diagnostics rely on these canonical price files
4. **Graded Status**: Supports full, partial, operational, and unavailable readiness levels

## Architecture

### Price Data Flow

```
generate_all_wave_prices.py
    ↓
analytics_pipeline.generate_prices_csv()
    ↓
analytics_pipeline.fetch_prices()
    ↓
data/waves/<wave_id>/prices.csv
    ↓
analytics_pipeline.compute_data_ready_status()
    ↓
Readiness Diagnostics
```

### File Structure

Each wave (except SmartSafe cash waves) has a dedicated directory:

```
data/waves/<wave_id>/
├── prices.csv              # Daily close prices for all tickers (REQUIRED)
├── benchmark_prices.csv    # Benchmark prices for comparison
├── positions.csv           # Current position snapshot
├── trades.csv              # Trade history
└── nav.csv                 # NAV history
```

## Usage

### Generate Prices for All Waves

```bash
# Generate prices for all waves (live data from yfinance)
python generate_all_wave_prices.py

# Generate with dummy data (for testing)
python generate_all_wave_prices.py --dummy

# Generate for specific waves only
python generate_all_wave_prices.py --waves sp500_wave gold_wave

# Skip waves that already have prices.csv
python generate_all_wave_prices.py --skip-existing

# Custom lookback period (default: 14 days)
python generate_all_wave_prices.py --lookback 30
```

### Programmatic Usage

```python
from generate_all_wave_prices import generate_all_prices

# Generate prices for all waves
summary = generate_all_prices(
    wave_ids=None,              # None = all waves
    lookback_days=14,           # Days of historical data
    use_dummy_data=False,       # Use live data
    skip_existing=False         # Regenerate existing files
)

print(f"Successful: {summary['successful']}")
print(f"Failed: {summary['failed']}")
print(f"SmartSafe exempt: {summary['skipped_smartsafe']}")
```

### Generate Prices for Single Wave

```python
from generate_all_wave_prices import generate_prices_for_wave

result = generate_prices_for_wave(
    wave_id='sp500_wave',
    lookback_days=14,
    use_dummy_data=False
)

if result['success']:
    print(f"Generated: {result['files_generated']}")
else:
    print(f"Errors: {result['errors']}")
```

## Readiness Diagnostics

The canonical price loading mechanism integrates with the readiness diagnostic system.

### Readiness Levels

1. **Full** (🟢): Complete data, all analytics available
   - Coverage ≥ 90%
   - History ≥ 365 days
   - All files present

2. **Partial** (🟡): Sufficient for basic analytics
   - Coverage ≥ 70%
   - History ≥ 7 days
   - prices.csv present

3. **Operational** (🟠): Current state display only
   - Coverage ≥ 50%
   - History ≥ 1 day
   - prices.csv present

4. **Unavailable** (🔴): Cannot display
   - Missing critical data
   - prices.csv missing or empty

### Check Wave Readiness

```python
from analytics_pipeline import compute_data_ready_status

status = compute_data_ready_status('sp500_wave')

print(f"Readiness: {status['readiness_status']}")
print(f"Coverage: {status['coverage_pct']:.1f}%")
print(f"History: {status['history_days']} days")

# Check if ready for analytics
if status['analytics_ready']:
    print("✓ Ready for advanced analytics")
else:
    print("⚠️  Limited to basic metrics")
```

## SmartSafe Cash Wave Exemption

SmartSafe cash waves are pure cash/money market instruments that don't require price data:

- `smartsafe_treasury_cash_wave`
- `smartsafe_tax_free_money_market_wave`

These waves:
- **Do NOT** need `prices.csv` files
- Are **always** marked as "full" readiness
- Have constant NAV = 1.0 and return = 0%
- Skip price generation automatically

### Implementation

```python
from waves_engine import is_smartsafe_cash_wave

if is_smartsafe_cash_wave(wave_id):
    # Skip price generation
    print(f"SmartSafe cash wave: {wave_id}")
    return
```

The exemption is implemented in:
- `generate_all_wave_prices.py` (skips generation)
- `analytics_pipeline.compute_data_ready_status()` (always marks as ready)
- `analytics_pipeline.generate_benchmark_prices_csv()` (skips benchmarks)

## Maintenance

### Daily Price Updates

To keep prices current, run the price generator daily:

```bash
# Update only existing waves (don't regenerate all)
python generate_all_wave_prices.py --skip-existing

# Or use the full analytics pipeline
python analytics_pipeline.py --all-waves
```

### Monitoring

Check the validation report for data health:

```python
import pandas as pd

# Load validation report
report = pd.read_csv('data/waves/validation_report.csv')

# Check for issues
issues = report[report['status'] != 'pass']
print(f"Waves with issues: {len(issues)}")
```

### Troubleshooting

**Problem**: "No usable price data" error

**Solution**: Run the canonical price loader
```bash
python generate_all_wave_prices.py --waves <wave_id>
```

**Problem**: Stale price data

**Solution**: Regenerate prices with fresh data
```bash
python generate_all_wave_prices.py
```

**Problem**: Network errors (DNS, timeouts)

**Solution**: Use dummy data for testing
```bash
python generate_all_wave_prices.py --dummy
```

## Integration Points

### Analytics Pipeline

The canonical price loading mechanism integrates with:

1. **`analytics_pipeline.py`**: Core price fetching and caching
2. **`analytics_truth.py`**: TruthFrame generation relies on cached prices
3. **`snapshot_ledger.py`**: Snapshot generation uses price data
4. **`waves_engine.py`**: Wave definitions and SmartSafe exemptions

### UI Components

Price data is consumed by:
- Wave Lens (current prices and returns)
- Overview Page (portfolio metrics)
- Data Health Panel (readiness indicators)
- Executive Summary (performance metrics)

## Best Practices

1. **Always use the canonical loader** for price generation
2. **Check SmartSafe status** before expecting price files
3. **Use readiness diagnostics** to verify data quality
4. **Run daily updates** to keep data fresh
5. **Monitor validation reports** for data health

## API Reference

### `generate_all_wave_prices.py`

Main functions:

- `generate_all_prices(wave_ids, lookback_days, use_dummy_data, skip_existing)`: Generate prices for multiple waves
- `generate_prices_for_wave(wave_id, lookback_days, use_dummy_data, skip_existing)`: Generate prices for single wave

### `analytics_pipeline.py`

Core functions:

- `generate_prices_csv(wave_id, lookback_days, use_dummy_data)`: Generate prices.csv
- `generate_benchmark_prices_csv(wave_id, lookback_days, use_dummy_data)`: Generate benchmark_prices.csv
- `compute_data_ready_status(wave_id)`: Check wave readiness
- `fetch_prices(tickers, start_date, end_date, use_dummy_data, wave_id, wave_name)`: Fetch price data

### `waves_engine.py`

Utility functions:

- `is_smartsafe_cash_wave(wave_identifier)`: Check if wave is SmartSafe cash
- `get_all_wave_ids()`: Get all registered wave IDs
- `get_display_name_from_wave_id(wave_id)`: Get human-readable name

## Version History

- **v1.0** (2026-01-03): Initial implementation
  - Created `generate_all_wave_prices.py`
  - Integrated with existing analytics pipeline
  - Added SmartSafe exemption support
  - Generated prices.csv for all 26 non-SmartSafe waves
