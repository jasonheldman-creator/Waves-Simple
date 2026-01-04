# Automated Initial Data Seeding

This document describes the automated data seeding system that ensures all waves in the WAVES Intelligence™ system have historical data for analytics.

## Overview

The seeding system generates synthetic placeholder data for waves that don't yet have real market data. This allows all analytics components (Attribution, Performance Deep Dive, Decision Ledger, Board Pack) to function immediately while the system collects real data over time.

## Key Features

1. **Idempotent**: Can be run multiple times without creating duplicates
2. **Transparent**: All synthetic data is marked with `is_synthetic=True`
3. **Automatic**: Detects which waves need seeding based on wave_id registry
4. **Realistic**: Generates market-like returns with appropriate volatility and correlation
5. **Phase 6 Ready**: Compatible with all analytics components

## Seeding Script

### Location
```
/seed_wave_history.py
```

### Usage

Basic usage (seeds 90 days of data):
```bash
python seed_wave_history.py
```

Custom number of days:
```bash
python seed_wave_history.py --days 180
```

Dry run (preview without changes):
```bash
python seed_wave_history.py --dry-run
```

Custom start date:
```bash
python seed_wave_history.py --start-date 2024-01-01 --days 365
```

### Options

- `--days DAYS`: Number of business days to generate (default: 90)
- `--start-date YYYY-MM-DD`: Start date for synthetic data (default: DAYS before today)
- `--dry-run`: Preview what would be done without modifying files
- `--output PATH`: Output CSV file path (default: wave_history.csv)

### What It Does

1. **Loads existing data**: Reads current wave_history.csv
2. **Identifies gaps**: Compares wave_id registry with existing data
3. **Generates synthetic data**: Creates realistic daily returns for missing waves
4. **Marks synthetic**: Sets `is_synthetic=True` for all generated rows
5. **Creates backup**: Backs up existing file before writing
6. **Saves results**: Writes combined data to wave_history.csv

### Synthetic Data Characteristics

The script generates realistic market returns:
- **Daily volatility**: ~1.2% (comparable to equity markets)
- **Daily drift**: ~0.03% (~7.5% annualized)
- **Correlation**: 65% correlation between portfolio and benchmark
- **Business days only**: Monday-Friday only (excludes weekends)
- **Deterministic**: Same seed for same wave_id (reproducible)

## Data Structure

### wave_history.csv Schema

```csv
wave_id,display_name,date,portfolio_return,benchmark_return,is_synthetic
sp500_wave,S&P 500 Wave,2025-08-11,0.0091,0.0066,True
```

Required columns:
- `wave_id`: Canonical wave identifier (snake_case)
- `display_name`: Human-readable wave name
- `date`: Date in YYYY-MM-DD format
- `portfolio_return`: Daily return (decimal, e.g., 0.01 = 1%)
- `benchmark_return`: Daily benchmark return (decimal)
- `is_synthetic`: Boolean flag (True for synthetic data, False for real)

## UI Integration

### Synthetic Data Banner

The system automatically displays informational banners when synthetic data is in use:

#### Banner Locations
- Attribution tab
- Performance Deep Dive
- Decision Ledger
- Board Pack

#### Banner Behavior
- **Single wave**: Shows percentage of synthetic data for that wave
- **Multiple waves**: Shows total count and expandable list
- **No synthetic data**: No banner displayed
- **Non-intrusive**: Informational only, doesn't block functionality

### Detection Functions

The app.py includes helper functions:

```python
# Check if data contains synthetic entries
status = check_synthetic_data_status(wave_name="S&P 500 Wave")
# Returns: {'has_synthetic': bool, 'synthetic_count': int, ...}

# Render banner if synthetic data present
render_synthetic_data_banner(wave_name="S&P 500 Wave")
```

## Validation

### Automated Tests

Run the validation test suite:
```bash
python test_seeding_validation.py
```

Tests verify:
1. ✅ All waves in registry have data
2. ✅ Synthetic data properly marked
3. ✅ Attribution module compatibility
4. ✅ Performance metrics calculation
5. ✅ Synthetic detection functions

### Manual Verification

Check seeding status:
```python
import pandas as pd
df = pd.read_csv('wave_history.csv')

# Count by type
print(df.groupby('is_synthetic').size())

# List synthetic waves
synthetic_waves = df[df['is_synthetic']]['wave_id'].unique()
print(f"Synthetic waves: {list(synthetic_waves)}")
```

## Migration Path

### From Synthetic to Real Data

As real market data becomes available:

1. **New data ingestion**: Real data is added with `is_synthetic=False`
2. **Automatic detection**: UI banners update to reflect real data percentage
3. **Gradual replacement**: Synthetic data can be manually removed or left (marked as synthetic)
4. **No code changes**: All analytics continue to work seamlessly

### Replacing Synthetic Data

To replace synthetic data with real data for a specific wave:

```python
import pandas as pd

# Load wave_history
df = pd.read_csv('wave_history.csv')

# Remove synthetic data for specific wave
wave_id = 'sp500_wave'
df = df[~((df['wave_id'] == wave_id) & (df['is_synthetic'] == True))]

# Add real data (your ingestion process here)
# ...

# Save
df.to_csv('wave_history.csv', index=False)
```

## Best Practices

### When to Run Seeding

- **Initial setup**: First time setting up the system
- **New waves added**: After adding new waves to WAVE_ID_REGISTRY
- **After data cleanup**: If wave_history.csv is reset or modified
- **Not needed**: If all waves already have data (script detects this)

### Production Deployment

1. **Run seeding once**: During initial deployment
2. **Commit seeded data**: Include wave_history.csv with is_synthetic columns
3. **Set up data ingestion**: Configure real data pipeline
4. **Monitor**: Use UI banners to track real vs synthetic data ratio
5. **Gradual replacement**: Let real data accumulate over time

### Troubleshooting

#### All waves show as needing seeding
- Check that wave_history.csv exists
- Verify wave_id column is present
- Ensure wave_ids match WAVE_ID_REGISTRY

#### Seeding creates duplicates
- This shouldn't happen (idempotent design)
- If it does, check for data corruption
- Restore from backup and re-run

#### Synthetic data not detected in UI
- Verify `is_synthetic` column exists in wave_history.csv
- Check that column values are boolean (True/False)
- Ensure app.py has latest synthetic detection functions

## Technical Details

### Idempotency Implementation

The script is idempotent through:
1. Loading existing wave_history.csv
2. Extracting unique wave_ids already present
3. Only seeding wave_ids not found in existing data
4. Returning early if no waves need seeding

### Deterministic Generation

Returns are deterministic by:
- Using `hash(wave_id) % (2**31)` as random seed
- Same wave_id always generates same data
- Reproducible for testing and debugging

### Performance Considerations

- **Fast**: Seeds 19 waves × 90 days = 1,710 rows in ~1 second
- **Memory efficient**: Processes one wave at a time
- **Backup safety**: Automatic backup before writing
- **Scalable**: Can handle hundreds of waves without issue

## Related Files

- `seed_wave_history.py`: Main seeding script
- `test_seeding_validation.py`: Validation test suite
- `waves_engine.py`: Wave registry (WAVE_ID_REGISTRY)
- `wave_history.csv`: Historical data storage
- `app.py`: UI integration and banner rendering

## Support

For issues or questions:
1. Check validation tests: `python test_seeding_validation.py`
2. Run dry-run: `python seed_wave_history.py --dry-run`
3. Verify wave_id registry: Check `waves_engine.py` WAVE_ID_REGISTRY
4. Review logs: Script provides detailed output during execution
