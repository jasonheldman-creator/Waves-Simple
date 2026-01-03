# Quick Reference: Canonical Price Loading

## Problem
Most waves were missing `prices.csv` files, causing "No usable price data" errors.

## Solution
Created `generate_all_wave_prices.py` - a canonical price loading mechanism that:
- Automatically generates prices.csv for all waves that need it
- Excludes SmartSafe cash waves (they don't need price data)
- Integrates with existing readiness diagnostics

## Quick Start

```bash
# Generate prices for all waves
python generate_all_wave_prices.py --dummy

# Generate for specific waves
python generate_all_wave_prices.py --waves sp500_wave gold_wave

# Skip waves that already have prices
python generate_all_wave_prices.py --skip-existing
```

## Key Files

- `generate_all_wave_prices.py` - Main script for price generation
- `CANONICAL_PRICE_LOADING.md` - Full documentation
- `test_canonical_price_loading.py` - Test suite

## Verification

```bash
# Run tests
python test_canonical_price_loading.py

# Check readiness
python -c "from analytics_pipeline import compute_data_ready_status; \
    status = compute_data_ready_status('sp500_wave'); \
    print(f\"Status: {status['readiness_status']}\")"
```

## SmartSafe Exemption

These waves don't need price data:
- `smartsafe_treasury_cash_wave`
- `smartsafe_tax_free_money_market_wave`

They are automatically marked as "full" readiness without price files.

## Integration

The mechanism integrates with:
- `analytics_pipeline.py` - Price fetching and caching
- `analytics_truth.py` - TruthFrame generation
- `snapshot_ledger.py` - Snapshot generation
- UI components - Wave Lens, Overview, Data Health Panel

See `CANONICAL_PRICE_LOADING.md` for complete documentation.
