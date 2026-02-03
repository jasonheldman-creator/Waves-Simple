# Equity Waves Validation - Quick Reference

## Quick Commands

### Validate Wave Configuration
```bash
# Run comprehensive validation
python validate_equity_waves.py

# Expected output: "✅ ALL VALIDATIONS PASSED"
```

### Run Integration Tests
```bash
# Run all integration tests
pytest test_equity_waves_integration.py -v

# Expected output: "9 passed in X.XXs"
```

### Check Individual Wave
```python
from helpers.wave_registry import get_wave_by_id

wave = get_wave_by_id('us_megacap_core_wave')
print(f"Name: {wave['wave_name']}")
print(f"Active: {wave['active']}")
print(f"Category: {wave['category']}")
print(f"Benchmark: {wave['benchmark_spec']}")
```

## Validation Script Output

### Success
```
================================================================================
EQUITY WAVES VALIDATION
================================================================================

Validating 9 equity waves:
  - clean_transit_infrastructure_wave
  - demas_fund_wave
  - ev_infrastructure_wave
  - future_power_energy_wave
  - infinity_multi_asset_growth_wave
  - next_gen_compute_semis_wave
  - quantum_computing_wave
  - small_to_mid_cap_growth_wave
  - us_megacap_core_wave

1. Validating wave registry entries...
   ✓ All waves found in registry with required fields

2. Validating positions files and weights...
✓ clean_transit_infrastructure_wave: 10 tickers, weight sum = 1.0000
✓ demas_fund_wave: 10 tickers, weight sum = 1.0000
✓ ev_infrastructure_wave: 10 tickers, weight sum = 1.0000
✓ future_power_energy_wave: 10 tickers, weight sum = 1.0000
✓ infinity_multi_asset_growth_wave: 9 tickers, weight sum = 1.0000
✓ next_gen_compute_semis_wave: 10 tickers, weight sum = 1.0000
✓ quantum_computing_wave: 8 tickers, weight sum = 1.0000
✓ small_to_mid_cap_growth_wave: 5 tickers, weight sum = 1.0000
✓ us_megacap_core_wave: 10 tickers, weight sum = 1.0000
   ✓ All positions files valid

3. Collecting tickers from all waves...
   Total unique tickers across all equity waves: 60

================================================================================
VALIDATION SUMMARY
================================================================================

✅ ALL VALIDATIONS PASSED

All 9 equity waves are properly configured:
  - Registry entries: ✓
  - Positions files: ✓
  - Weights sum to 1.0: ✓
  - Total unique tickers: 60
```

### Failure Example
```
❌ ERRORS (2):
  1. next_gen_compute_semis_wave: Weights sum to 0.8400, expected 1.0 (tolerance 0.01)
  2. us_megacap_core_wave: Weights sum to 0.4500, expected 1.0 (tolerance 0.01)

❌ VALIDATION FAILED with 2 error(s)
```

## Wave Configuration Files

### Registry Entry (data/wave_registry.csv)
```csv
wave_id,wave_name,mode_default,benchmark_spec,category,active,...
us_megacap_core_wave,US MegaCap Core Wave,Standard,SPY:1.0000,equity_growth,True,...
```

### Positions File (data/waves/{wave_id}/positions.csv)
```csv
ticker,weight,description,exposure,cash,safe_fraction
AAPL,0.1556,Apple Inc.,0.1556,0.0,0.0
MSFT,0.1556,Microsoft Corp.,0.1556,0.0,0.0
...
```

## Integration Test Coverage

| Test | Purpose | Status |
|------|---------|--------|
| `test_all_equity_waves_in_registry` | Verify all waves in WAVE_ID_REGISTRY | ✅ PASS |
| `test_all_equity_waves_have_weights` | Verify all waves have WAVE_WEIGHTS | ✅ PASS |
| `test_ticker_discovery` | Verify ticker discovery works | ✅ PASS |
| `test_benchmark_definitions` | Verify benchmarks are valid | ✅ PASS |
| `test_wave_registry_csv` | Verify CSV registry entries | ✅ PASS |
| `test_positions_files_exist` | Verify positions files exist | ✅ PASS |
| `test_ticker_collection_complete` | Verify all tickers collected | ✅ PASS |
| `test_get_all_wave_ids_includes_equity_waves` | Verify wave ID discovery | ✅ PASS |
| `test_active_waves_includes_equity_waves` | Verify active wave filtering | ✅ PASS |

## Troubleshooting

### Weight Sum Error
**Problem:** Weights don't sum to 1.0

**Solution:**
1. Open positions.csv file
2. Sum the weight column
3. Rescale each weight by dividing by the sum
4. Round to 4 decimal places
5. Adjust last weight to make total exactly 1.0

**Example Python:**
```python
import pandas as pd

df = pd.read_csv('data/waves/example_wave/positions.csv')
total = df['weight'].sum()
df['weight'] = df['weight'] / total

# Fine-tune last weight to ensure exact 1.0
df.loc[df.index[-1], 'weight'] = 1.0 - df['weight'][:-1].sum()

df.to_csv('data/waves/example_wave/positions.csv', index=False)
```

### Missing Wave Error
**Problem:** Wave not found in registry

**Solution:**
1. Check wave_id spelling (use snake_case)
2. Verify entry exists in data/wave_registry.csv
3. Ensure 'active' column is True
4. Check WAVE_ID_REGISTRY in waves_engine.py

### Ticker Discovery Error
**Problem:** Wave has no tickers

**Solution:**
1. Verify WAVE_WEIGHTS contains wave display_name
2. Check positions.csv exists in data/waves/{wave_id}/
3. Ensure ticker column is populated
4. Verify WAVE_ID_REGISTRY mapping is correct

## CI/CD Integration

### GitHub Actions Workflow
Automatically runs on:
- Push to any branch (when wave files change)
- Pull requests
- Manual workflow dispatch

### Workflow File
`.github/workflows/validate_equity_waves.yml`

### Triggering Paths
- `data/wave_registry.csv`
- `data/waves/*/positions.csv`
- `waves_engine.py`
- `validate_equity_waves.py`

## Development Workflow

### Adding a New Wave
1. Add entry to `data/wave_registry.csv`
2. Create directory `data/waves/{wave_id}/`
3. Create `positions.csv` with proper format
4. Add to WAVE_WEIGHTS in `waves_engine.py`
5. Add to WAVE_ID_REGISTRY in `waves_engine.py`
6. Run `python validate_equity_waves.py`
7. Run `pytest test_equity_waves_integration.py -v`
8. Commit changes

### Modifying Wave Weights
1. Edit `data/waves/{wave_id}/positions.csv`
2. Ensure weights sum to 1.0
3. Run `python validate_equity_waves.py`
4. Commit if validation passes

## Related Documentation

- `EQUITY_WAVES_IMPLEMENTATION.md` - Full implementation details
- `WAVE_REGISTRY_SELF_HEALING.md` - Registry architecture
- `data/wave_registry.csv` - Wave registry data
- `waves_engine.py` - Wave definitions and logic
