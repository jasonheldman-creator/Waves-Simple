# Benchmark Diagnostics & Proof Fields - Implementation Guide

## Overview

This implementation adds auditable proof fields to wave computations to validate the accuracy and explainability of 365D alpha metrics. These fields provide transparency into benchmark composition, data quality, and calculation integrity.

## New Fields Added

All new fields are available in `result.attrs["coverage"]` from `compute_history_nav()` calls.

### 1. Benchmark Mode & Components

#### `benchmark_mode` (string)
- **Values**: `"DYNAMIC"` or `"STATIC"`
- **Description**: Indicates whether the wave uses a dynamic (multi-component) or static benchmark
- **Example**: `"DYNAMIC"`

#### `benchmark_components_preview` (string)
- **Description**: Human-readable preview of benchmark components showing top 5 tickers with weights
- **Format**: `"TICKER1:XX.X%, TICKER2:YY.Y% +N more"`
- **Examples**:
  - `"SPY:100.0%"` (single component)
  - `"QQQ:60.0%, SMH:25.0%, IGV:15.0%"` (multi-component)
  - `"SPY:40.0%, QQQ:40.0%, BTC-USD:20.0%"` (multi-asset)

#### `benchmark_hash` (string)
- **Description**: Stable SHA256-based hash of benchmark tickers and weights (first 16 hex chars)
- **Purpose**: Provides auditable proof - hash changes only when components or weights change
- **Example**: `"a1b2c3d4e5f6g7h8"`
- **Properties**:
  - Deterministic (same components → same hash)
  - Order-independent (sorted internally)
  - Sensitive to weight changes (even small changes alter hash)

### 2. 365D Window Integrity Fields

#### `window_365d_integrity` (dict)

Complete diagnostic information about the 365D window used for alpha calculations:

```python
{
    'wave_365d_days': 252,           # Actual days available for wave
    'bench_365d_days': 252,          # Actual days available for benchmark
    'intersection_days_used': 252,   # Days with data for both
    'wave_365d_start': '2023-01-15', # Wave data start date (YYYY-MM-DD)
    'wave_365d_end': '2024-01-15',   # Wave data end date
    'bench_365d_start': '2023-01-15',# Benchmark data start date
    'bench_365d_end': '2024-01-15',  # Benchmark data end date
    'last_date_wave': '2024-01-15',  # Latest date with wave data
    'last_date_bench': '2024-01-15', # Latest date with benchmark data
    'sufficient_history': True,      # Boolean: >= 200 days?
    'min_required_days': 200,        # Minimum threshold
    'warning_message': None          # Warning if history insufficient
}
```

**Warning Messages:**
- `None` - Sufficient history (≥252 days)
- `"PARTIAL HISTORY: XXX days of overlap (less than full 252 trading days)"` - 200-251 days
- `"LIMITED HISTORY: Only XXX days of overlap (minimum 200 recommended)"` - <200 days
- `"Empty return series"` - No data available

### 3. Alpha Reconciliation Fields

#### `alpha_365d_reconciliation` (dict)

Validation that alpha_365d matches (wave_365d_return - bench_365d_return):

```python
{
    'reconciliation_passed': True,    # Boolean: passed tolerance check?
    'expected_alpha': 0.0523,         # wave_365d_return - bench_365d_return
    'computed_alpha': 0.0523,         # Actual alpha value
    'mismatch': 0.0000,               # Absolute difference
    'mismatch_bps': 0.02,             # Mismatch in basis points
    'tolerance': 0.001,               # Tolerance threshold (10 bps)
    'warning_message': None           # Warning if reconciliation failed
}
```

**Tolerance**: Default 0.001 (0.1% = 10 basis points)

**Warning Messages:**
- `None` - Reconciliation passed
- `"RECONCILIATION FAILED: Alpha mismatch of XX.X bps (expected Y.YYYY, got Z.ZZZZ)"` - Failed
- `"Insufficient history: XXX days available (need 252)"` - Not enough data
- `"Missing data for reconciliation"` - None values
- `"NaN values in reconciliation inputs"` - NaN values

#### `wave_365d_return` (float)
- **Description**: Cumulative 365-day wave return (last 252 trading days)
- **Example**: `0.1523` (15.23%)

#### `bench_365d_return` (float)
- **Description**: Cumulative 365-day benchmark return (last 252 trading days)
- **Example**: `0.0998` (9.98%)

#### `alpha_365d` (float)
- **Description**: Computed 365-day alpha (wave - benchmark)
- **Example**: `0.0525` (5.25%)

## Usage Examples

### Example 1: Accessing Benchmark Diagnostics

```python
from waves_engine import compute_history_nav
from helpers.price_book import get_price_book

# Compute wave history
price_book = get_price_book()
result = compute_history_nav(
    wave_name="AI & Cloud MegaCap Wave",
    mode="Standard",
    days=365,
    price_df=price_book
)

# Access diagnostics
coverage = result.attrs['coverage']

# Benchmark mode
print(f"Benchmark Mode: {coverage['benchmark_mode']}")
# Output: "Benchmark Mode: DYNAMIC"

# Components
print(f"Components: {coverage['benchmark_components_preview']}")
# Output: "Components: QQQ:60.0%, SMH:25.0%, IGV:15.0%"

# Hash
print(f"Hash: {coverage['benchmark_hash']}")
# Output: "Hash: 7a8f9b2c4d1e5g3h"
```

### Example 2: Checking Window Integrity

```python
integrity = coverage['window_365d_integrity']

if not integrity['sufficient_history']:
    print(f"⚠️  WARNING: {integrity['warning_message']}")
    print(f"Only {integrity['intersection_days_used']} days of overlap")
else:
    print(f"✓ Sufficient history: {integrity['intersection_days_used']} days")
    print(f"Date range: {integrity['wave_365d_start']} to {integrity['wave_365d_end']}")
```

### Example 3: Validating Alpha Reconciliation

```python
recon = coverage['alpha_365d_reconciliation']

if recon['reconciliation_passed']:
    print(f"✓ Alpha reconciliation passed")
    print(f"Alpha: {coverage['alpha_365d']*100:.2f}%")
    print(f"Mismatch: {recon['mismatch_bps']:.2f} bps")
else:
    print(f"✗ Reconciliation failed!")
    print(f"Expected: {recon['expected_alpha']*100:.4f}%")
    print(f"Computed: {recon['computed_alpha']*100:.4f}%")
    print(f"Mismatch: {recon['mismatch_bps']:.1f} bps")
```

## UI Integration Guidelines

### Displaying in Wave Detail Panels

Recommended layout for wave detail panel:

```
Wave Details: [Wave Name]
─────────────────────────────────
Benchmark Information:
  Mode: DYNAMIC
  Components: QQQ:60.0%, SMH:25.0%, IGV:15.0%
  Hash: 7a8f9b2c4d1e5g3h

365D Alpha Metrics:
  Wave Return: +15.23%
  Benchmark Return: +9.98%
  Alpha: +5.25%
  
Window Integrity:
  Days Used: 252 / 252 ✓
  Date Range: 2023-01-15 to 2024-01-15
  Status: Sufficient History
```

### Warning Banners

#### LIMITED HISTORY Warning

When `intersection_days_used < 200`:

```
⚠️  LIMITED HISTORY WARNING
Only XXX days of overlapping data available (minimum 200 recommended).
365D alpha metrics should be treated as preliminary.
```

Display alpha as: `N/A` or `~XX.X% (limited data)`

#### Reconciliation Failure Warning

When `reconciliation_passed == False`:

```
🔴 RECONCILIATION FAILED
Alpha calculation mismatch detected: XX.X basis points
Expected: YY.YY%, Computed: ZZ.ZZ%
This may indicate a data quality or calculation issue.
```

## Testing

### Unit Tests

Run comprehensive test suite:

```bash
pytest test_benchmark_diagnostics.py -v
```

**Test Coverage** (23 tests):
- Benchmark hash: determinism, order-independence, change detection
- Components preview: formatting, top N display, remaining count
- Window integrity: full/partial overlap, limited history, date ranges
- Alpha reconciliation: perfect match, tolerance, mismatches, edge cases
- Integration: realistic scenarios with controlled data

### Validation Script

Run validation script to see fields in action:

```bash
python validate_benchmark_diagnostics.py
```

This will:
1. Load price data
2. Compute history for test waves
3. Display all diagnostic fields
4. Show warnings and validation results

## Implementation Notes

### Helper Functions Added

1. `_compute_benchmark_hash(components)` - Stable hash computation
2. `_format_benchmark_components_preview(components, max_display=5)` - Format preview
3. `_compute_365d_window_integrity(wave_ret, bm_ret, trading_days_365d)` - Window metrics
4. `_compute_alpha_reconciliation(wave_ret, bench_ret, alpha, tolerance)` - Validation

### Location in Code

- **Functions**: `waves_engine.py` (after `build_portfolio_composite_benchmark_returns`)
- **Integration**: `waves_engine.py` in `_compute_core()` function (after line 4350)
- **Tests**: `test_benchmark_diagnostics.py`
- **Validation**: `validate_benchmark_diagnostics.py`

### Backward Compatibility

All new fields are additive - existing code continues to work without modification. The fields are available in `result.attrs['coverage']` for consumers that need them.

### Performance Impact

Minimal - all computations use existing data:
- Hash computation: O(n log n) where n = number of components
- Window integrity: O(1) - simple counting and date extraction
- Reconciliation: O(1) - simple arithmetic

## Future Enhancements

Potential improvements for future iterations:

1. **Multi-period reconciliation**: Extend reconciliation to 30D, 60D periods
2. **Benchmark drift tracking**: Monitor how benchmark composition changes over time
3. **Historical hash archive**: Store benchmark hashes for audit trail
4. **Automated alerting**: Notify when reconciliation fails or history is limited
5. **Visualization**: Charts showing window integrity and data overlap

## References

- Problem statement: GitHub issue (original requirement)
- Dynamic Benchmarks: `DYNAMIC_BENCHMARKS.md`
- Price Book: `helpers/price_book.py`
- Wave Engine: `waves_engine.py`
