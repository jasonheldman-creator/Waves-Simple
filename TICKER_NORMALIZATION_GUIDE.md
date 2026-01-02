# Ticker Normalization and Wave Coverage Implementation

## Overview

This implementation adds canonical ticker normalization and intelligent wave coverage validation to the `build_wave_history_from_prices.py` script.

## Part A: Ticker Normalization Helper

### Location
`helpers/ticker_normalize.py`

### Function Signature
```python
def normalize_ticker(t: str) -> str:
    """
    Normalize ticker symbol to canonical format.
    
    Handles:
    - None values → empty string
    - Whitespace trimming
    - Uppercase conversion
    - Unicode dash/hyphen variants → standard hyphen (-)
    - Dots (.) → hyphens (-)
    """
```

### Examples
```python
normalize_ticker("brk.b")     # → "BRK-B"
normalize_ticker("BRK–B")     # → "BRK-B" (en-dash)
normalize_ticker("BRK—B")     # → "BRK-B" (em-dash)
normalize_ticker("  aapl  ")  # → "AAPL"
normalize_ticker(None)        # → ""
```

## Part B: Build Script Enhancements

### Changes to `build_wave_history_from_prices.py`

#### 1. Ticker Normalization Application
- All tickers from `wave_weights.csv` are normalized via new `ticker_norm` column
- All tickers from `prices.csv` are normalized before processing
- Benchmark tickers are normalized before lookup

#### 2. Wave Coverage Validation (90% Threshold)

**Before**: Waves were invalidated if ANY ticker was missing

**After**: Waves remain valid if they have ≥ 90% weight coverage

**Example**:
```
Wave has 5 tickers:
  - AAPL: 20% weight ✓ available
  - MSFT: 20% weight ✓ available
  - GOOGL: 20% weight ✓ available
  - MISSING1: 20% weight ✗ missing
  - MISSING2: 20% weight ✗ missing

Coverage: 60% (3/5 tickers by weight)
Result: Wave EXCLUDED (below 90% threshold)
```

```
Wave has 5 tickers:
  - AAPL: 30% weight ✓ available
  - MSFT: 30% weight ✓ available
  - GOOGL: 30% weight ✓ available
  - SMALL1: 5% weight ✗ missing
  - SMALL2: 5% weight ✗ missing

Coverage: 90% (90% of weight available)
Result: Wave INCLUDED (meets 90% threshold)
```

#### 3. Proportional Reweighting

When tickers are missing, available tickers are proportionally reweighted:

**Original weights**: A=0.1, B=0.2, C=0.3, D=0.2 (missing), E=0.2 (missing)
**Available sum**: 0.6
**Reweighted**: A=0.167, B=0.333, C=0.500 (sum=1.0)

This maintains the relative proportions among available tickers.

#### 4. Coverage Snapshot Output

The script generates `wave_coverage_snapshot.json` with:

```json
{
  "timestamp": "2026-01-02T17:09:19.945166",
  "total_waves": 25,
  "waves_meeting_threshold": 25,
  "waves_below_threshold": 0,
  "min_coverage_threshold": 0.9,
  "waves": [
    {
      "wave": "AI & Cloud MegaCap Wave",
      "total_tickers": 10,
      "available_tickers": 10,
      "missing_tickers": 0,
      "missing_ticker_list": [],
      "total_weight": 0.78,
      "available_weight": 0.78,
      "coverage_pct": 1.0,
      "meets_threshold": true
    },
    ...
  ]
}
```

## Configuration

### Adjusting Coverage Threshold

To change the minimum coverage threshold, edit `build_wave_history_from_prices.py`:

```python
# Current setting: 90%
MIN_COVERAGE_THRESHOLD = 0.90

# For stricter validation (95%):
MIN_COVERAGE_THRESHOLD = 0.95

# For more lenient validation (80%):
MIN_COVERAGE_THRESHOLD = 0.80
```

### Adding Benchmark Mappings

To add or update wave benchmark mappings, edit the `BENCHMARK_BY_WAVE` dictionary:

```python
BENCHMARK_BY_WAVE = {
    "Your Wave Name": "BENCHMARK_TICKER",
    # ... existing mappings
}
```

## Testing

### Run All Tests

```bash
# Test ticker normalization
python test_ticker_normalize.py

# Test wave coverage computation
python test_wave_coverage_computation.py

# Run demonstration
python demo_ticker_normalization.py
```

### Test Results

All tests passing:
- ✓ Ticker normalization (7 test cases)
- ✓ Coverage computation (4 test scenarios)
- ✓ Proportional reweighting validation
- ✓ Threshold validation

## Build Output

### Console Output

```
Loading wave weights...
Normalizing tickers...
Found 124 unique tickers in wave weights and benchmarks
Found existing prices.csv
Loading prices...
Computing daily returns...
[INFO] Wave 'Test Wave' missing 2 tickers: ['MISSING1', 'MISSING2']
       Coverage: 60.00% (threshold: 90.00%)
[WARN] Wave 'Test Wave' coverage 60.00% is below 90.00% threshold. Skipping.
Writing wave_history.csv with 12475 rows...
Writing coverage snapshot to wave_coverage_snapshot.json...

======================================================================
COVERAGE SUMMARY
======================================================================
Total waves processed: 25
Waves meeting 90% threshold: 25
Waves below threshold: 0

Waves below threshold:
======================================================================
Done.
```

### Generated Files

- `wave_history.csv` - Updated wave history with normalized tickers
- `wave_coverage_snapshot.json` - Detailed coverage metrics (gitignored)

## Integration with Existing Code

The normalization helper is standalone and can be imported by other scripts:

```python
# Method 1: Direct import (if streamlit dependencies available)
from helpers.ticker_normalize import normalize_ticker

# Method 2: Dynamic import (avoids streamlit dependency)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ticker_normalize",
    os.path.join(os.path.dirname(__file__), "helpers", "ticker_normalize.py")
)
ticker_normalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ticker_normalize)
normalize_ticker = ticker_normalize.normalize_ticker
```

## Benefits

1. **Consistent Ticker Handling**: All ticker variations (BRK.B, BRK-B, brk.b) map to same canonical form
2. **Resilient Wave Computation**: Waves aren't invalidated by minor ticker gaps
3. **Transparent Coverage Tracking**: JSON snapshot provides detailed diagnostics
4. **Proportional Accuracy**: Missing tickers don't distort remaining weight distribution
5. **Configurable Threshold**: Easy to adjust coverage requirements per business needs

## Future Enhancements

Potential improvements for future iterations:

1. Add coverage trend tracking over time
2. Implement automatic ticker alias resolution
3. Add email/webhook notifications for coverage drops
4. Create coverage visualization dashboard
5. Integrate with ticker validation service
