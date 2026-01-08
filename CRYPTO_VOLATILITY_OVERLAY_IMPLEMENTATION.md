# Crypto Volatility Overlay Implementation

## Overview

This implementation adds a crypto-specific volatility control overlay to the portfolio system to manage exposure dynamically based on market regimes. The overlay is completely isolated to crypto Waves and does not affect equity Wave logic or benchmarks.

## Key Features

### 1. Regime Classification
The system classifies crypto market conditions into four regimes based on short-term vs. long-term volatility ratios:

- **Calm** (ratio < 0.7): Low volatility, favorable market conditions
- **Normal** (0.7 ≤ ratio < 1.3): Moderate volatility, typical market conditions
- **Elevated** (1.3 ≤ ratio < 2.0): High volatility, caution warranted
- **Stress** (ratio ≥ 2.0): Extreme volatility, defensive positioning required

### 2. Volatility Computation
- Uses BTC-USD and ETH-USD as signal tickers
- Computes short-term volatility (10-day window)
- Computes long-term volatility (30-day window)
- Calculates ratio: short_vol / long_vol
- Averages BTC and ETH ratios when both available

### 3. Exposure Scaling

#### Growth Waves (Aggressive)
- Calm: 100% exposure
- Normal: 85% exposure
- Elevated: 60% exposure
- Stress: 30% exposure

#### Income Waves (Conservative)
- Calm: 100% exposure
- Normal: 90% exposure
- Elevated: 70% exposure
- Stress: 50% exposure

### 4. Data Safety
The system enforces strict data safety:
- Returns full default exposure (100%) when data is insufficient
- Requires minimum 40 data points for regime computation
- Gracefully handles missing BTC or ETH data
- Uses partial data if only one signal is available

## Architecture

### Module Structure

```
helpers/
  crypto_volatility_overlay.py    # Core crypto overlay module
  wave_performance.py              # Integration with performance diagnostics

tests/
  test_crypto_volatility_overlay.py       # Unit tests for overlay
  test_crypto_volatility_integration.py   # Integration tests
```

### Key Functions

#### `helpers/crypto_volatility_overlay.py`

1. **`compute_volatility_ratio(price_series, short_window, long_window)`**
   - Computes the ratio of short-term to long-term volatility
   - Returns None if insufficient data

2. **`classify_regime(vol_ratio)`**
   - Maps volatility ratio to regime classification
   - Returns 'calm', 'normal', 'elevated', or 'stress'

3. **`compute_exposure_scaling(regime, is_growth)`**
   - Returns exposure scaling factor for given regime and strategy type
   - Separate handling for growth vs. income waves

4. **`compute_crypto_volatility_regime(price_book)`**
   - Main function to compute crypto market regime
   - Returns comprehensive regime data with exposure recommendations

5. **`get_crypto_wave_exposure(wave_name, price_book)`**
   - Convenience function for wave-specific exposure recommendations

#### `helpers/wave_performance.py`

1. **`compute_crypto_regime_diagnostics(price_book, wave_name)`**
   - Computes crypto diagnostics for a specific wave
   - Returns N/A for equity waves (isolation)

2. **`compute_all_waves_performance_with_crypto_diagnostics(...)`**
   - Extends standard performance computation with crypto diagnostics
   - Adds columns: Crypto_Regime, Crypto_Exposure, Crypto_Overlay_Status

## Isolation from Equity Waves

The crypto overlay is completely isolated from equity wave logic:

1. **Wave Detection**: Uses `_is_crypto_wave()` to identify crypto waves
2. **Separate Overlays**: Crypto waves use BTC/ETH-based regime, equity waves use VIX-based regime
3. **No Cross-Contamination**: Equity wave diagnostics return N/A for crypto fields
4. **Independent Configuration**: Crypto overlay has its own thresholds and parameters

## Usage Examples

### Basic Regime Computation

```python
from helpers.crypto_volatility_overlay import compute_crypto_volatility_regime
from helpers.price_book import get_price_book

price_book = get_price_book()
regime_result = compute_crypto_volatility_regime(price_book)

print(f"Regime: {regime_result['regime']}")
print(f"Growth Exposure: {regime_result['growth_exposure']:.2%}")
print(f"Income Exposure: {regime_result['income_exposure']:.2%}")
print(f"Data Quality: {regime_result['data_quality']}")
```

### Wave-Specific Exposure

```python
from helpers.crypto_volatility_overlay import get_crypto_wave_exposure
from helpers.price_book import get_price_book

price_book = get_price_book()

# For a growth wave
result = get_crypto_wave_exposure("Crypto L1 Growth Wave", price_book)
print(f"Exposure: {result['exposure']:.2%}")
print(f"Regime: {result['regime']}")

# For an income wave
result = get_crypto_wave_exposure("Crypto Income Wave", price_book)
print(f"Exposure: {result['exposure']:.2%} (more conservative)")
```

### Integration with Performance Diagnostics

```python
from helpers.wave_performance import compute_crypto_regime_diagnostics
from helpers.price_book import get_price_book

price_book = get_price_book()

# For a crypto wave
diag = compute_crypto_regime_diagnostics(price_book, "Crypto L1 Growth Wave")
print(f"Applicable: {diag['applicable']}")  # True
print(f"Regime: {diag['regime']}")
print(f"Exposure: {diag['exposure']:.2%}")

# For an equity wave
diag = compute_crypto_regime_diagnostics(price_book, "US MegaCap Core Wave")
print(f"Applicable: {diag['applicable']}")  # False
print(f"Regime: {diag['regime']}")  # 'N/A'
```

### Performance Table with Crypto Diagnostics

```python
from helpers.wave_performance import compute_all_waves_performance_with_crypto_diagnostics
from helpers.price_book import get_price_book

price_book = get_price_book()

df = compute_all_waves_performance_with_crypto_diagnostics(
    price_book=price_book,
    periods=[1, 30, 60, 365],
    include_crypto_diagnostics=True
)

# Display crypto-specific columns
crypto_waves = df[df['Crypto_Regime'] != 'N/A']
print(crypto_waves[['Wave', 'Crypto_Regime', 'Crypto_Exposure', 'Crypto_Overlay_Status']])
```

## Testing

### Unit Tests (`test_crypto_volatility_overlay.py`)

Comprehensive unit tests covering:
- Volatility ratio computation
- Regime classification (all 4 regimes)
- Exposure scaling for growth and income waves
- Signal extraction (BTC, ETH, both, neither)
- Data safety (insufficient data, missing data, empty inputs)
- Exposure limits validation

Run: `python test_crypto_volatility_overlay.py`

### Integration Tests (`test_crypto_volatility_integration.py`)

Integration tests validating:
- Integration with wave_performance module
- Crypto regime diagnostics for crypto waves
- N/A handling for equity waves
- Proper isolation between crypto and equity waves

Run: `python test_crypto_volatility_integration.py`

### All Tests Pass
```
CRYPTO VOLATILITY OVERLAY TEST SUITE
================================================================================
TEST SUMMARY: 12 passed, 0 failed

CRYPTO VOLATILITY OVERLAY INTEGRATION TEST SUITE
================================================================================
TEST SUMMARY: 4 passed, 0 failed
```

## Diagnostics Display

### Crypto Wave Diagnostics
For crypto waves, the diagnostics show:
- **Regime**: Current market regime (calm/normal/elevated/stress)
- **Exposure**: Recommended exposure percentage
- **Overlay Status**: Status message indicating data quality and signals used
  - "Active (BTC, ETH)" - Both signals available
  - "Active (Partial: BTC)" - Only BTC available
  - "Default (No BTC or ETH data available)" - Fallback to default

### Equity Wave Diagnostics
For equity waves:
- **Regime**: N/A
- **Exposure**: N/A
- **Overlay Status**: Not Applicable

## Data Quality Levels

1. **Good**: Both BTC and ETH data available
2. **Partial**: Only one of BTC or ETH available
3. **Insufficient**: Less than 40 data points, returns default exposure

## Configuration Constants

Located in `helpers/crypto_volatility_overlay.py`:

```python
CRYPTO_VOL_SHORT_WINDOW = 10     # Short-term volatility window
CRYPTO_VOL_LONG_WINDOW = 30      # Long-term volatility window
MIN_DATA_POINTS = 40             # Minimum data points required

# Regime thresholds
REGIME_THRESHOLDS = {
    'calm': (0.0, 0.7),
    'normal': (0.7, 1.3),
    'elevated': (1.3, 2.0),
    'stress': (2.0, float('inf'))
}

# Exposure maps
GROWTH_EXPOSURE_MAP = {
    'calm': 1.00,
    'normal': 0.85,
    'elevated': 0.60,
    'stress': 0.30
}

INCOME_EXPOSURE_MAP = {
    'calm': 1.00,
    'normal': 0.90,
    'elevated': 0.70,
    'stress': 0.50
}
```

## Acceptance Criteria Checklist

- [x] Dynamic exposure scaling for crypto income and growth strategies automated
- [x] Transparency through diagnostics panels
- [x] Unit tests validating deterministic, safety-enabled operation
- [x] Completely isolated from non-crypto Waves
- [x] No regressions to existing benchmarks/portfolio systems
- [x] Growth/income strategies handled with different conservative levels
- [x] Data safety: returns full default exposure when insufficient data
- [x] Regime classification based on BTC/ETH volatility ratios
- [x] Diagnostics populated in performance pipeline

## CI Integration

The implementation is tested as part of the standard CI workflow. The crypto volatility overlay tests are included in the test suite and must pass for green CI indicators.

## Future Enhancements

Potential future improvements:
1. Additional crypto signals (e.g., on-chain metrics, funding rates)
2. Machine learning-based regime classification
3. Dynamic threshold adjustment based on market conditions
4. Integration with portfolio rebalancing logic
5. Backtesting framework for overlay optimization
