# Crypto Volatility Overlay - Phase 1B.2 Implementation

## Overview

The Crypto Volatility Overlay module provides crypto-native volatility-regime scaling for portfolio exposure management. It implements deterministic overlay logic based on volatility and drawdown thresholds to dynamically adjust portfolio exposure in response to market conditions.

## Key Features

- **Volatility-based regime classification**: Analyzes 30-day realized volatility
- **Drawdown monitoring**: Tracks peak-to-trough drawdowns over 60 days
- **Four regime states**: LOW, MED, HIGH, CRISIS
- **Exposure scaling**: 0.2 (20% minimum in crisis) to 1.0 (100% full exposure)
- **Cache-only operation**: Uses cached price data, no live fetching
- **Multi-benchmark support**: Aggregates signals from BTC, ETH, or other crypto benchmarks

## Installation

The module is located at `helpers/crypto_volatility_overlay.py` and can be imported as follows:

```python
from helpers.crypto_volatility_overlay import compute_crypto_overlay
```

Or from the helpers package:

```python
from helpers import compute_crypto_overlay
```

## Usage

### Basic Usage

```python
from helpers.crypto_volatility_overlay import compute_crypto_overlay
from helpers.price_book import get_price_book

# Load cached price data for BTC and ETH
price_data = get_price_book(active_tickers=['BTC-USD', 'ETH-USD'])

# Compute overlay
overlay = compute_crypto_overlay(
    benchmarks=['BTC-USD', 'ETH-USD'],
    price_data=price_data,
    vol_window=30,   # 30-day volatility lookback
    dd_window=60     # 60-day drawdown lookback
)

print(f"Regime: {overlay['regime']}")
print(f"Exposure: {overlay['exposure']:.1%}")
```

### Output Structure

The `compute_crypto_overlay` function returns a dictionary with the following fields:

```python
{
    'overlay_label': 'Crypto Vol',           # Fixed label
    'regime': 'MED',                         # LOW, MED, HIGH, or CRISIS
    'exposure': 0.75,                        # Exposure multiplier (0.2-1.0)
    'volatility': 0.461,                     # Annualized realized volatility
    'max_drawdown': -0.253,                  # Maximum drawdown (negative %)
    'vol_regime': 'medium',                  # Volatility classification
    'dd_severity': 'minor'                   # Drawdown severity
}
```

### Portfolio Integration

```python
# Define base portfolio allocation
base_allocation = {
    'BTC-USD': 0.40,
    'ETH-USD': 0.30,
    'SOL-USD': 0.20,
    'AVAX-USD': 0.10
}

# Get overlay
overlay = compute_crypto_overlay(
    benchmarks=['BTC-USD', 'ETH-USD'],
    price_data=price_data
)

# Scale allocations by exposure
scaled_allocation = {
    ticker: weight * overlay['exposure']
    for ticker, weight in base_allocation.items()
}

# Calculate cash allocation
cash = 1.0 - sum(scaled_allocation.values())
```

## Regime Classification

### Volatility Regimes

| Regime | Annualized Volatility | Description |
|--------|----------------------|-------------|
| Low | < 40% | Stable, low-volatility conditions |
| Medium | 40-80% | Normal crypto volatility |
| High | 80-120% | Elevated volatility |
| Extreme | > 120% | Extreme volatility (crisis candidate) |

### Drawdown Severity

| Severity | Drawdown Range | Description |
|----------|---------------|-------------|
| None | 0% to -15% | Minimal drawdown |
| Minor | -15% to -30% | Minor correction |
| Moderate | -30% to -50% | Moderate drawdown |
| Severe | -50% to -60% | Severe drawdown |
| Critical | < -60% | Critical drawdown (triggers crisis) |

### Overall Regime Determination

The overall regime is determined by combining volatility and drawdown signals:

- **CRISIS**: Triggered by critical drawdown (< -60%) OR severe drawdown + extreme volatility
- **HIGH**: Moderate or severe drawdown, OR high/extreme volatility
- **MED**: Medium volatility with minor/no drawdown
- **LOW**: Low volatility with minimal drawdown

## Exposure Levels

| Regime | Exposure Multiplier | Description |
|--------|-------------------|-------------|
| LOW | 1.00 (100%) | Full exposure - favorable conditions |
| MED | 0.75 (75%) | Moderate exposure - normal conditions |
| HIGH | 0.50 (50%) | Reduced exposure - elevated risk |
| CRISIS | 0.20 (20%) | Minimum exposure - crisis conditions |

**Crisis Hard Threshold**: The crisis regime implements a minimum exposure floor of 20% (≤ 20% val) as specified in the requirements.

## Configuration

### Default Parameters

```python
compute_crypto_overlay(
    benchmarks=['BTC-USD', 'ETH-USD'],
    price_data=price_data,
    vol_window=30,      # Default: 30 days
    dd_window=60        # Default: 60 days
)
```

### Custom Lookback Windows

You can customize the volatility and drawdown lookback periods:

```python
# Short-term responsive overlay
overlay = compute_crypto_overlay(
    benchmarks=['BTC-USD', 'ETH-USD'],
    price_data=price_data,
    vol_window=14,      # 2-week volatility
    dd_window=30        # 1-month drawdown
)

# Long-term stable overlay
overlay = compute_crypto_overlay(
    benchmarks=['BTC-USD', 'ETH-USD'],
    price_data=price_data,
    vol_window=60,      # 2-month volatility
    dd_window=90        # 3-month drawdown
)
```

## Technical Details

### Volatility Calculation

Realized volatility is computed using daily returns over the specified window:

1. Calculate daily returns: `r_t = (P_t - P_{t-1}) / P_{t-1}`
2. Compute daily standard deviation: `σ_daily = std(r_t)`
3. Annualize: `σ_annual = σ_daily × √252`

### Drawdown Calculation

Maximum drawdown is computed as the largest peak-to-trough decline:

1. Calculate running maximum (peak): `peak_t = max(P_0, ..., P_t)`
2. Calculate drawdown: `dd_t = (P_t - peak_t) / peak_t`
3. Return minimum (most negative): `max_dd = min(dd_t)`

### Multi-Benchmark Aggregation

When multiple benchmarks are provided, the overlay uses **worst-case metrics**:

- **Volatility**: Maximum volatility across all benchmarks
- **Drawdown**: Most severe (most negative) drawdown

This conservative approach ensures portfolio protection when any benchmark shows stress.

## Error Handling

The module includes robust error handling for edge cases:

- **Empty benchmarks**: Returns default MED regime with 75% exposure
- **Missing price data**: Logs warning and uses default regime
- **Insufficient data**: Returns NaN for metrics, defaults to MED regime
- **Invalid inputs**: Handles None and empty DataFrames gracefully

## Testing

Comprehensive tests are provided in `test_crypto_volatility_overlay.py`:

```bash
python test_crypto_volatility_overlay.py
```

Test coverage includes:
- Volatility computation accuracy
- Drawdown calculation correctness
- Regime classification logic
- Exposure scaling validation
- Integration testing
- Edge case handling
- Deterministic behavior verification

## Examples

See `demo_crypto_volatility_overlay.py` for working examples:

```bash
python demo_crypto_volatility_overlay.py
```

The demo includes:
1. Basic usage
2. Different market scenarios
3. Single vs. multi-benchmark
4. Custom lookback windows
5. Portfolio integration

## Integration with Price Book

The overlay is designed to work seamlessly with the canonical price book:

```python
from helpers.price_book import get_price_book
from helpers.crypto_volatility_overlay import compute_crypto_overlay

# Load cached prices (no network fetch)
price_data = get_price_book(active_tickers=['BTC-USD', 'ETH-USD'])

# Compute overlay
overlay = compute_crypto_overlay(
    benchmarks=['BTC-USD', 'ETH-USD'],
    price_data=price_data
)
```

**Important**: The overlay uses cached data only. Ensure price cache is up-to-date by running `build_price_cache.py` or similar cache update utilities.

## Best Practices

1. **Update price cache regularly**: Run cache updates daily or before trading sessions
2. **Use multiple benchmarks**: BTC and ETH provide better signal than single benchmark
3. **Monitor regime changes**: Track when regime transitions occur for portfolio rebalancing
4. **Test with historical data**: Validate overlay behavior during past market events
5. **Combine with other signals**: Use overlay as one component of risk management system

## Limitations

- **Backward-looking**: Based on historical volatility and drawdowns, not forward-looking
- **No microstructure**: Doesn't account for orderbook depth, spread, or liquidity
- **Correlation assumptions**: Assumes crypto assets move together during stress
- **Data dependency**: Requires high-quality, consistent price data

## Future Enhancements (Phase 2)

Potential future improvements:
- On-chain metrics integration (network activity, stablecoin flows)
- Correlation regime detection
- Market structure metrics (funding rates, open interest)
- Machine learning regime prediction
- Real-time streaming overlay updates

## Support

For issues or questions:
1. Review test suite: `test_crypto_volatility_overlay.py`
2. Run demo: `demo_crypto_volatility_overlay.py`
3. Check module documentation: `helpers/crypto_volatility_overlay.py`

## License

This module is part of the Waves-Simple portfolio management system.

---

**Version**: 1.0.0 (Phase 1B.2)  
**Last Updated**: January 2026  
**Status**: Production Ready
