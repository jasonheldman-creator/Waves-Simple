# Crypto Volatility Overlay Integration Guide

## Overview

The crypto volatility overlay system provides crypto-specific risk management for crypto waves (Crypto L1 Growth Wave, Crypto DeFi Growth Wave, Crypto L2 Growth Wave, Crypto AI Growth Wave, Crypto Broad Growth Wave, and Crypto Income Wave).

## Architecture

### Wave Detection
- Waves are identified as crypto through the `category` field in `data/wave_registry.csv`
- Categories: `crypto_growth` and `crypto_income`
- Detection functions: `_is_crypto_wave()`, `_is_crypto_growth_wave()`, `_is_crypto_income_wave()`

### Overlay Components (in `waves_engine.py`)

#### Crypto Growth Waves
1. **Trend/Momentum Overlay** (`_crypto_trend_momentum_overlay`)
   - Monitors 60-day trend
   - Regimes: strong_uptrend, uptrend, neutral, downtrend, strong_downtrend
   - Adjusts exposure based on trend strength
   - Increases safe allocation in downtrends

2. **Volatility State Overlay** (`_crypto_volatility_overlay`)
   - Monitors realized volatility (30-day rolling)
   - States: extreme_compression, compression, normal, expansion, extreme_expansion
   - Adjusts exposure based on volatility state

3. **Liquidity Overlay** (`_crypto_liquidity_overlay`)
   - Monitors liquidity/market structure
   - States: strong_volume, normal_volume, weak_volume
   - Adjusts exposure based on liquidity conditions

#### Crypto Income Wave
1. **Stability Overlay** (`_crypto_income_stability_overlay`)
   - Conservative baseline (80% exposure)
   - Stress detection (high vol or downtrend)
   - Exposure cap: min 0.40, max 0.80

2. **Drawdown Guard** (`_crypto_income_drawdown_guard`)
   - Monitors 30-day drawdown
   - States: normal, minor, moderate, severe
   - Increases safe allocation during drawdowns

3. **Liquidity Gate** (`_crypto_income_liquidity_gate`)
   - More conservative than growth waves
   - Ensures adequate liquidity before allocation

### Exposure Floors
- **Crypto Growth Waves**: Minimum 0.20 (20%) exposure
- **Crypto Income Wave**: Minimum 0.40 (40%) exposure

These floors ensure crypto waves maintain minimum market participation even in defensive regimes.

## Integration with UI

### Display Fields

For crypto waves, the UI should display:

1. **Crypto Vol Regime**: Current trend regime (e.g., "↗️ Uptrend", "📉 Strong Downtrend")
2. **Overlay Live**: Status indicator ("✓ Overlay active" or "Overlay inactive")
3. **Exposure**: Current exposure percentage (e.g., "75%")
4. **Volatility State**: Current volatility level (e.g., "🟡 Normal (60.0%)")
5. **Liquidity State**: Current liquidity condition (e.g., "Normal Volume")

### Implementation Example

```python
from helpers.crypto_overlay_diagnostics import (
    get_crypto_overlay_diagnostics,
    format_crypto_regime,
    format_crypto_exposure,
    format_crypto_volatility,
    format_crypto_overlay_status
)

# Get wave history with diagnostics
wave_history = we.compute_history_nav(
    wave_name="Crypto L1 Growth Wave",
    mode="Standard",
    days=90,
    include_diagnostics=True
)

# Extract crypto overlay diagnostics
crypto_diag = get_crypto_overlay_diagnostics("Crypto L1 Growth Wave", wave_history)

# Display in UI
if crypto_diag and crypto_diag.get('overlay_active'):
    st.metric("Crypto Vol Regime", format_crypto_regime(crypto_diag))
    st.metric("Exposure", format_crypto_exposure(crypto_diag))
    st.metric("Volatility", format_crypto_volatility(crypto_diag))
    st.caption(format_crypto_overlay_status(crypto_diag))
```

## Portfolio Integration

The crypto overlay is already integrated into `compute_history_nav()` in `waves_engine.py`. The overlay is applied during daily NAV computation, and the post-overlay returns are automatically used by portfolio aggregation functions.

### No Double-Scaling

The portfolio snapshot functions in `helpers/wave_performance.py` use wave returns directly from `compute_history_nav()`. The overlay is applied **once** in the wave computation, not again during portfolio aggregation.

## Testing

Run the crypto overlay tests:

```bash
python3 -m pytest test_crypto_volatility_overlay.py -v
```

Tests include:
- Crypto wave identification
- Overlay field computation
- Exposure floor enforcement
- Graceful degradation
- Regime classification
- Portfolio integration

## CI Enforcement

The GitHub Actions workflow `.github/workflows/validate_crypto_overlay.yml` runs on every PR that touches:
- `test_crypto_volatility_overlay.py`
- `waves_engine.py`
- `helpers/wave_performance.py`
- `data/wave_registry.csv`

PRs are blocked unless tests pass.

## Graceful Degradation

The crypto overlay system gracefully handles:
- Missing price data (returns empty DataFrame)
- Missing benchmark data (uses BTC/ETH fallback)
- Network errors (skips data download)
- Incomplete history (computes with available data)

## Future Enhancements

1. **Real-time volume data**: Replace proxy volume calculation with actual volume metrics
2. **Multiple BTC/ETH fallbacks**: Support multiple fallback benchmarks
3. **Custom overlay parameters**: Allow per-wave overlay configuration
4. **Overlay history visualization**: Chart showing overlay state over time
5. **Alpha attribution**: Break down alpha into selection vs. overlay components
