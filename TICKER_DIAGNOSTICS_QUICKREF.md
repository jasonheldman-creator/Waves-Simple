# Ticker Failure Diagnostics - Quick Reference

## Quick Start

### 1. Check Wave Readiness
View the diagnostics panel in the app:
- Navigate to **Overview** page
- Expand **🔍 Ticker Failure Root Cause Analysis**

### 2. Run Analytics Pipeline
```bash
# For all waves
python -c "from analytics_pipeline import run_daily_analytics_pipeline; run_daily_analytics_pipeline(all_waves=True, lookback_days=365)"

# For specific wave
python -c "from analytics_pipeline import run_daily_analytics_pipeline; run_daily_analytics_pipeline(all_waves=False, wave_ids=['crypto_broad_growth_wave'], lookback_days=30)"
```

### 3. Review Failures
Check the generated report:
```bash
cat reports/failed_tickers_report.csv
```

## Common Issues & Fixes

### Issue: Wave shows 0% coverage
**Quick Fix**: Run analytics pipeline for that wave

### Issue: BRK.B or other special tickers failing
**Quick Fix**: Automatic normalization handles this (BRK.B → BRK-B)

### Issue: Crypto tickers failing
**Quick Fix**: Automatic normalization handles this (BTC → BTC-USD)

### Issue: New special ticker needs normalization
**Quick Fix**: Add to TICKER_ALIASES in `waves_engine.py`:
```python
TICKER_ALIASES: Dict[str, str] = {
    # ... existing ...
    "YOUR.TICKER": "YOUR-TICKER",
}
```

## Readiness Levels

| Level | Coverage | Days | Features |
|-------|----------|------|----------|
| **Full** | 90%+ | 365+ | All analytics |
| **Partial** | 70%+ | 7+ | Basic analytics |
| **Operational** | 50%+ | 1+ | Current pricing |
| **Unavailable** | <50% | <1 | None |

## Failure Types

| Type | Meaning | Action |
|------|---------|--------|
| **PROVIDER_EMPTY** | No data returned | Check ticker validity |
| **SYMBOL_NEEDS_NORMALIZATION** | Format issue | Add to TICKER_ALIASES |
| **RATE_LIMIT** | Too many requests | Wait and retry |
| **NETWORK_TIMEOUT** | Connection issue | Check network |

## Key Files

- **CSV Report**: `reports/failed_tickers_report.csv`
- **Normalization**: `waves_engine.py` (TICKER_ALIASES)
- **Pipeline**: `analytics_pipeline.py`
- **Diagnostics**: `helpers/ticker_diagnostics.py`

## Testing

Test ticker normalization:
```python
from waves_engine import _normalize_ticker

print(_normalize_ticker("BRK.B"))   # BRK-B
print(_normalize_ticker("BTC"))     # BTC-USD
print(_normalize_ticker("AAPL"))    # AAPL
```

Test with dummy data:
```python
from analytics_pipeline import run_daily_analytics_pipeline
run_daily_analytics_pipeline(all_waves=True, lookback_days=7, use_dummy_data=True)
```

## Full Documentation

See `TICKER_DIAGNOSTICS_GUIDE.md` for comprehensive documentation.
