# Income Wave Strategy Family Implementation

## Overview

This implementation introduces a distinct set of algorithmic overlays for non-crypto Income Waves, enabling comparisons between three strategy families:
1. **Equity Growth/Thematic** (existing logic)
2. **Crypto** (crypto-specific logic)
3. **Non-Crypto Income** (new income-first logic)

## Implementation Details

### 1. Income Wave Identification

**Function:** `_is_income_wave(wave_name: str) -> bool`

Identifies the following non-crypto income waves:
- Income Wave
- Vector Treasury Ladder Wave
- Vector Muni Ladder Wave
- SmartSafe Treasury Cash Wave
- SmartSafe Tax-Free Money Market Wave

### 2. Income-Specific Overlays

#### A) Rates/Duration Regime Overlay

**Purpose:** Monitor trends for rates proxies (TNX - 10-year Treasury) to calibrate duration sensitivity.

**Functions:**
- `_rates_duration_regime(tnx_trend: float) -> str`
- `_rates_duration_overlay(tnx_trend: float) -> tuple[float, str]`

**Regimes:**
- `rising_fast`: TNX trend >= +10% → 80% exposure (reduce duration)
- `rising`: TNX trend >= +3% → 90% exposure
- `stable`: TNX trend between -3% and +3% → 100% exposure
- `falling`: TNX trend <= -10% → 105% exposure
- `falling_fast`: TNX trend < -10% → 110% exposure (increase duration)

#### B) Credit/Risk Regime Overlay

**Purpose:** Evaluate credit risk via HYG vs LQD relative strength to assess credit stress.

**Functions:**
- `_credit_risk_regime(hyg_lqd_spread: float) -> str`
- `_credit_risk_overlay(hyg_lqd_spread: float) -> tuple[float, float, str]`

**Regimes:**
- `risk_on`: HYG outperforming LQD by 2%+ → 105% exposure, 0% safe boost
- `neutral`: Spread within ±2% → 100% exposure, 5% safe boost
- `risk_off`: LQD outperforming (spread < -2%) → 90% exposure, 15% safe boost

#### C) Carry + Drawdown Guard Overlay

**Purpose:** Implement protections based on drawdown and volatility spike conditions.

**Function:** `_drawdown_guard_overlay(current_nav, peak_nav, recent_vol) -> tuple[float, str]`

**Stress States:**
- `normal`: No significant drawdown → 0% safe boost
- `minor`: Drawdown <= -3% → 10% safe boost
- `moderate`: Drawdown <= -5% → 20% safe boost
- `severe`: Drawdown <= -8% or volatility spike → 30% safe boost

#### D) Turnover Discipline Overlay

**Purpose:** Enforce low-turnover disciplines.

**Configuration:**
- `INCOME_MIN_REBALANCE_DAYS = 5` - Minimum days between rebalances
- `INCOME_MAX_TURNOVER_PER_PERIOD = 0.20` - Max 20% turnover unless strong signals

**Note:** Currently implemented as a placeholder in the strategy contribution framework.

### 3. Overlay Independence

**Key Features:**
- Income waves **do NOT use** equity VIX overlays
- Income waves **do NOT use** equity regime detection overlays
- Income waves **do NOT use** crypto overlays
- Crypto waves **do NOT use** income overlays
- Equity growth waves **do NOT use** income or crypto overlays

**Implementation:** The `_compute_core()` function detects wave type using:
- `is_crypto = _is_crypto_wave(wave_name)`
- `is_income = _is_income_wave(wave_name)`

And conditionally applies appropriate overlays based on wave type.

### 4. Attribution Labels

**Diagnostics Include:**
- `strategy_family`: "income", "crypto", or "equity_growth"
- `income_rates_regime`: Rates regime state (for income waves)
- `income_credit_regime`: Credit regime state (for income waves)
- `income_stress_state`: Drawdown stress level (for income waves)

**Strategy Contributions:**
- `income_rates_regime`: Rates/duration overlay contribution
- `income_credit_regime`: Credit/risk overlay contribution
- `income_drawdown_guard`: Drawdown guard contribution
- `income_turnover_discipline`: Turnover discipline placeholder

## Testing

### Test Coverage

**File:** `test_income_overlays.py`

**Tests:**
1. ✓ Income wave detection
2. ✓ Income wave exclusivity (not crypto, not equity growth)
3. ✓ Rates/duration regime classification
4. ✓ Credit/risk regime classification
5. ✓ Drawdown guard functionality
6. ✓ Income wave NAV computation with diagnostics
7. ✓ Equity growth waves unaffected
8. ✓ Crypto waves unaffected

**Results:** All 8 tests passing

### Running Tests

```bash
python test_income_overlays.py
```

## Usage Example

```python
import waves_engine as we

# Compute Income Wave NAV with diagnostics
result = we.compute_history_nav("Income Wave", "Standard", 365, include_diagnostics=True)

# Access diagnostics
if hasattr(result, 'attrs') and 'diagnostics' in result.attrs:
    diag = result.attrs['diagnostics']
    
    # Check strategy family
    print(diag['strategy_family'].unique())  # ['income']
    
    # View income-specific overlays
    print(diag[['income_rates_regime', 'income_credit_regime', 'income_stress_state']])
    
    # Verify equity overlays are disabled
    print(diag['regime'].unique())  # ['n/a']
```

## Future Enhancements

1. **Turnover Monitoring:** Implement actual turnover calculation and enforcement
2. **Attribution Integration:** Extend `alpha_attribution.py` to support income-specific attribution buckets
3. **Additional Income Waves:** Identify and tag any new income-focused waves
4. **Dynamic Thresholds:** Make overlay thresholds configurable per wave
5. **Yield Overlay:** Add explicit yield/carry tracking for income waves

## Compatibility

- ✓ Preserves compatibility with existing systems
- ✓ No breaking changes to non-income waves
- ✓ Equity growth waves continue using VIX/regime overlays
- ✓ Crypto waves continue using crypto-specific overlays
- ✓ Income waves use distinct income overlays only
