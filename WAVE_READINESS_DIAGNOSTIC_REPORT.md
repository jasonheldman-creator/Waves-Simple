2025-12-29 13:10:42.682 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
2025-12-29 13:10:42.683 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
2025-12-29 13:10:42.684 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
2025-12-29 13:10:42.684 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
# WAVES Intelligence™ - Comprehensive Readiness Diagnostic Report

**Generated:** 2025-12-29 13:10:42

## 1. Ground Truth - Wave Universe Verification

- **Total Waves (Engine):** 28
- **Total Waves (Registry):** 28
- **Total Waves (Weights):** 28
- **Consistency:** ✓ CONSISTENT

## 2. Wave Readiness Summary

**Total Waves:** 28

### By Status
- Full Ready: 0 (0.0%)
- Partial Ready: 9 (32.1%)
- Operational: 3 (10.7%)
- Unavailable: 16 (57.1%)

### By Grade
- A (Excellent): 0
- B (Good): 6
- C (Acceptable): 3
- D (Poor): 3
- F (Failing): 16

## 3. Per-Wave Diagnostic Details

| Wave | Status | Grade | Coverage | Days | Primary Issue |
|------|--------|-------|----------|------|---------------|
| Next-Gen Compute & Semis Wave | operational | D | 60.0% | 500 | STALE_DATA |
| US Small-Cap Disruptors Wave | operational | D | 50.0% | 500 | STALE_DATA |
| Vector Treasury Ladder Wave | operational | D | 60.0% | 500 | STALE_DATA |
| AI & Cloud MegaCap Wave | partial | C | 70.0% | 500 | STALE_DATA |
| Crypto Broad Growth Wave | partial | B | 100.0% | 10 | OK |
| Future Energy & EV Wave | partial | C | 70.0% | 500 | STALE_DATA |
| Future Power & Energy Wave | partial | C | 70.0% | 500 | STALE_DATA |
| Gold Wave | partial | B | 100.0% | 20 | OK |
| Income Wave | partial | B | 100.0% | 20 | OK |
| Quantum Computing Wave | partial | B | 87.5% | 500 | STALE_DATA |
| S&P 500 Wave | partial | B | 100.0% | 20 | OK |
| US MegaCap Core Wave | partial | B | 100.0% | 10 | OK |
| Clean Transit-Infrastructure Wave | unavailable | F | 20.0% | 500 | LOW_COVERAGE |
| Crypto AI Growth Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| Crypto DeFi Growth Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| Crypto Income Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| Crypto L1 Growth Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| Crypto L2 Growth Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| Demas Fund Wave | unavailable | F | 20.0% | 500 | LOW_COVERAGE |
| EV & Infrastructure Wave | unavailable | F | 10.0% | 500 | LOW_COVERAGE |
| Infinity Multi-Asset Growth Wave | unavailable | F | 44.4% | 500 | LOW_COVERAGE |
| Russell 3000 Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| Small Cap Growth Wave | unavailable | F | 25.0% | 500 | LOW_COVERAGE |
| Small to Mid Cap Growth Wave | unavailable | F | 20.0% | 500 | LOW_COVERAGE |
| SmartSafe Tax-Free Money Market Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| SmartSafe Treasury Cash Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| US Mid/Small Growth & Semis Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |
| Vector Muni Ladder Wave | unavailable | F | 0.0% | 0 | MISSING_PRICES |

## 4. Root Cause Analysis

### Failure Distribution

- **MISSING_PRICES**: 10 waves (35.7%)
- **STALE_DATA**: 7 waves (25.0%)
- **LOW_COVERAGE**: 6 waves (21.4%)
- **READY**: 5 waves (17.9%)

## 5. Recommended Actions

1. **CRITICAL**: Run analytics pipeline for 10 waves with MISSING_PRICES
   ```bash
   python analytics_pipeline.py --all-waves --lookback=14
   ```

2. **HIGH**: Investigate ticker download failures for 6 waves with LOW_COVERAGE
   - Review ticker diagnostics and circuit breaker logs

