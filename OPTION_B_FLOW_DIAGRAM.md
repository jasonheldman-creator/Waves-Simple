# Option B: Staleness Handling Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENVIRONMENT VARIABLES                           │
│                                                                         │
│  PRICE_CACHE_OK_DAYS=14          PRICE_CACHE_DEGRADED_DAYS=30         │
│  (configurable via env)          (configurable via env)                │
└────────────────────────┬────────────────────────┬───────────────────────┘
                         │                        │
                         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      helpers/price_loader.py                            │
│                                                                         │
│  1. Read environment variables with validation                         │
│  2. Validate constraint: DEGRADED_DAYS > OK_DAYS                       │
│  3. check_cache_readiness() determines staleness:                      │
│                                                                         │
│     days_since_update = (now - max_cache_date).days                    │
│                                                                         │
│     if days_since_update > PRICE_CACHE_DEGRADED_DAYS:                 │
│         status = 'STALE'        # >30 days                             │
│     elif days_since_update > PRICE_CACHE_OK_DAYS:                     │
│         status = 'DEGRADED'     # 15-30 days                           │
│     else:                                                               │
│         status = 'OK/READY'     # ≤14 days                             │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      helpers/price_book.py                              │
│                                                                         │
│  1. Imports same thresholds from environment                           │
│  2. Aliases legacy constants for backward compatibility:               │
│     - STALE_DAYS_THRESHOLD = PRICE_CACHE_DEGRADED_DAYS (30)           │
│     - DEGRADED_DAYS_THRESHOLD = PRICE_CACHE_OK_DAYS (14)              │
│  3. compute_system_health() uses these thresholds                      │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           app.py (UI)                                   │
│                                                                         │
│  Display Logic:                                                         │
│                                                                         │
│  if data_age > STALE_DAYS_THRESHOLD (30):                             │
│      Display: "❌ {data_age} days (STALE)"                             │
│      Action: Show warning message                                       │
│                                                                         │
│  elif data_age > DEGRADED_DAYS_THRESHOLD (14):                        │
│      Display: "⚠️ {data_age} days (DEGRADED)"                          │
│      Action: Show info message (consider refresh)                      │
│                                                                         │
│  else: # data_age ≤ 14 days                                            │
│      Display: "{data_age} days"                                        │
│      Action: Normal display (no warning)                                │
└─────────────────────────────────────────────────────────────────────────┘


                         THREE-TIER SYSTEM

┌──────────────────┬──────────────────┬──────────────────────┐
│   OK (GREEN)     │  DEGRADED (YELLOW)│   STALE (RED)       │
├──────────────────┼──────────────────┼──────────────────────┤
│   ≤14 days       │   15-30 days     │   >30 days           │
├──────────────────┼──────────────────┼──────────────────────┤
│   ✅ Fresh       │   ⚠️ Consider    │   ❌ Needs           │
│   No action      │   refreshing     │   refresh            │
│   needed         │                  │                      │
├──────────────────┼──────────────────┼──────────────────────┤
│ Status: READY    │ Status: DEGRADED │ Status: STALE        │
│ Ready: True      │ Ready: True      │ Ready: False         │
└──────────────────┴──────────────────┴──────────────────────┘


                      VALIDATION METRICS

┌─────────────────────────────────────────────────────────────┐
│  Required Metrics (Problem Statement):                      │
│                                                              │
│  1. Missing Tickers = 0         ✅ Actual: 0                │
│  2. Coverage = 100%             ✅ Actual: 100.0%           │
│  3. Health = GREEN/STABLE       ✅ Actual: READY            │
│                                                              │
│  Current Cache Status:                                       │
│  - Days Stale: 9 days (OK threshold)                        │
│  - Status: READY                                             │
│  - Cache Age Accurately Displayed: ✅                        │
└─────────────────────────────────────────────────────────────┘


                    ERROR HANDLING FLOW

┌─────────────────────────────────────────────────────────────┐
│  Environment Variable Validation:                            │
│                                                              │
│  1. Invalid PRICE_CACHE_OK_DAYS (non-integer)               │
│     → Log warning                                            │
│     → Use default: 14                                        │
│                                                              │
│  2. Invalid PRICE_CACHE_DEGRADED_DAYS (non-integer)         │
│     → Log warning                                            │
│     → Use default: 30                                        │
│                                                              │
│  3. Constraint Violation (DEGRADED_DAYS ≤ OK_DAYS)          │
│     → Log warning                                            │
│     → Reset both to defaults: OK=14, DEGRADED=30            │
│                                                              │
│  Example:                                                    │
│    export PRICE_CACHE_OK_DAYS="invalid"                     │
│    → Warning: "Invalid PRICE_CACHE_OK_DAYS, using default" │
│    → PRICE_CACHE_OK_DAYS = 14                               │
└─────────────────────────────────────────────────────────────┘


                BACKWARD COMPATIBILITY

┌─────────────────────────────────────────────────────────────┐
│  Legacy Code:                                                │
│    if data_age > STALE_DAYS_THRESHOLD:                      │
│        # Data is stale                                       │
│                                                              │
│  Still Works Because:                                        │
│    STALE_DAYS_THRESHOLD = PRICE_CACHE_DEGRADED_DAYS (30)   │
│                                                              │
│  No Breaking Changes:                                        │
│    ✅ All existing imports work                              │
│    ✅ All existing logic works                               │
│    ✅ All existing tests work                                │
└─────────────────────────────────────────────────────────────┘
```
