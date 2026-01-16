# Implemented Changes

## 1. PRICE_BOOK Change at Runtime
- Option A: Price data fetched from `yfinance` ensuring at least one ticker (e.g., SPY) refreshes on each render.
- Existing cache mechanism for PRICE_BOOK bypassed.

## 2. Portfolio Snapshot Metrics
- Computed directly from PRICE_BOOK without dependency on `live_snapshot.csv` or cached artifacts.
- `st.cache_data` disabled in the metrics computation pipeline.

## 3. Expanded Diagnostics Overlay
- Fields added to display:
  - PRICE_BOOK source: live API, cache refresh, or mutation mode.
  - PRICE_BOOK shape and the latest date.
  - Render UTC timestamp.
  - Confirmations reading:
    - `live_snapshot.csv: NOT USED`
    - `metrics caching: DISABLED`

## 4. Visual Proof Requirement
- Attach screenshots proving functionality:
  - Screenshot A: Portfolio Snapshot with diagnostics expanded and rendered UTC timestamp (2026-01-16 14:28:53).
  - Screenshot B: Same view with changed numeric values and a new UTC timestamp.
  
**Note:** No review or merge will proceed unless visual proof is provided and attached to this PR.