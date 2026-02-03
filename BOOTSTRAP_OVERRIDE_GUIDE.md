# ALLOW_METADATA_BOOTSTRAP Override Guide

## Problem: Bootstrap Deadlock

The Update Price Cache workflow can enter a bootstrap deadlock when:

1. The cache metadata (`spy_max_date`) becomes stale (e.g., workflow hasn't run in several days)
2. The validation step (`scripts/validate_cache_metadata.py`) blocks the commit because `spy_max_date` doesn't match the latest SPY trading day
3. But advancing `spy_max_date` requires the cache to be built and committed successfully
4. This creates a deadlock where the pipeline cannot self-heal

## Solution: Temporary Bootstrap Override

The `ALLOW_METADATA_BOOTSTRAP` environment variable provides a **temporary, one-time override** to break the deadlock:

- When enabled, trading-day freshness validation logs **warnings** instead of **failing**
- This allows the cache to be built and committed, advancing `spy_max_date` to the current trading day
- After one successful run, the override should be **removed** immediately

## How to Use (GitHub Actions Workflow)

### Step 1: Enable the Override

Run the **Update Price Cache** workflow manually with the override enabled:

1. Go to **Actions** → **Update Price Cache**
2. Click **Run workflow**
3. Set `allow_bootstrap` to **true**
4. Click **Run workflow**

### Step 2: Verify Success

After the workflow completes successfully:

1. Check that `data/cache/prices_cache_meta.json` was committed
2. Verify that `spy_max_date` has advanced to the latest trading day
3. Confirm the workflow logs show:
   ```
   ⚠️  BOOTSTRAP MODE ENABLED (ALLOW_METADATA_BOOTSTRAP=1)
   ⚠ WARNING (BOOTSTRAP MODE): spy_max_date is X trading days behind latest trading day
   ```

### Step 3: Remove the Override

**IMPORTANT**: The override is temporary and must be removed after one successful run.

In a follow-up PR:
1. Remove the `allow_bootstrap` input from `.github/workflows/update_price_cache.yml`
2. Remove the `ALLOW_METADATA_BOOTSTRAP` environment variable handling
3. Remove the bootstrap override logic from `scripts/validate_cache_metadata.py`

## Safety Measures

The override is **strictly limited** to ensure safety:

### What the Override Does
- ✅ Allows Validation 3 (trading-day freshness) to pass with warnings when stale
- ✅ Logs clear warnings indicating bootstrap mode is active
- ✅ Reminds operator to remove the override after use

### What the Override Does NOT Do
- ❌ Does NOT bypass Validation 1 (`spy_max_date` must exist and not be null)
- ❌ Does NOT bypass Validation 2 (`tickers_total` must be >= 50)
- ❌ Does NOT modify cache building logic in `build_price_cache.py`
- ❌ Does NOT introduce any fallback logic
- ❌ Does NOT persist beyond a single workflow run (must be explicitly enabled each time)

## Example Workflow Log

With override enabled:

```
======================================================================
CACHE METADATA VALIDATION
======================================================================

⚠️  BOOTSTRAP MODE ENABLED (ALLOW_METADATA_BOOTSTRAP=1)
   Trading-day freshness will log warnings instead of failing.
   This override should be removed after a successful cache update.

Metadata Values:
  spy_max_date: 2026-01-09
  max_price_date: 2026-01-09
  tickers_total: 124
  tickers_successful: 124
  generated_at_utc: 2026-01-14T20:05:12.643128Z

Validation 1: spy_max_date exists and is not null
✓ PASS: spy_max_date = 2026-01-09

Validation 2: tickers_total >= 50
✓ PASS: tickers_total = 124

Validation 3: spy_max_date matches latest SPY trading day
  spy_max_date from metadata: 2026-01-09
  latest_trading_day from SPY: 2026-01-14
  Difference: 5 calendar days
  Sessions behind: 3 trading days
  Comparison: BEHIND by 3 trading day(s)
⚠ WARNING (BOOTSTRAP MODE): spy_max_date is 3 trading days behind latest trading day
  (Grace period: 1 trading day(s))
  Allowing cache metadata to advance in this run.
  ⚠️  REMOVE ALLOW_METADATA_BOOTSTRAP override after this run completes.

======================================================================
✓ ALL VALIDATIONS PASSED
======================================================================
```

## Testing

Comprehensive tests in `test_validate_cache_metadata.py` verify:

1. Without override: Validation fails when cache is stale ✅
2. With override: Validation passes with warnings when cache is stale ✅
3. With override: Other validations (spy_max_date exists, tickers_total >= 50) still fail ✅

Run tests:
```bash
python test_validate_cache_metadata.py
```

## When to Use

Use this override **only** when:
- The cache metadata is stale and blocking commits
- The workflow cannot self-heal (bootstrap deadlock)
- You need **one successful run** to advance `spy_max_date`

**Do NOT use** for:
- Regular workflow runs (let validation enforce freshness normally)
- Bypassing legitimate validation failures
- Long-term workarounds (this is a temporary fix)

## Acceptance Criteria

- [x] Override allows stale cache to be committed once
- [x] Override logs clear warnings instead of failing
- [x] Override does NOT bypass other validations
- [x] Override is easy to enable/disable via workflow inputs
- [x] Override includes reminders to remove it after use
- [x] Tests confirm override behavior is correct
