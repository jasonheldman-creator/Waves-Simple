#!/usr/bin/env python3
"""
Workflow Validation Script

This script runs comprehensive validations for the GitHub Actions workflow.
It validates the cache and determines whether to commit based on no-change logic.

Exit codes:
  0 - Success (cache valid, proceed based on outputs)
  1 - Failure (validation failed or stale + unchanged)
"""

import sys
import os
import subprocess
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

from helpers.cache_validation import (
    validate_trading_day_freshness,
    validate_required_symbols,
    validate_cache_integrity,
)

CACHE_PATH = 'data/cache/prices_cache.parquet'


def set_workflow_output(name, value):
    """Set a GitHub Actions workflow output."""
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f'{name}={value}\n')


def main():
    """Run validations and set workflow outputs."""
    print('\n' + '=' * 70)
    print('CACHE VALIDATION PIPELINE')
    print('=' * 70)
    print(f'Cache path: {CACHE_PATH}')
    print(f'Today: {datetime.now().date()}')
    print()
    
    # Validation 1: Cache integrity
    print('--- Validation 1: Cache Integrity ---')
    integrity_result = validate_cache_integrity(CACHE_PATH)
    if not integrity_result['valid']:
        print(f'✗ FAIL: {integrity_result["error"]}')
        set_workflow_output('should_commit', 'false')
        set_workflow_output('workflow_status', 'failure')
        set_workflow_output('message', f'Cache integrity failed: {integrity_result["error"]}')
        return 1
    
    print(f'  File exists: {integrity_result["file_exists"]}')
    print(f'  File size: {integrity_result["file_size_bytes"]:,} bytes')
    print(f'  Symbol count: {integrity_result["symbol_count"]}')
    print()
    
    # Validation 2: Required symbols
    print('--- Validation 2: Required Symbols ---')
    symbols_result = validate_required_symbols(CACHE_PATH)
    if not symbols_result['valid']:
        print(f'✗ FAIL: {symbols_result["error"]}')
        set_workflow_output('should_commit', 'false')
        set_workflow_output('workflow_status', 'failure')
        set_workflow_output('message', f'Required symbols missing: {symbols_result["error"]}')
        return 1
    
    print(f'  Total symbols in cache: {len(symbols_result["symbols_in_cache"])}')
    print(f'  VIX group present: {symbols_result["present_vix_group"]}')
    print(f'  T-bill group present: {symbols_result["present_tbill_group"]}')
    print()
    
    # Validation 3: Trading-day freshness
    print('--- Validation 3: Trading-Day Freshness ---')
    freshness_result = validate_trading_day_freshness(CACHE_PATH, max_market_feed_gap_days=5)
    
    # Store freshness result for no-change logic
    cache_is_fresh = freshness_result['valid']
    
    if not cache_is_fresh:
        print(f'⚠️  WARNING: {freshness_result["error"]}')
        if freshness_result["today"]:
            print(f'  Today: {freshness_result["today"].date()}')
        if freshness_result["last_trading_day"]:
            print(f'  Last trading day: {freshness_result["last_trading_day"].date()}')
        if freshness_result["cache_max_date"]:
            print(f'  Cache max date: {freshness_result["cache_max_date"].date()}')
        print(f'  Delta days: {freshness_result["delta_days"]}')
        print(f'  Market feed gap: {freshness_result["market_feed_gap_days"]}')
    else:
        print(f'  Cache is fresh: {cache_is_fresh}')
    print()
    
    # Check for git changes
    print('--- Git Status Check ---')
    try:
        git_status = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
        has_changes = bool(git_status)
        print(f'  Has uncommitted changes: {has_changes}')
        
        if git_status:
            print(f'  Changed files:')
            for line in git_status.split('\n')[:10]:
                print(f'    {line}')
        
        # Show git diff stats
        try:
            git_diff = subprocess.check_output(['git', 'diff', '--stat'], text=True).strip()
            if git_diff:
                print(f'  Git diff stats:')
                for line in git_diff.split('\n')[:10]:
                    print(f'    {line}')
        except Exception:
            pass
    except Exception as e:
        print(f'  Error checking git status: {e}')
        has_changes = False
    print()
    
    # No-change logic
    print('--- No-Change Logic ---')
    print(f'  Cache fresh: {cache_is_fresh}')
    print(f'  Has changes: {has_changes}')
    
    if cache_is_fresh and not has_changes:
        # Fresh + unchanged → SUCCESS (no commit)
        print('✓ PASS: Fresh but unchanged — no commit needed')
        set_workflow_output('should_commit', 'false')
        set_workflow_output('workflow_status', 'success')
        set_workflow_output('message', 'Fresh but unchanged — no commit needed')
        return 0
    elif cache_is_fresh and has_changes:
        # Fresh + changed → SUCCESS (commit)
        print('✓ PASS: Fresh and changed — will commit updates')
        set_workflow_output('should_commit', 'true')
        set_workflow_output('workflow_status', 'success')
        set_workflow_output('message', 'Fresh and changed — committing updates')
        return 0
    elif not cache_is_fresh and not has_changes:
        # Stale + unchanged → FAIL
        print('✗ FAIL: Stale + unchanged')
        set_workflow_output('should_commit', 'false')
        set_workflow_output('workflow_status', 'failure')
        set_workflow_output('message', 'Stale + unchanged')
        return 1
    else:
        # Stale + changed → SUCCESS (commit stale data that was updated)
        print('✓ PASS: Stale but changed — will commit updates')
        set_workflow_output('should_commit', 'true')
        set_workflow_output('workflow_status', 'success')
        set_workflow_output('message', 'Stale but changed — committing updates')
        return 0


if __name__ == '__main__':
    sys.exit(main())
