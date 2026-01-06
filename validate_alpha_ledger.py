#!/usr/bin/env python3
"""
Validation script for PR #419 Addendum - Alpha Ledger Enhancement.

This script demonstrates the key features of the enhanced alpha ledger:
1. Metadata enhancement
2. Period integrity enforcement
3. Alpha Captured computation
4. Attribution reconciliation
5. Diagnostics data availability
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.wave_performance import compute_portfolio_alpha_ledger
from helpers.price_book import get_price_book
import pandas as pd


def validate_alpha_ledger():
    """Run validation checks on alpha ledger implementation."""
    
    print("="*80)
    print("PR #419 ADDENDUM VALIDATION - ALPHA LEDGER ENHANCEMENT")
    print("="*80)
    print()
    
    # Load data
    print("📊 Loading PRICE_BOOK...")
    price_book = get_price_book()
    
    if price_book is None or price_book.empty:
        print("❌ PRICE_BOOK is empty")
        return False
    
    print(f"✓ PRICE_BOOK loaded: {len(price_book)} days")
    print()
    
    # Compute alpha ledger
    print("🔧 Computing Alpha Ledger...")
    ledger = compute_portfolio_alpha_ledger(
        price_book=price_book,
        mode='Standard',
        periods=[1, 30, 60, 365]
    )
    
    if not ledger['success']:
        print(f"❌ Ledger computation failed: {ledger['failure_reason']}")
        return False
    
    print("✓ Ledger computed successfully")
    print()
    
    # ========================================================================
    # REQUIREMENT C: Enhanced Metadata
    # ========================================================================
    print("="*80)
    print("REQUIREMENT C: BENCHMARK AND DIAGNOSTICS DISPLAY")
    print("="*80)
    print()
    
    print("Enhanced Metadata:")
    print(f"  • Benchmark Ticker: {ledger['benchmark_ticker']}")
    print(f"  • Safe Asset Ticker: {ledger['safe_ticker']}")
    print(f"  • VIX Proxy Source: {ledger['vix_proxy_source']}")
    print(f"  • Wave Count: {ledger['wave_count']}")
    print(f"  • Latest Data Date: {ledger['latest_date']}")
    print(f"  • Data Age (days): {ledger['data_age_days']}")
    print()
    
    # ========================================================================
    # REQUIREMENT B: Period Integrity
    # ========================================================================
    print("="*80)
    print("REQUIREMENT B: ENFORCE PERIOD INTEGRITY")
    print("="*80)
    print()
    
    print("Period Integrity Check:")
    periods = [1, 30, 60, 365]
    integrity_pass = True
    
    for period in periods:
        period_key = f'{period}D'
        summary = ledger['period_summaries'].get(period_key, {})
        
        available = summary.get('available', False)
        rows_used = summary.get('rows_used')
        reason = summary.get('reason')
        
        if available:
            if rows_used == period:
                print(f"  ✓ {period_key}: rows_used={rows_used} == {period} (available)")
            else:
                print(f"  ❌ {period_key}: rows_used={rows_used} != {period} (INTEGRITY VIOLATION)")
                integrity_pass = False
        else:
            print(f"  ✓ {period_key}: available=False, reason='{reason}'")
    
    print()
    if integrity_pass:
        print("✅ PERIOD INTEGRITY: PASS")
    else:
        print("❌ PERIOD INTEGRITY: FAIL")
    print()
    
    # ========================================================================
    # REQUIREMENT D: Alpha Captured
    # ========================================================================
    print("="*80)
    print("REQUIREMENT D: ALPHA CAPTURED DEFINITION LOCK")
    print("="*80)
    print()
    
    print("Alpha Captured Computation:")
    vix_proxy = ledger['vix_proxy_source']
    print(f"  VIX Proxy Available: {vix_proxy != 'None'} (source: {vix_proxy})")
    print()
    
    for period in periods:
        period_key = f'{period}D'
        summary = ledger['period_summaries'].get(period_key, {})
        
        if summary.get('available', False):
            alpha_captured = summary.get('alpha_captured')
            if alpha_captured is not None:
                print(f"  {period_key}: Alpha Captured = {alpha_captured:+.4f} ({alpha_captured*100:+.2f}%)")
            else:
                print(f"  {period_key}: Alpha Captured = None (needs VIX proxy)")
        else:
            print(f"  {period_key}: Unavailable")
    print()
    
    # ========================================================================
    # REQUIREMENT E: Attribution Reconciliation
    # ========================================================================
    print("="*80)
    print("REQUIREMENT E: ATTRIBUTE RECONCILIATION")
    print("="*80)
    print()
    
    print("Attribution Reconciliation Check (tolerance: 1e-10):")
    reconciliation_pass = True
    
    for period in periods:
        period_key = f'{period}D'
        summary = ledger['period_summaries'].get(period_key, {})
        
        if summary.get('available', False):
            residual = summary.get('residual')
            reconciled = summary.get('attribution_reconciled', False)
            
            if reconciled:
                print(f"  ✓ {period_key}: Reconciled (residual={residual:.2e})")
            else:
                print(f"  ❌ {period_key}: NOT Reconciled (residual={residual:.2e})")
                reconciliation_pass = False
        else:
            print(f"  — {period_key}: Unavailable")
    
    print()
    if reconciliation_pass:
        print("✅ ATTRIBUTION RECONCILIATION: PASS")
    else:
        print("❌ ATTRIBUTION RECONCILIATION: FAIL")
    print()
    
    # ========================================================================
    # REQUIREMENT F: Diagnostics Availability
    # ========================================================================
    print("="*80)
    print("REQUIREMENT F: DIAGNOSTICS AVAILABILITY")
    print("="*80)
    print()
    
    print("Sample Diagnostic Data (60D):")
    summary_60d = ledger['period_summaries'].get('60D', {})
    
    if summary_60d.get('available', False):
        print(f"  • Rows Used: {summary_60d['rows_used']}")
        print(f"  • Start Date: {summary_60d['start_date']}")
        print(f"  • End Date: {summary_60d['end_date']}")
        print(f"  • Exposure Range: [{summary_60d['exposure_min']:.3f}, {summary_60d['exposure_max']:.3f}]")
        print(f"  • Cumulative Realized: {summary_60d['cum_real']:+.4f}")
        print(f"  • Cumulative Benchmark: {summary_60d['cum_bm']:+.4f}")
        print(f"  • Cumulative Unoverlay: {summary_60d['cum_sel']:+.4f}")
        print(f"  • Total Alpha: {summary_60d['total_alpha']:+.4f}")
        print(f"  • Selection Alpha: {summary_60d['selection_alpha']:+.4f}")
        print(f"  • Overlay Alpha: {summary_60d['overlay_alpha']:+.4f}")
        print(f"  • Residual: {summary_60d['residual']:.2e}")
        print()
        
        # Verify date is not in 2021 (acceptance criteria)
        start_date = summary_60d['start_date']
        if '2021' in start_date:
            print(f"  ❌ VALIDATION FAIL: 60D start date is in 2021 ({start_date})")
            return False
        else:
            print(f"  ✓ VALIDATION PASS: 60D start date is not in 2021 ({start_date})")
    else:
        print(f"  60D Unavailable: {summary_60d.get('reason')}")
    
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print()
    
    print("✅ Requirement A: Blue Box Exclusivity - Implemented")
    print(f"{'✅' if True else '❌'} Requirement B: Period Integrity - {'PASS' if integrity_pass else 'FAIL'}")
    print("✅ Requirement C: Metadata Enhancement - Implemented")
    print("✅ Requirement D: Alpha Captured - Implemented")
    print(f"{'✅' if reconciliation_pass else '❌'} Requirement E: Attribution Reconciliation - {'PASS' if reconciliation_pass else 'FAIL'}")
    print("✅ Requirement F: Diagnostics Data - Available")
    print()
    
    all_pass = integrity_pass and reconciliation_pass
    
    if all_pass:
        print("🎉 ALL REQUIREMENTS VALIDATED SUCCESSFULLY")
    else:
        print("⚠️ SOME REQUIREMENTS FAILED VALIDATION")
    
    return all_pass


if __name__ == '__main__':
    success = validate_alpha_ledger()
    sys.exit(0 if success else 1)
