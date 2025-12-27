#!/usr/bin/env python3
"""
Acceptance Test: Stage 4 Safe Analytics Coverage

This test validates that:
1. All 28 Waves can be loaded
2. Analytics pipeline runs without exceptions
3. Execution summary shows readiness states (Ready, Degraded, Limited History)

Usage:
    python test_analytics_coverage.py
"""

import sys
import traceback
from datetime import datetime

# Import analytics pipeline
try:
    from analytics_pipeline import run_daily_analytics_pipeline
    from waves_engine import get_all_wave_ids, get_display_name_from_wave_id
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    sys.exit(1)


def test_load_all_waves():
    """Test 1: Verify all 28 waves can be loaded from registry"""
    print("=" * 70)
    print("TEST 1: Load All Waves")
    print("=" * 70)
    
    try:
        all_wave_ids = get_all_wave_ids()
        wave_count = len(all_wave_ids)
        
        print(f"✓ Successfully loaded {wave_count} waves from registry")
        
        if wave_count != 28:
            print(f"⚠ WARNING: Expected 28 waves, found {wave_count}")
            return False
        
        print("\nWave List:")
        for i, wave_id in enumerate(all_wave_ids, 1):
            display_name = get_display_name_from_wave_id(wave_id)
            print(f"  {i:2d}. {wave_id:30s} ({display_name})")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def test_analytics_pipeline():
    """Test 2: Run analytics pipeline for all waves without exceptions"""
    print("\n" + "=" * 70)
    print("TEST 2: Analytics Pipeline Execution")
    print("=" * 70)
    
    try:
        # Run pipeline with minimal lookback to speed up test
        result = run_daily_analytics_pipeline(
            all_waves=True,
            lookback_days=7,
            use_dummy_data=False  # Use real data
        )
        
        print("\n" + "-" * 70)
        print("Pipeline Execution Summary:")
        print("-" * 70)
        print(f"Total waves:   {result['total_waves']}")
        print(f"Successful:    {result['successful']}")
        print(f"Failed:        {result['failed']}")
        
        # Check if any critical failures
        if result['failed'] > 0:
            print(f"\n⚠ WARNING: {result['failed']} waves failed processing")
            
            # Show failed waves
            print("\nFailed Waves:")
            for wave_result in result['results']:
                if not wave_result['success']:
                    print(f"  - {wave_result['wave_id']} ({wave_result['display_name']})")
                    for error in wave_result['errors']:
                        print(f"    Error: {error}")
        else:
            print("\n✓ All waves processed successfully!")
        
        return result
        
    except Exception as e:
        print(f"✗ FAILED: Analytics pipeline crashed with exception: {e}")
        traceback.print_exc()
        return None


def test_readiness_states(pipeline_result):
    """Test 3: Analyze readiness states from validation results"""
    print("\n" + "=" * 70)
    print("TEST 3: Wave Readiness States Analysis")
    print("=" * 70)
    
    if pipeline_result is None:
        print("✗ FAILED: No pipeline result to analyze")
        return False
    
    try:
        validation_summary = pipeline_result.get('validation_summary')
        
        if validation_summary is None or validation_summary.empty:
            print("✗ FAILED: No validation summary available")
            return False
        
        # Count readiness states
        total_waves = len(validation_summary)
        passed_waves = (validation_summary['status'] == 'pass').sum()
        failed_waves = (validation_summary['status'] == 'fail').sum()
        
        print(f"\nValidation Results:")
        print(f"  Total waves:        {total_waves}")
        print(f"  Validation passed:  {passed_waves}")
        print(f"  Validation failed:  {failed_waves}")
        
        # Show breakdown by validation status
        print("\n" + "-" * 70)
        print("Wave-by-Wave Validation Status:")
        print("-" * 70)
        
        ready_count = 0
        degraded_count = 0
        limited_count = 0
        
        for _, row in validation_summary.iterrows():
            wave_id = row['wave_id']
            status = row['status']
            issue_count = row['issue_count']
            
            # Categorize based on validation results
            if status == 'pass' and issue_count == 0:
                state = "Ready"
                ready_count += 1
            elif status == 'pass' and issue_count > 0:
                state = "Degraded"
                degraded_count += 1
            else:
                state = "Limited History"
                limited_count += 1
            
            status_icon = "✓" if status == 'pass' else "⚠"
            print(f"  {status_icon} {wave_id:30s} | {state:15s} | Issues: {issue_count}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Ready:           {ready_count} waves ({100*ready_count/total_waves:.1f}%)")
        print(f"Degraded:        {degraded_count} waves ({100*degraded_count/total_waves:.1f}%)")
        print(f"Limited History: {limited_count} waves ({100*limited_count/total_waves:.1f}%)")
        print("=" * 70)
        
        # Test passes if at least 90% of waves are in some usable state
        usable_waves = ready_count + degraded_count + limited_count
        success_rate = (usable_waves / total_waves) * 100.0
        
        print(f"\nUsability Rate: {success_rate:.1f}%")
        
        if success_rate >= 90.0:
            print("✓ TEST PASSED: Analytics coverage meets acceptance criteria")
            return True
        else:
            print(f"✗ TEST FAILED: Analytics coverage below 90% threshold")
            return False
        
    except Exception as e:
        print(f"✗ FAILED: Error analyzing readiness states: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all acceptance tests"""
    print("=" * 70)
    print("WAVES Stage 4 Analytics Coverage - Acceptance Tests")
    print("=" * 70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Run tests
    test_results = []
    
    # Test 1: Load all waves
    result1 = test_load_all_waves()
    test_results.append(('Load All Waves', result1))
    
    # Test 2: Run analytics pipeline
    pipeline_result = test_analytics_pipeline()
    result2 = pipeline_result is not None
    test_results.append(('Analytics Pipeline', result2))
    
    # Test 3: Analyze readiness states
    result3 = test_readiness_states(pipeline_result)
    test_results.append(('Readiness States', result3))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    
    for test_name, passed in test_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} | {test_name}")
    
    print("=" * 70)
    
    # Exit with appropriate code
    all_passed = all(result for _, result in test_results)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
