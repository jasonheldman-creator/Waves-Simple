#!/usr/bin/env python3
"""
Demo script for strategy-aware return pipeline.
Demonstrates the full strategy stack in action.
"""

import sys
try:
    from helpers import strategy_return_pipeline
    from helpers import wave_registry
    print("✓ Successfully imported modules")
except Exception as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

def demo_strategy_pipeline():
    """Demonstrate strategy pipeline with a sample wave."""
    print("\n" + "="*70)
    print("STRATEGY RETURN PIPELINE DEMO")
    print("="*70 + "\n")
    
    # Test with S&P 500 Wave
    wave_id = 'sp500_wave'
    
    print(f"Computing returns for {wave_id}...")
    result = strategy_return_pipeline.compute_wave_returns_with_strategy(
        wave_id=wave_id,
        apply_strategy_stack=True
    )
    
    if not result['success']:
        print(f"✗ Failed: {result['failure_reason']}")
        return
    
    print(f"\n✓ Success!")
    print(f"  Wave ID: {result['wave_id']}")
    print(f"  Strategy Stack: {result['strategy_stack']}")
    
    if result['attribution']:
        print(f"\n  Overlays Applied: {result['attribution']['overlays_applied']}")
        print(f"\n  Alpha Attribution:")
        for component, value in result['attribution']['component_alphas'].items():
            print(f"    {component:30s}: {value:+.6f}")
        
        print(f"\n  Total Alpha: {result['attribution']['total_alpha']:+.6f}")
    
    # Test with wave that has vol_targeting
    wave_id2 = 'ai_cloud_megacap_wave'
    print(f"\n{'-'*70}")
    print(f"Computing returns for {wave_id2}...")
    
    result2 = strategy_return_pipeline.compute_wave_returns_with_strategy(
        wave_id=wave_id2,
        apply_strategy_stack=True
    )
    
    if result2['success']:
        print(f"\n✓ Success!")
        print(f"  Wave ID: {result2['wave_id']}")
        print(f"  Strategy Stack: {result2['strategy_stack']}")
        print(f"  Overlays Applied: {result2['attribution']['overlays_applied']}")
    else:
        print(f"✗ Failed: {result2['failure_reason']}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")

if __name__ == '__main__':
    demo_strategy_pipeline()
