#!/usr/bin/env python3
"""
Script to add strategy_stack column to wave_registry.csv
This configures equity waves with the default strategy pipeline.
"""

import pandas as pd
import json

# Load the wave registry
registry_path = 'data/wave_registry.csv'
registry = pd.read_csv(registry_path)

print(f"Loaded {len(registry)} waves from {registry_path}")
print(f"Existing columns: {list(registry.columns)}")

# Define the default strategy stack for different categories
def get_default_strategy_stack(category):
    """
    Get the default strategy stack for a wave category.
    
    Args:
        category: Wave category (equity_growth, crypto_growth, equity_income, etc.)
    
    Returns:
        List of strategy names in execution order
    """
    if category in ['equity_growth']:
        # Standard equity growth waves: momentum, trend, and vix_safesmart
        return ['momentum', 'trend', 'vix_safesmart']
    elif category in ['equity_income']:
        # Income waves: simpler stack, no momentum
        return ['trend', 'vix_safesmart']
    elif category in ['crypto_growth', 'crypto_income']:
        # Crypto waves: no VIX overlay (use crypto-specific overlays)
        return ['momentum', 'trend']
    elif category == 'special':
        # Special assets like gold: minimal overlays
        return ['vix_safesmart']
    else:
        # Default: full stack
        return ['momentum', 'trend', 'vix_safesmart']

# Add strategy_stack column if it doesn't exist
if 'strategy_stack' not in registry.columns:
    print("\nAdding strategy_stack column...")
    
    # Initialize with empty lists
    registry['strategy_stack'] = None
    
    # Populate based on category
    for idx, row in registry.iterrows():
        category = row['category']
        wave_id = row['wave_id']
        
        # Get default strategy stack for this category
        stack = get_default_strategy_stack(category)
        
        # Optional: Add volatility targeting for select high-conviction growth waves
        if wave_id in ['ai_cloud_megacap_wave', 'next_gen_compute_semis_wave', 'quantum_computing_wave']:
            # These waves get volatility targeting as an optional overlay
            stack_with_vol = stack.copy()
            # Insert vol_targeting before vix_safesmart (if present)
            if 'vix_safesmart' in stack_with_vol:
                vix_idx = stack_with_vol.index('vix_safesmart')
                stack_with_vol.insert(vix_idx, 'vol_targeting')
            else:
                stack_with_vol.append('vol_targeting')
            stack = stack_with_vol
        
        # Convert to JSON string for CSV storage
        registry.at[idx, 'strategy_stack'] = json.dumps(stack)
        
        print(f"  {wave_id:40s} ({category:15s}): {stack}")
    
    print(f"\n✓ Added strategy_stack column to {len(registry)} waves")
else:
    print("\nstrategy_stack column already exists!")

# Save the updated registry
registry.to_csv(registry_path, index=False)
print(f"\n✓ Saved updated registry to {registry_path}")

# Validation: reload and check
registry_check = pd.read_csv(registry_path)
print(f"\n✓ Validation: Reloaded registry has {len(registry_check)} rows and 'strategy_stack' column exists: {'strategy_stack' in registry_check.columns}")

# Show sample
print("\nSample strategy stacks:")
for idx, row in registry_check.head(5).iterrows():
    try:
        stack = json.loads(row['strategy_stack'])
        print(f"  {row['wave_id']:40s}: {stack}")
    except:
        print(f"  {row['wave_id']:40s}: {row['strategy_stack']}")
