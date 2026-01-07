# VIX Overlay Configuration Guide

## Overview

The VIX Overlay functionality is now **LIVE** in the WAVES Intelligence™ system. This guide explains how to configure, toggle, and verify the VIX overlay feature.

## Quick Status Check

The VIX overlay is enabled by default. To verify it's active:

### In the UI
- Navigate to any equity wave (e.g., "US MegaCap Core Wave")
- Look for the wave metrics section
- You'll see "VIX Overlay" status indicator:
  - **🟢 LIVE** = Active for this wave
  - **⚪ Off** = Disabled for this wave
  - **—** = Not applicable (crypto/income/safe waves)

### Via Python
```python
from config.vix_overlay_config import is_vix_overlay_live, get_vix_overlay_status

# Check global status
if is_vix_overlay_live():
    print("VIX Overlay is LIVE!")

# Get detailed status
status = get_vix_overlay_status()
print(f"VIX Overlay: {'LIVE' if status['is_live'] else 'OFF'}")
print(f"Resilient Mode: {status['resilient_mode']}")
print(f"Fallback VIX Level: {status['fallback_vix_level']}")
```

### Via Operator Toolbox
- Open the Operator Toolbox tab in the UI
- Check the "System Health" section
- VIX Overlay status will be displayed with configuration details

## Configuration Options

### Global Toggle

Enable or disable VIX overlay globally for all equity waves:

```python
from config.vix_overlay_config import enable_vix_overlay, disable_vix_overlay

# Enable VIX overlay (make it LIVE)
enable_vix_overlay()

# Disable VIX overlay
disable_vix_overlay()
```

### Per-Wave Configuration

Configure VIX overlay for specific waves:

```python
from config.vix_overlay_config import VIXOverlayConfig, set_vix_overlay_config

# Create custom configuration
config = VIXOverlayConfig(
    enabled=True,  # Global enable
    resilient_mode=True,  # Handle missing data gracefully
    fallback_vix_level=20.0,  # Neutral VIX when data missing
    wave_overrides={
        # Disable for specific waves
        "Bitcoin Wave": {"enabled": False},
        
        # Custom fallback for specific waves
        "AI & Cloud MegaCap Wave": {
            "enabled": True,
            "fallback_vix_level": 18.0,
            "log_diagnostics": True
        },
    }
)

# Apply configuration
set_vix_overlay_config(config)
```

### Configuration Parameters

#### VIXOverlayConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Global toggle for VIX overlay |
| `resilient_mode` | bool | True | Handle missing VIX data gracefully |
| `fallback_vix_level` | float | 20.0 | VIX level to use when data missing |
| `min_data_points` | int | 10 | Minimum days of data required |
| `wave_overrides` | dict | {} | Per-wave configuration overrides |
| `log_diagnostics` | bool | False | Enable detailed logging |

#### Wave Override Options

Each wave can have these overrides:
- `enabled`: bool - Enable/disable VIX overlay for this wave
- `fallback_vix_level`: float - Custom fallback VIX level
- `log_diagnostics`: bool - Enable diagnostics logging for this wave

## Resilience & Missing Data Handling

The VIX overlay is designed to handle missing data gracefully:

### When VIX Data is Missing

1. **With Resilient Mode Enabled (default)**:
   - System uses `fallback_vix_level` (default: 20.0)
   - Exposure and safe allocation are calculated based on fallback
   - Returns continue smoothly without breaking
   - Example: Missing VIX → uses 20.0 → exposure = 0.95 (slightly defensive)

2. **With Resilient Mode Disabled**:
   - System returns neutral values (exposure = 1.0, safe = 0.0)
   - No overlay adjustments are applied
   - Returns are calculated without VIX influence

### Testing Resilience

```python
from waves_engine import _vix_exposure_factor, _vix_safe_fraction
import numpy as np

# Test with missing VIX (NaN)
exposure = _vix_exposure_factor(np.nan, "Standard", "US MegaCap Core Wave")
safe = _vix_safe_fraction(np.nan, "Standard", "US MegaCap Core Wave")

print(f"With missing VIX:")
print(f"  Exposure: {exposure:.2%}")
print(f"  Safe Allocation: {safe:.2%}")
# Output: Exposure: 95.00%, Safe Allocation: 5.00% (using fallback VIX=20)
```

## Wave Applicability

VIX overlay only applies to **equity growth waves**. It is automatically disabled for:

- **Crypto Waves**: Bitcoin Wave, Crypto L1 Growth Wave, etc.
- **Income Waves**: Income Wave, Balanced Income Wave, etc.
- **Safe Waves**: SmartSafe Treasury Cash Wave, etc.

You can verify applicability:

```python
from waves_engine import is_vix_overlay_active_for_wave

# Check specific waves
print(f"US MegaCap Core Wave: {is_vix_overlay_active_for_wave('US MegaCap Core Wave')}")  # True
print(f"Bitcoin Wave: {is_vix_overlay_active_for_wave('Bitcoin Wave')}")  # False
print(f"Income Wave: {is_vix_overlay_active_for_wave('Income Wave')}")  # False
```

## UI Integration

### Wave Metrics Display

Individual wave views show VIX overlay status:

```
Beta: 1.05    VIX Regime: Normal    VIX Overlay: 🟢 LIVE    Exposure: 98.5%
```

Status indicators:
- **🟢 LIVE**: VIX overlay is active for this equity wave
- **⚪ Off**: VIX overlay is disabled for this wave
- **—**: Not applicable (non-equity wave)

### Operator Toolbox Integration

The Operator Toolbox displays system-wide VIX overlay status:

```
VIX Overlay Status: 🟢 LIVE and Active
  - Config Available: Yes
  - Resilient Mode: Enabled
  - Fallback VIX Level: 20.0
```

## Validation & Diagnostics

### Validate VIX Overlay is Working

Use the validation script to confirm overlay is active:

```bash
python validate_vix_overlay.py "US MegaCap Core Wave" "Standard" 365
```

This will output:
- Overall statistics (exposure range, safe fraction)
- VIX impact analysis (high vs low VIX periods)
- Regime impact analysis (risk-off vs risk-on)
- Recent diagnostic examples
- Validation result (material/minimal scaling)

### Get Detailed Diagnostics

```python
from waves_engine import get_vix_regime_diagnostics

# Get per-day diagnostics
diag = get_vix_regime_diagnostics("US MegaCap Core Wave", "Standard", 365)

# View columns
print(diag.columns)
# ['regime', 'vix', 'safe_fraction', 'exposure', 'vol_adjust', 
#  'vix_exposure', 'vix_gate', 'regime_gate']

# Analyze stress periods
high_vix = diag[diag['vix'] >= 25]
print(f"High VIX periods: {len(high_vix)} days")
print(f"Average exposure during high VIX: {high_vix['exposure'].mean():.2%}")
print(f"Average safe allocation during high VIX: {high_vix['safe_fraction'].mean():.2%}")
```

## Common Use Cases

### 1. Disable VIX Overlay for Specific Wave

```python
from config.vix_overlay_config import VIXOverlayConfig, set_vix_overlay_config

config = VIXOverlayConfig(
    enabled=True,  # Keep globally enabled
    wave_overrides={
        "Small Cap Growth Wave": {"enabled": False}  # Disable for this wave
    }
)
set_vix_overlay_config(config)
```

### 2. Use More Conservative Fallback

```python
config = VIXOverlayConfig(
    enabled=True,
    resilient_mode=True,
    fallback_vix_level=25.0,  # More defensive when data missing
)
set_vix_overlay_config(config)
```

### 3. Enable Diagnostics Logging

```python
config = VIXOverlayConfig(
    enabled=True,
    log_diagnostics=True,  # Enable for all waves
    wave_overrides={
        "AI & Cloud MegaCap Wave": {
            "log_diagnostics": True  # Extra logging for this wave
        }
    }
)
set_vix_overlay_config(config)
```

### 4. Temporary Disable for Testing

```python
from config.vix_overlay_config import disable_vix_overlay, enable_vix_overlay

# Disable temporarily
disable_vix_overlay()

# Run tests...
# ...

# Re-enable
enable_vix_overlay()
```

## Best Practices

1. **Keep Resilient Mode Enabled**: This ensures smooth operation even with data gaps
2. **Use Neutral Fallback (20.0)**: This provides reasonable behavior when VIX data is missing
3. **Monitor VIX Overlay Status**: Check the UI indicator to ensure it's active
4. **Review Diagnostics Periodically**: Use `validate_vix_overlay.py` to verify material impact
5. **Test Configuration Changes**: Always test in non-production before applying broadly

## Troubleshooting

### VIX Overlay Shows "N/A"
- **Cause**: Configuration module not loaded
- **Fix**: Ensure `config/vix_overlay_config.py` is in the Python path

### VIX Overlay Shows "⚪ Off" for Equity Wave
- **Cause**: Wave-specific override disabling it
- **Fix**: Check `wave_overrides` in configuration

### No Exposure Changes Observed
- **Cause**: Market conditions stable (no high/low VIX periods)
- **Fix**: This is normal - overlay only adjusts during volatility extremes

### Missing VIX Data Errors
- **Cause**: Resilient mode disabled
- **Fix**: Enable resilient mode in configuration

## API Reference

### Configuration Functions

- `get_vix_overlay_config()` - Get current configuration
- `set_vix_overlay_config(config)` - Set new configuration
- `enable_vix_overlay()` - Enable globally
- `disable_vix_overlay()` - Disable globally
- `is_vix_overlay_live()` - Check if LIVE
- `get_vix_overlay_status()` - Get detailed status

### Engine Functions

- `get_vix_overlay_status_for_wave(wave_name)` - Get status for specific wave
- `is_vix_overlay_active_for_wave(wave_name)` - Check if active for wave
- `get_vix_regime_diagnostics(wave_name, mode, days)` - Get detailed diagnostics

## Support

For questions or issues:
1. Check this documentation
2. Review `VIX_REGIME_OVERLAY_DOCUMENTATION.md` for technical details
3. Run `validate_vix_overlay.py` for diagnostics
4. Check operator toolbox for system status

## Version History

- **v1.0 (Current)**: Initial LIVE release with configuration system, resilience, and UI integration
