# Auto-Refresh Feature Documentation

## Overview

The WAVES Streamlit Console includes an institutional-grade Auto-Refresh feature that provides live updates to analytics, overlays, attribution, diagnostics, and summary metrics without requiring manual page refreshes.

## Key Features

### 1. **Default Behavior**
- Auto-Refresh is **enabled by default** when the app loads
- Default refresh interval: **60 seconds**
- Users can toggle it OFF manually via Mission Control sidebar

### 2. **Configurable Refresh Intervals**
Users can select from pre-configured safe intervals:
- **30 seconds** - Fast refresh for active monitoring
- **60 seconds** (default) - Balanced refresh for general use
- **120 seconds** - Slower refresh to reduce resource usage

### 3. **Live Status Indicators**

#### Mission Control Display
The Mission Control panel shows:
- **Auto-Refresh Status**: `🟢 LIVE (60s)` or `⏸️ PAUSED`
- Current refresh interval in seconds
- Real-time status updates

#### Sidebar Control Panel
The sidebar provides detailed controls and status:
- **Status**: Shows `🟢 Status: LIVE (60s)` when active or `⏸️ Status: PAUSED` when disabled
- **Toggle Switch**: Enable/disable Auto-Refresh
- **Interval Selector**: Choose refresh interval (30/60/120 seconds)
- **Timestamps**:
  - Last refresh: Shows when the last refresh occurred
  - Last successful refresh: Shows when the last error-free refresh completed
  - Data status: Visual indicator (🟢 Fresh / 🟡 Recent / 🔴 Stale)

### 4. **Fail-Safe Error Handling**

The Auto-Refresh feature includes robust error handling:

#### Automatic Error Detection
- Monitors refresh cycles for exceptions
- Tracks consecutive error count
- Maximum of 3 consecutive errors before auto-pause

#### Auto-Pause on Errors
When errors occur:
- Auto-Refresh automatically pauses after 3 consecutive failures
- Toggle is disabled and shows `⏸️ PAUSED` status
- Sidebar displays warning: `⚠️ Auto-Refresh paused due to errors`
- Error count is shown for transparency

#### Recovery Mechanism
Users can resume Auto-Refresh after errors:
1. Click **Resume Auto-Refresh** button in sidebar
2. Error count resets to zero
3. Auto-Refresh re-enables if toggle is ON
4. Console remains fully usable in manual mode

### 5. **Safe Refresh Execution**

The refresh mechanism is designed for stability:

#### What Gets Refreshed
- Live analytics and metrics
- Overlay diagnostics (VIX, Alpha Attribution)
- Summary statistics
- Mission Control data
- Timestamp displays

#### What Doesn't Get Refreshed
- Heavy backtest computations (cached)
- Historical data processing (cached)
- Wave universe rebuilds (manual trigger only)
- API calls (respects caching rules)

#### Performance Optimizations
- Leverages Streamlit's built-in caching (`@st.cache_data`)
- Only refreshes UI layer, not data layer
- No redundant API calls or data reprocessing
- Prevents infinite refresh loops

### 6. **Configuration**

#### Application-Level Settings
Located in `app.py` under `AUTO_REFRESH_CONFIG`:

```python
AUTO_REFRESH_CONFIG = {
    "default_enabled": True,  # Default state: ON when app loads
    "default_interval_seconds": 60,  # Default interval: 60 seconds
    "allowed_intervals": [30, 60, 120],  # Allowed interval choices
    "allow_custom_interval": False,  # Disable custom intervals for safety
    "pause_on_error": True,  # Auto-pause on exceptions
    "max_consecutive_errors": 3,  # Max errors before forcing pause
}
```

#### Customization
To change default settings, edit `AUTO_REFRESH_CONFIG` in `app.py`:

1. **Change Default Interval**:
   ```python
   "default_interval_seconds": 120,  # Change to 120 seconds
   ```

2. **Disable Auto-Refresh by Default**:
   ```python
   "default_enabled": False,  # Start with Auto-Refresh OFF
   ```

3. **Add Custom Intervals**:
   ```python
   "allowed_intervals": [30, 60, 90, 120, 180],  # Add 90 and 180 seconds
   ```

4. **Adjust Error Threshold**:
   ```python
   "max_consecutive_errors": 5,  # Allow 5 errors before pausing
   ```

## User Controls

### Enabling/Disabling Auto-Refresh
1. Navigate to sidebar **Mission Control** section
2. Locate **Auto-Refresh Control** panel
3. Check/uncheck **Enable Auto-Refresh** toggle
4. Changes take effect immediately

### Changing Refresh Interval
1. Navigate to sidebar **Mission Control** section
2. Use **Refresh Interval** dropdown selector
3. Choose from: 30 seconds, 60 seconds, or 120 seconds
4. New interval applies on next refresh cycle

### Monitoring Refresh Status
- **Mission Control Panel**: Quick status view in main dashboard
- **Sidebar**: Detailed status with timestamps and error information
- **Data Status Indicator**: 
  - 🟢 Fresh (< 2 minutes old)
  - 🟡 Recent (2-5 minutes old)
  - 🔴 Stale (> 5 minutes old)

### Recovering from Errors
If Auto-Refresh pauses due to errors:
1. Check sidebar for error count
2. Review console logs for specific error details
3. Click **Resume Auto-Refresh** button
4. Monitor for successful refreshes

### Manual Refresh Alternative
If Auto-Refresh is disabled or paused:
- Use **Force Reload Wave Universe** button (sidebar)
- Use **Force Reload Data (Clear Cache)** button (sidebar)
- Both trigger immediate manual refresh

## Technical Details

### Dependencies
- **streamlit-autorefresh**: Primary refresh library
- Requires: `streamlit-autorefresh>=1.0.1`
- Install: `pip install streamlit-autorefresh`

### Session State Variables
The feature uses these session state keys:
- `auto_refresh_enabled`: Boolean, toggle state
- `auto_refresh_interval`: Integer, selected interval in seconds
- `auto_refresh_paused_by_error`: Boolean, error pause state
- `auto_refresh_error_count`: Integer, consecutive error counter
- `last_refresh_time`: DateTime, last refresh timestamp
- `last_successful_refresh`: DateTime, last successful refresh timestamp

### Error Handling Flow
```
Refresh Attempt
    ↓
Try Execute Auto-Refresh
    ↓
Success? → Reset error_count → Update last_successful_refresh
    ↓
Failure? → Increment error_count
    ↓
error_count >= max_consecutive_errors?
    ↓
Yes → Pause Auto-Refresh → Disable Toggle → Show Warning
    ↓
No → Continue with next refresh cycle
```

## Best Practices

### For Users
1. **Start with default 60-second interval** for most use cases
2. **Use 30-second interval** only for active trading/monitoring sessions
3. **Use 120-second interval** to reduce resource usage during analysis
4. **Monitor error warnings** and resume after addressing issues
5. **Use manual refresh** for immediate data updates when needed

### For Administrators
1. **Keep default interval at 60 seconds** for balanced performance
2. **Monitor error logs** for recurring refresh issues
3. **Adjust max_consecutive_errors** based on environment stability
4. **Test configuration changes** in dev environment first
5. **Document any custom interval additions** for users

## Troubleshooting

### Auto-Refresh Not Working
**Symptom**: Toggle enabled but no refresh occurs
**Solutions**:
- Check if `streamlit-autorefresh` is installed: `pip list | grep streamlit-autorefresh`
- Verify sidebar shows "Status: LIVE" not "not supported"
- Check browser console for JavaScript errors
- Try clearing cache with Force Reload button

### Frequent Auto-Pause
**Symptom**: Auto-Refresh pauses often with errors
**Solutions**:
- Review error count in sidebar
- Check console logs for specific error messages
- Increase `max_consecutive_errors` in config if transient errors
- Verify network connectivity for data sources
- Check if data files are accessible

### Performance Issues
**Symptom**: App slows down with Auto-Refresh enabled
**Solutions**:
- Increase refresh interval to 120 seconds
- Disable Auto-Refresh during heavy analysis tasks
- Clear cache with Force Reload Data button
- Check browser memory usage
- Verify caching is working properly

### Stale Data Despite Refresh
**Symptom**: Data status shows "Stale" even with Auto-Refresh ON
**Solutions**:
- Check last_successful_refresh timestamp in sidebar
- Verify data sources are updating
- Use Force Reload Data to clear all caches
- Check if error pause is active
- Review data file modification times

## Version History

### v1.0.0 (Current)
- Initial implementation with configurable intervals
- Fail-safe error handling with auto-pause
- LIVE/PAUSED status indicators
- Timestamp tracking for refresh monitoring
- Default ON with 60-second interval
- Support for 30/60/120-second intervals

## Support

For issues or questions:
1. Check this documentation first
2. Review console logs for error details
3. Test with manual refresh to isolate issues
4. Contact development team with error logs and reproduction steps
