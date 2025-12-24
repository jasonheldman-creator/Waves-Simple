# Auto-Refresh Feature - Quick Start Guide

## What is Auto-Refresh?

Auto-Refresh is an institutional-grade feature that automatically refreshes the WAVES Streamlit Console at configurable intervals, ensuring you always see the latest analytics, overlays, and metrics without manual page reloads.

## Key Features at a Glance

✅ **On by Default** - Auto-Refresh starts automatically when you load the app  
✅ **60-Second Interval** - Balanced default for institutional use  
✅ **Configurable** - Choose 30, 60, or 120-second intervals  
✅ **LIVE/PAUSED Indicators** - Clear status display in Mission Control  
✅ **Error Recovery** - Auto-pauses on errors, manual resume available  
✅ **Safe Execution** - Respects caching, no performance degradation  

## Quick Start

### Using Auto-Refresh

1. **Start the App**
   ```bash
   streamlit run app.py
   ```

2. **Check Status**
   - Look at Mission Control panel (top of page)
   - You'll see: `🟢 LIVE (60s)` or `⏸️ PAUSED`

3. **Change Interval** (Optional)
   - Open sidebar (click arrow if collapsed)
   - Find "Auto-Refresh Control" section
   - Select interval: 30s, 60s, or 120s

4. **Disable/Enable**
   - Toggle the "Enable Auto-Refresh" checkbox in sidebar

## Where to Find Controls

### Mission Control Panel (Top of Page)
```
┌─────────────────────────────────────────┐
│ Auto-Refresh: 🟢 LIVE (60s)            │
└─────────────────────────────────────────┘
```

### Sidebar Control Panel
```
┌─────────────────────────────────────────┐
│ 🔄 Auto-Refresh Control                │
├─────────────────────────────────────────┤
│ ☑ Enable Auto-Refresh                  │
│                                         │
│ Refresh Interval: ▼ 60 seconds         │
│                                         │
│ 🟢 Status: LIVE (60s)                  │
│                                         │
│ Refresh Timestamps                      │
│ Last refresh: 2025-12-24 13:05:34      │
│ Last successful: 2025-12-24 13:05:34   │
│ Data status: 🟢 Fresh (15s ago)        │
└─────────────────────────────────────────┘
```

## Common Scenarios

### Scenario 1: Active Monitoring
**Use Case**: You're actively monitoring the portfolio  
**Recommended**: 30-second interval  
**How**: Select "30 seconds" in the Refresh Interval dropdown

### Scenario 2: General Dashboard View
**Use Case**: You have the dashboard open while working  
**Recommended**: 60-second interval (default)  
**How**: No action needed, this is the default

### Scenario 3: Background Monitoring
**Use Case**: Dashboard is on a secondary monitor  
**Recommended**: 120-second interval  
**How**: Select "120 seconds" in the Refresh Interval dropdown

### Scenario 4: Manual Control Only
**Use Case**: You want to control refreshes manually  
**Recommended**: Disable Auto-Refresh  
**How**: Uncheck "Enable Auto-Refresh" in sidebar

## Error Handling

### What Happens When Errors Occur?

1. **First Error**: Counted, refresh continues
2. **Second Error**: Counted, refresh continues
3. **Third Error**: Auto-Refresh pauses automatically

### How to Recover

When you see: `⚠️ Auto-Refresh paused due to errors`

1. Check what caused the error (network, data issue, etc.)
2. Fix the underlying issue if possible
3. Click **Resume Auto-Refresh** button in sidebar
4. Auto-Refresh will re-enable if toggle is ON

### Manual Refresh Options

Even with Auto-Refresh disabled, you can manually refresh:
- **Force Reload Wave Universe** - Reloads wave data
- **Force Reload Data (Clear Cache)** - Clears all caches

Both buttons are in the sidebar under "Quick Actions"

## Performance Impact

### ✅ What Refreshes (Light Operations)
- Mission Control metrics
- Chart displays
- Table data (from cache)
- Status indicators
- Timestamp displays

### ✅ What Doesn't Refresh (Heavy Operations)
- Backtest computations (cached)
- Historical data loads (cached)
- Wave universe rebuilds
- API calls (cached)

**Result**: Auto-Refresh has minimal performance impact

## Timestamps Explained

### Last Refresh
Shows when the app last refreshed (even if errors occurred)

### Last Successful Refresh  
Shows when the last error-free refresh completed

### Data Status
- 🟢 **Fresh**: Less than 2 minutes old
- 🟡 **Recent**: 2-5 minutes old
- 🔴 **Stale**: More than 5 minutes old

## Installation

Auto-Refresh requires `streamlit-autorefresh`:

```bash
pip install streamlit-autorefresh
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### "Auto-refresh not supported"
**Cause**: streamlit-autorefresh not installed  
**Fix**: `pip install streamlit-autorefresh`

### Frequent Auto-Pause
**Cause**: Recurring errors in data loading  
**Fix**: Check console logs, verify data files exist, check network

### App Feels Slow
**Cause**: Refresh interval too fast  
**Fix**: Increase interval to 120 seconds

### Not Refreshing
**Cause**: Auto-Refresh disabled or paused  
**Fix**: Check toggle is ON and status shows LIVE

## Advanced Configuration

For administrators wanting to change defaults:

Edit `app.py` and modify `AUTO_REFRESH_CONFIG`:

```python
AUTO_REFRESH_CONFIG = {
    "default_enabled": True,           # Start with ON/OFF
    "default_interval_seconds": 60,    # Default interval
    "allowed_intervals": [30, 60, 120], # Available options
    "pause_on_error": True,            # Auto-pause enabled
    "max_consecutive_errors": 3,       # Error threshold
}
```

## Documentation Files

- **AUTO_REFRESH_DOCUMENTATION.md** - Comprehensive guide (8.7 KB)
- **AUTO_REFRESH_IMPLEMENTATION_SUMMARY.md** - Technical details (7.1 KB)
- **test_auto_refresh.py** - Automated tests
- **verify_auto_refresh.py** - Verification script

## Support

Need help? Check these resources:

1. **User Guide**: AUTO_REFRESH_DOCUMENTATION.md
2. **Technical Summary**: AUTO_REFRESH_IMPLEMENTATION_SUMMARY.md
3. **Run Tests**: `python test_auto_refresh.py`
4. **Verify Setup**: `python verify_auto_refresh.py`

## Version

**Current Version**: 1.0.0  
**Release Date**: December 24, 2025  
**Status**: Production Ready ✅

---

**Tip**: For the best experience, use the default 60-second interval and let Auto-Refresh keep your dashboard current!
