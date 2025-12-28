# WAVE SNAPSHOT LEDGER Documentation

## Overview

The WAVE SNAPSHOT LEDGER is a new analytics pipeline that provides **28/28 Waves performance metrics** without depending on full ticker coverage or data-ready gating. It uses a tiered fallback approach to ensure no Wave is excluded, even when data is incomplete or unavailable.

## Key Features

### 1. Complete Coverage Guarantee
- **28/28 Waves Always Rendered**: Every wave in the registry is included in the snapshot
- **No Exclusions**: Waves are never hidden due to missing data or ticker failures
- **Tiered Fallback System**: Automatic degradation to ensure metrics are always available

### 2. Tiered Data Sourcing

The pipeline uses a 4-tier fallback ladder (A → B → C → D) to compute metrics:

#### Tier A: Full History (Preferred)
- Uses `compute_history_nav()` from waves_engine
- Requires 7+ days of NAV data
- Provides complete analytics with all timeframes
- Status: `Full` or `Partial` depending on history length

#### Tier B: Limited History
- Uses recent NAV points (7-30 days)
- Computes available timeframes only
- Lower beta computation threshold (10 points vs 20)
- Status: `Operational`

#### Tier C: Holdings Reconstruction (Future)
- Reconstructs returns from portfolio weights
- Uses top N holdings that successfully fetch
- Renormalizes weights for missing tickers
- Status: `Operational`

#### Tier D: Benchmark Fallback (Always Succeeds)
- Sets Wave return = Benchmark return
- Alpha = 0 (no outperformance)
- Exposure and Cash computed from VIX ladder
- Status: `Unavailable`
- **Important**: This tier ensures no Wave is ever excluded

### 3. Comprehensive Metrics

The snapshot includes 29 columns for each Wave:

#### Identity
- `Wave`: Wave display name
- `Mode`: Operating mode (Standard, Alpha-Minus-Beta, Private Logic)
- `Date`: Snapshot generation date

#### NAV Metrics
- `NAV`: Current Net Asset Value
- `NAV_1D_Change`: 1-day NAV change

#### Returns (Wave)
- `Return_1D`: 1-day return
- `Return_30D`: 30-day return
- `Return_60D`: 60-day return
- `Return_365D`: 365-day return

#### Returns (Benchmark)
- `Benchmark_Return_1D`: 1-day benchmark return
- `Benchmark_Return_30D`: 30-day benchmark return
- `Benchmark_Return_60D`: 60-day benchmark return
- `Benchmark_Return_365D`: 365-day benchmark return

#### Alpha (Outperformance)
- `Alpha_1D`: Wave return - Benchmark return (1D)
- `Alpha_30D`: Wave return - Benchmark return (30D)
- `Alpha_60D`: Wave return - Benchmark return (60D)
- `Alpha_365D`: Wave return - Benchmark return (365D)

#### Risk & Exposure
- `Exposure`: Current market exposure (0.0-1.5)
- `CashPercent`: Percentage in safe assets (0.0-1.0)
- `VIX_Level`: Current VIX level
- `VIX_Regime`: VIX regime (low/normal/elevated/high)
- `Beta_Real`: Realized beta vs benchmark
- `Beta_Target`: Target beta
- `Beta_Drift`: |Beta_Real - Beta_Target|
- `Turnover_Est`: Estimated annual turnover
- `MaxDD`: Maximum drawdown

#### Data Quality
- `Flags`: Human-readable data quality indicators
- `Data_Regime_Tag`: Overall data status (Full/Partial/Operational/Unavailable)
- `Coverage_Score`: Data coverage percentage (0-100)

### 4. VIX Ladder Logic

Exposure and Cash percentages are computed independently of ticker availability:

| VIX Level | Exposure Multiplier | Cash Percentage | Regime |
|-----------|-------------------|-----------------|--------|
| < 15      | 1.1               | 0%              | Low    |
| 15-20     | 1.0               | 5%              | Normal |
| 20-25     | 0.9               | 15%             | Elevated |
| 25-30     | 0.8               | 30%             | High   |
| > 30      | 0.7               | 50%             | High   |

Mode-specific adjustments:
- **Alpha-Minus-Beta**: More defensive (+15% cash, max 85% exposure)
- **Private Logic**: More aggressive (-5% cash, max 150% exposure)

### 5. Snapshot Persistence

#### Cache Location
- File: `data/live_snapshot.csv`
- Format: CSV with 29 columns
- TTL: 24 hours

#### Cache Management
- **Auto-refresh**: Snapshot regenerates after 24 hours
- **Force refresh**: Manual regeneration via UI button
- **Runtime guard**: Max 300 seconds (5 minutes) per generation
- **Timeout handling**: Remaining waves use Tier D fallback

## Usage

### Programmatic Access

```python
from snapshot_ledger import load_snapshot, generate_snapshot, get_snapshot_metadata

# Load snapshot (uses cache if available)
snapshot_df = load_snapshot(force_refresh=False)

# Force regeneration
snapshot_df = generate_snapshot(force_refresh=True, max_runtime_seconds=300)

# Get metadata
metadata = get_snapshot_metadata()
print(f"Snapshot age: {metadata['age_hours']:.1f} hours")
print(f"Wave count: {metadata['wave_count']}")
print(f"Is stale: {metadata['is_stale']}")
```

### UI Access

1. Navigate to **Overview** tab
2. View **Wave Snapshot Ledger** section at top
3. Click **Force Refresh** to regenerate snapshot
4. Expand **Full Snapshot Table** to view all 28 waves
5. Review summary statistics for data quality distribution

## Integration with Overview Tab

### Snapshot Section
- Located at top of Overview tab
- Shows:
  - Last snapshot timestamp
  - Force refresh button
  - Summary statistics (Full/Partial/Operational/Unavailable counts)
  - Expandable table with all 28 waves

### Display Format
- Percentage columns formatted as `XX.XX%`
- Numeric columns formatted to 4 decimal places
- NaN values displayed as `N/A`
- Flags column shows human-readable status

## Implementation Details

### Error Handling

The implementation includes comprehensive error handling:

1. **Import Failures**: Graceful degradation if modules unavailable
2. **Network Failures**: Tier D fallback ensures completion
3. **Runtime Limits**: Timeout protection with partial results
4. **Missing Data**: NaN handling in all metric computations

### Performance Optimization

1. **Global Price Cache**: Reuses cached price data across waves
2. **Persistent Snapshot**: Avoids regeneration on every page load
3. **TTL-based Refresh**: Only regenerates when stale
4. **Runtime Guard**: Prevents infinite generation loops

### Code Structure

```
snapshot_ledger.py
├── Helper Functions
│   ├── _safe_return()          # Compute returns from NAV
│   ├── _compute_beta()         # Compute beta from returns
│   ├── _compute_max_drawdown() # Compute max drawdown
│   ├── _estimate_turnover()    # Estimate from trades.csv
│   ├── _get_vix_level_and_regime() # Get VIX data
│   └── _compute_exposure_and_cash() # VIX ladder logic
│
├── Tier Functions
│   ├── _build_snapshot_row_tier_a() # Full history
│   ├── _build_snapshot_row_tier_b() # Limited history
│   ├── _build_snapshot_row_tier_c() # Holdings reconstruction
│   └── _build_snapshot_row_tier_d() # Benchmark fallback
│
└── Public API
    ├── generate_snapshot()     # Generate snapshot
    ├── load_snapshot()         # Load from cache
    └── get_snapshot_metadata() # Get metadata
```

## Benefits

### For Users
1. **Consistent UI**: Overview tab always shows 28/28 waves
2. **No Loading Issues**: Infinite loading eliminated
3. **Data Quality Visibility**: Flags show data status
4. **Fast Rendering**: Cached snapshot loads instantly

### For Developers
1. **Graceful Degradation**: No crashes on ticker failures
2. **Testable**: Each tier can be tested independently
3. **Maintainable**: Clear separation of concerns
4. **Extensible**: Easy to add new metrics or tiers

### For Operations
1. **Monitoring**: Metadata shows snapshot health
2. **Performance**: Runtime guard prevents hangs
3. **Diagnostics**: Flags indicate data issues
4. **Self-healing**: Auto-refresh keeps data current

## Future Enhancements

### Tier C Implementation
- Implement holdings-based reconstruction
- Renormalize weights for missing tickers
- Compute degraded-mode returns

### Advanced Metrics
- Sharpe ratio
- Sortino ratio
- Information ratio
- Tracking error

### Historical Snapshots
- Store daily snapshots for trending
- Compare snapshot changes over time
- Alert on significant changes

### Real-time Updates
- WebSocket integration for live updates
- Incremental snapshot updates
- Push notifications for data changes

## Troubleshooting

### Snapshot Not Generating
1. Check waves_engine availability
2. Verify wave registry has 28 entries
3. Review logs for error messages
4. Try force refresh with network access

### Stale Data
1. Click "Force Refresh" button
2. Verify snapshot file permissions
3. Check TTL settings (24 hours default)

### Missing Metrics
1. Review Flags column for data quality
2. Check tier used (A/B/C/D)
3. Verify ticker availability
4. Review Coverage_Score

### Performance Issues
1. Adjust max_runtime_seconds (default: 300)
2. Pre-fetch global price cache
3. Verify disk I/O performance
4. Check network latency

## Security Considerations

✅ **CodeQL Analysis**: No security vulnerabilities detected

- No SQL injection risks (uses pandas CSV)
- No command injection risks (no shell commands)
- No path traversal risks (fixed file path)
- No XSS risks (data sanitized by Streamlit)

## Conclusion

The WAVE SNAPSHOT LEDGER provides a robust, fault-tolerant analytics pipeline that ensures complete Wave coverage regardless of data availability. The tiered fallback approach, combined with comprehensive error handling and performance optimization, delivers a reliable foundation for the Overview tab and future analytics features.
