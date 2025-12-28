# Wave Registry CSV Self-Healing System

## Overview

The Wave Registry CSV Self-Healing System ensures correctness and completeness of the 28-Wave universe registry, even when the application encounters a corrupted or partial CSV file.

## Key Features

### 1. Canonical Source of Truth
- **Source:** `waves_engine.py` (WAVE_WEIGHTS, WAVE_ID_REGISTRY, BENCHMARK_WEIGHTS_STATIC)
- **28 Waves Total:** All waves defined in code are automatically included
- **No Manual Edits Required:** CSV is generated programmatically from code

### 2. Auto-Healing on Startup
- **Validation Before Analytics:** Runs `auto_heal_wave_registry()` on app startup
- **Automatic Rebuild:** If validation fails, CSV is rebuilt from canonical source
- **Clear Logging:** Success/failure messages logged with ✅/⚠️ indicators

### 3. Validation Checks
The system validates the CSV for:
1. **File Existence:** CSV must exist at `data/wave_registry.csv`
2. **Row Count:** Must have >= 28 rows (one per wave)
3. **Required Columns:** All mandatory columns present
4. **No Duplicate wave_ids:** Each wave has unique identifier
5. **No Blank wave_names:** All waves have display names
6. **Completeness:** All wave_ids from canonical source are present

### 4. Ticker Normalization
Automatic ticker format normalization for yfinance compatibility:
- `BRK.B` → `BRK-B`
- `BF.B` → `BF-B`
- `BF.A` → `BF-A`

Both raw and normalized tickers are stored in CSV.

### 5. Rebuild Button (UI)
Sidebar button: **"📋 Rebuild Wave CSV + Clear Cache"**

**Actions:**
1. Rebuilds CSV from canonical source (force=True)
2. Clears all Streamlit caches (st.cache_data, st.cache_resource)
3. Clears wave-related session state
4. Rebuilds global price cache
5. Reruns analytics pipeline
6. Displays summary report

**Summary Display:**
```
Rebuild Summary:
- Total Waves: 28
- Waves Loaded: 28
- Tickers Success: 245
- Tickers Failed: 5

⚠️ 5 tickers failed to load
[Table showing failed tickers and impacted waves]
```

## CSV Structure

### File Location
```
data/wave_registry.csv
```

### Required Columns
1. **wave_id** - Canonical identifier (snake_case) - e.g., `sp500_wave`
2. **wave_name** - Display name - e.g., `S&P 500 Wave`
3. **mode_default** - Default operating mode - e.g., `Standard`
4. **benchmark_spec** - Benchmark specification - e.g., `SPY:1.0` or `QQQ:0.6,IGV:0.4`
5. **holdings_source** - Source of holdings - e.g., `canonical`
6. **category** - Wave category - e.g., `equity_growth`, `crypto_growth`, `equity_income`, `crypto_income`, `special`
7. **active** - Boolean flag for wave status - e.g., `True`

### Optional Columns (Auto-Generated)
8. **ticker_raw** - Raw ticker symbols (comma-separated) - e.g., `AAPL,MSFT,BRK.B`
9. **ticker_normalized** - Normalized tickers for yfinance - e.g., `AAPL,MSFT,BRK-B`
10. **created_at** - Timestamp of CSV creation - ISO 8601 format
11. **updated_at** - Timestamp of last update - ISO 8601 format

## Usage

### Python API

```python
from wave_registry_manager import (
    rebuild_wave_registry_csv,
    validate_wave_registry_csv,
    auto_heal_wave_registry
)

# Validate existing CSV
validation_result = validate_wave_registry_csv()
if not validation_result['is_valid']:
    print(f"Validation failed: {validation_result['checks_failed']}")

# Rebuild CSV (force overwrite)
rebuild_result = rebuild_wave_registry_csv(force=True)
if rebuild_result['success']:
    print(f"✅ Rebuilt CSV with {rebuild_result['waves_written']} waves")

# Auto-heal (validate + rebuild if needed)
healed = auto_heal_wave_registry()
if healed:
    print("✅ CSV is valid or was successfully healed")
```

### Streamlit App Integration

**On Startup** (automatic):
```python
# In main() function
if "wave_registry_validated" not in st.session_state:
    healed = auto_heal_wave_registry()
    st.session_state.wave_registry_validated = True
```

**Manual Rebuild** (sidebar button):
- Click **"📋 Rebuild Wave CSV + Clear Cache"**
- Wait for rebuild and cache refresh
- View summary report with any ticker failures

## Wave Categories

The system automatically infers wave categories:

### equity_growth
Traditional equity growth waves:
- S&P 500 Wave
- AI & Cloud MegaCap Wave
- Quantum Computing Wave
- etc.

### equity_income
Income-focused equity waves:
- Income Wave
- SmartSafe Treasury Cash Wave
- Vector Treasury Ladder Wave
- Vector Muni Ladder Wave

### crypto_growth
Cryptocurrency growth waves:
- Crypto L1 Growth Wave
- Crypto DeFi Growth Wave
- Crypto L2 Growth Wave
- Crypto AI Growth Wave
- Crypto Broad Growth Wave

### crypto_income
Cryptocurrency income waves:
- Crypto Income Wave

### special
Special asset waves:
- Gold Wave

## Testing

### Test Validation
```bash
cd /home/runner/work/Waves-Simple/Waves-Simple
python wave_registry_manager.py
```

### Test Auto-Heal
```python
# Delete CSV
import os
os.remove('data/wave_registry.csv')

# Run auto-heal
from wave_registry_manager import auto_heal_wave_registry
healed = auto_heal_wave_registry()  # Should create new CSV

# Verify
import pandas as pd
df = pd.read_csv('data/wave_registry.csv')
print(f"CSV has {len(df)} waves")  # Should be 28
```

## Acceptance Criteria

✅ **All 28 waves in CSV** - Verified programmatically from waves_engine.py

✅ **Startup validation** - Runs before analytics processing

✅ **Auto-rebuild on corruption** - Tested with missing rows and missing file scenarios

✅ **Ticker normalization** - BRK.B → BRK-B, BF.B → BF-B applied

✅ **Sidebar rebuild button** - Integrated with summary display and failed ticker reporting

✅ **Clear error reporting** - Failed tickers shown with impacted waves

## Architecture

```
waves_engine.py (Canonical Source)
    ↓
    ↓ WAVE_WEIGHTS, WAVE_ID_REGISTRY, BENCHMARK_WEIGHTS_STATIC
    ↓
wave_registry_manager.py
    ↓
    ↓ rebuild_wave_registry_csv()
    ↓ validate_wave_registry_csv()
    ↓ auto_heal_wave_registry()
    ↓
data/wave_registry.csv (28 waves)
    ↓
    ↓ Validated on startup
    ↓
app.py (Streamlit Application)
    ↓
    ↓ Sidebar: "Rebuild Wave CSV + Clear Cache"
    ↓
Analytics Pipeline (All 28 waves processed)
```

## Benefits

1. **Resilience:** Handles corrupted, partial, or missing CSV files
2. **Consistency:** Single source of truth (waves_engine.py)
3. **Automation:** No manual CSV editing required
4. **Transparency:** Clear validation messages and error reporting
5. **Recovery:** Self-healing on startup prevents infinite loaders
6. **Diagnostics:** Failed ticker reporting helps identify data issues

## Future Enhancements

- [ ] CSV versioning (track changes over time)
- [ ] Historical CSV backups
- [ ] Wave activation/deactivation flags
- [ ] Custom mode overrides per wave
- [ ] Extended metadata (description, tags, strategy family)

## Support

For issues or questions:
- Check validation logs in console output
- Review CSV at `data/wave_registry.csv`
- Use "Rebuild Wave CSV + Clear Cache" button to force refresh
- Check `wave_registry_manager.py` source code for implementation details
