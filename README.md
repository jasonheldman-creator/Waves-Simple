# Waves-Simple
WAVES Intelligence™ - Institutional Console v2

## Live Site

**Production URL:** `https://www.wavesintelligence.app` (after Vercel setup)

**Preview URLs:** Automatically generated for each pull request
- Preview URLs appear in GitHub PR checks under "Vercel – site"
- Click "Details" on the Vercel check to visit the preview deployment
- The Vercel bot also comments on PRs with direct deployment links

**Deployment Documentation:**
- **[VERCEL_SETUP.md](VERCEL_SETUP.md)** - Step-by-step Vercel configuration guide (⚠️ **START HERE**)
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment reference and troubleshooting
- **[site/README.md](site/README.md)** - Marketing site documentation

## Overview

WAVES Intelligence™ provides advanced portfolio analytics, alpha attribution, and decision-making tools for institutional investors. The system tracks multiple investment "waves" with comprehensive performance metrics, risk analysis, and automated reporting.

## Safe Mode

### What is Safe Mode?

Safe Mode is a stability feature that prevents infinite loops and ensures the app becomes interactive quickly. When Safe Mode is ON (default):

- ✅ **No external data fetching** - No calls to yfinance, Alpaca, Coinbase, or other price providers
- ✅ **No snapshot auto-builds** - Snapshots are never automatically regenerated during page load
- ✅ **Read-only operation** - The app loads pre-existing snapshot files only
- ✅ **Fast startup** - App becomes interactive immediately without waiting for data downloads

### How to Use Safe Mode

1. **Safe Mode Toggle** - Located at the top of the sidebar
   - Default: **ON** (recommended for stability)
   - Turn OFF only when you need to rebuild snapshots

2. **Manual Snapshot Rebuild** - Use the sidebar buttons to rebuild data:
   - **"Rebuild Snapshot Now (Manual)"** - Rebuilds the main analytics snapshot
   - **"Rebuild Proxy Snapshot Now (Manual)"** - Rebuilds the proxy analytics snapshot
   - Both buttons are only available when Safe Mode is OFF

3. **Run Guard Protection** - Automatically stops the app if it detects an infinite loop
   - Triggers after 3+ consecutive runs
   - Shows a warning banner: "Run Guard triggered — preventing infinite loop"
   - Reset by successfully rebuilding a snapshot

### When to Turn Safe Mode OFF

Turn Safe Mode OFF only when:
- You need to fetch fresh market data
- You want to rebuild analytics snapshots
- You're updating wave configurations and need to regenerate data

**Important:** Always turn Safe Mode back ON after rebuilding to prevent infinite loops.

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Seed initial historical data (first-time setup):
```bash
python seed_wave_history.py
```

3. Run the application:
```bash
streamlit run app.py
```

### Data Seeding

The system includes an automated data seeding tool that generates synthetic historical data for waves without real market data. This allows all analytics to function immediately.

**Usage:**
```bash
# Seed with default settings (90 days)
python seed_wave_history.py

# Custom configuration
python seed_wave_history.py --days 180 --start-date 2024-01-01

# Preview without changes
python seed_wave_history.py --dry-run
```

**Features:**
- ✅ Idempotent (safe to run multiple times)
- ✅ Marks synthetic data with `is_synthetic=True`
- ✅ Automatic backup before changes
- ✅ Compatible with all analytics components

For detailed documentation, see [SEEDING_DOCUMENTATION.md](SEEDING_DOCUMENTATION.md)

## Build Version Tracking

The application displays build/version stamps in the UI to help verify which version is currently deployed.

### Streamlit App (app.py)

The Streamlit application displays a build banner at the top of the page showing:
- **Git SHA**: Short commit hash (automatically detected from Git)
- **Date**: Current date
- **Branch**: Current Git branch

**Format:** `WAVES BUILD: {short_git_sha} | {date} | {branch}`

### Next.js Site (site/)

The Next.js marketing site displays a build ID in the footer.

### Setting BUILD_ID for Deployment

For deployments where Git is not available (e.g., containerized environments), set the `BUILD_ID` environment variable:

**Streamlit App:**
```bash
# Set BUILD_ID before running the app
export BUILD_ID="v1.2.3-abc123"
streamlit run app.py
```

**Next.js Site:**
```bash
# Set NEXT_PUBLIC_BUILD_ID before building
export NEXT_PUBLIC_BUILD_ID="v1.2.3-abc123"
cd site
npm run build
npm run start
```

**Docker/Container Deployments:**
```dockerfile
# In your Dockerfile or docker-compose.yml
ENV BUILD_ID="v1.2.3-abc123"
ENV NEXT_PUBLIC_BUILD_ID="v1.2.3-abc123"
```

**CI/CD Pipelines:**
```yaml
# GitHub Actions example
env:
  BUILD_ID: ${{ github.sha }}
  NEXT_PUBLIC_BUILD_ID: ${{ github.sha }}
```

**Vercel Deployment:**
- For the Next.js site, add `NEXT_PUBLIC_BUILD_ID` as an environment variable in your Vercel project settings
- The value can use Vercel system environment variables: `VERCEL_GIT_COMMIT_SHA`

This ensures that when a PR is deployed, users can immediately verify that their changes are live by checking the build stamp in the UI.

## Key Features

### Analytics Components

- **📊 Attribution Analysis**: Precise alpha decomposition into actionable components
- **📈 Performance Deep Dive**: Detailed wave performance charts and metrics
- **📋 Decision Ledger**: Governance layer for tracking decisions and contracts
- **📄 Board Pack**: Comprehensive PDF reports for institutional stakeholders
- **🎯 Mission Control**: Real-time market regime and risk monitoring
- **📊 WaveScore**: Proprietary performance scoring system

### Data Management

- **wave_id System**: Canonical wave identifiers for robust data management
- **Synthetic Data**: Automatic placeholder data with clear UI indicators
- **Real Data Integration**: Seamless transition from synthetic to real market data
- **Backward Compatibility**: Supports legacy data formats

### Plan B Proxy Analytics System

The **Wave Intelligence (Plan B)** system provides a parallel analytics pipeline that uses proxy tickers to deliver consistent analytics for all 28 waves, independent of individual wave ticker health.

#### Overview

- **Purpose**: Ensure all 28 waves always show analytics, even when specific wave tickers are broken or unavailable
- **Approach**: Use proxy ETFs and representative tickers as stand-ins for wave performance
- **Location**: Accessible via the "Wave Intelligence (Plan B)" tab in the UI

#### Proxy Registry

The proxy registry is the single source of truth for proxy ticker assignments:

**File**: `config/wave_proxy_registry.csv`

**Schema**:
- `wave_id`: Unique wave identifier
- `display_name`: Human-readable wave name
- `category`: Wave category (Equity, Crypto, Fixed Income, Commodity)
- `primary_proxy_ticker`: Primary proxy ticker (e.g., SPY, QQQ, MSFT)
- `secondary_proxy_ticker`: Fallback proxy ticker if primary fails
- `benchmark_ticker`: Benchmark ticker for alpha calculations
- `enabled`: Whether the wave is enabled (true/false)

**Validation**: Run `python -c "from helpers.proxy_registry_validator import validate_proxy_registry; print(validate_proxy_registry()['report'])"`

#### Updating the Proxy Registry

To update proxy tickers for a wave:

1. Edit `config/wave_proxy_registry.csv`
2. Update the `primary_proxy_ticker`, `secondary_proxy_ticker`, or `benchmark_ticker` for the target wave
3. Save the file
4. Run validation: `python -c "from helpers.proxy_registry_validator import validate_proxy_registry; print(validate_proxy_registry()['report'])"`
5. Rebuild the snapshot from the UI or via CLI: `python -c "from planb_proxy_pipeline import build_proxy_snapshot; build_proxy_snapshot()"`

#### Snapshot Files

The proxy pipeline generates and maintains the following files:

- **`site/data/live_proxy_snapshot.csv`**: Latest proxy analytics for all 28 waves
  - Contains returns (1D, 30D, 60D, 365D)
  - Contains alpha calculations vs. benchmarks
  - Contains confidence labels (FULL, PARTIAL, UNAVAILABLE)
  
- **`planb_diagnostics_run.json`**: Diagnostics from the last pipeline run
  - Timestamp and parameters
  - Success/failure counts
  - Detailed ticker failure information

#### Using the Plan B System

1. **UI Access**: Navigate to "Wave Intelligence (Plan B)" tab
2. **Wave Selection**: Use dropdown to select a wave
3. **View Analytics**: See proxy returns, alpha, and diagnostics
4. **Universe View**: See summary table for all 28 waves
5. **Rebuild Snapshot**: Click "🔄 Rebuild Snapshot" to fetch latest data
6. **Download Data**: Click "📥 Download CSV" to export snapshot

#### Pipeline Features

- **MAX_RETRIES=2**: Each ticker fetch is retried up to 2 times on failure
- **Graceful Degradation**: Returns UNAVAILABLE status when data cannot be fetched
- **No Hanging**: Pipeline completes within bounded time, logs all failures
- **Confidence Labels**:
  - `FULL`: Primary proxy and benchmark data available
  - `PARTIAL`: Secondary proxy used (primary failed), benchmark available
  - `UNAVAILABLE`: No proxy data available

## Testing

### Run Validation Tests
```bash
# Wave ID system tests
python test_wave_id_system.py

# Seeding validation tests
python test_seeding_validation.py

# Alpha attribution tests
python test_alpha_attribution.py

# Strategy integration tests
python test_strategy_integration.py
```

All tests should pass before deployment.

## Architecture

### Core Files

- `app.py`: Main Streamlit application
- `waves_engine.py`: Core engine with wave registry and calculations
- `alpha_attribution.py`: Alpha decomposition engine
- `decision_ledger.py`: Decision tracking and logging
- `seed_wave_history.py`: Automated data seeding tool

### Data Files

- `wave_history.csv`: Historical performance data (wave_id, returns, is_synthetic)
- `wave_config.csv`: Wave configuration and benchmarks
- `wave_weights.csv`: Wave weightings
- `prices.csv`: Market price data

### Documentation

- `SEEDING_DOCUMENTATION.md`: Data seeding guide
- `ALPHA_ATTRIBUTION_DOCUMENTATION.md`: Attribution system docs
- `VIX_REGIME_OVERLAY_DOCUMENTATION.md`: VIX overlay system
- `CRYPTO_ASSET_EXPANSION.md`: Crypto asset integration

## Synthetic Data Notice

When you see the 📊 **Synthetic Data Notice** banner in the UI:
- This indicates that some waves are using placeholder data
- Synthetic data is marked with `is_synthetic=True` in wave_history.csv
- Data will be automatically replaced as real market data is ingested
- All analytics function normally with synthetic data
- The banner shows the percentage and affected waves

## Development

### Adding New Waves

1. Add wave definition to `waves_engine.py` WAVE_ID_REGISTRY
2. Add holdings to WAVE_WEIGHTS
3. Run seeding: `python seed_wave_history.py`
4. Verify: `python test_wave_id_system.py`

### Contributing

- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all validation tests pass

## Support

For issues, questions, or contributions:
1. Check documentation files
2. Run validation tests
3. Review error messages and logs
4. Check wave_id registry in waves_engine.py

