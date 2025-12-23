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

