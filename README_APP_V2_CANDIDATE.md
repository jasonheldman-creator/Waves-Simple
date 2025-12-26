# app_v2_candidate.py - EXPERIMENTAL VERSION

> **⚠️ CRITICAL WARNING**  
> **This is NOT the production console.**  
> **For production use, run: `streamlit run app.py`**

## Purpose

This file (`app_v2_candidate.py`) was created from PR #73, which proposed a complete rewrite of the WAVES console. To preserve the existing production features and avoid disruption, this proposed version has been saved as `app_v2_candidate.py` rather than replacing the production `app.py`.

## Background

PR #73 proposed replacing the existing ~12,000 line production console with a new ~660 line implementation. While the new version offers a streamlined approach, it lacks many critical features present in the current production system:

### Features in Production app.py (NOT in V2 Candidate)
- WaveScore v1.0 (range of 0-100)
- Advanced overlays system
- Comprehensive benchmark truth system
- Detailed attribution sections
- Current table outputs
- Multiple existing tabs
- Institutional rail ticker
- Auto-refresh functionality
- Wave profile features
- Rich HTML rendering
- And many more production features developed over time

### Features in V2 Candidate (NEW)
- Simplified 6-tab interface
- Vector Truth attribution layer
- Decision Intelligence integration
- Streamlined NAV computation
- Direct engine integration

## Usage

### Running the Experimental Version

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the experimental V2 candidate
streamlit run app_v2_candidate.py

# Access at http://localhost:8501
```

### Running the Production Version

```bash
# Run the production console (recommended)
streamlit run app.py

# Access at http://localhost:8501
```

## Verification Requirements

Before this experimental version could be considered for production use, the following must be verified:

### 1. Metric Accuracy Verification
Provide screenshots demonstrating that the V2 candidate correctly displays the following metrics for at least 3 waves, and that they match the production console exactly:
- 30D, 60D, and 365D Return
- 30D, 60D, and 365D Benchmark Return  
- 30D, 60D, and 365D Alpha values

### 2. Feature Preservation Verification
Confirm through testing that the following existing features are preserved (either in V2 candidate or guaranteed to remain in main branch):
- ✅ WaveScore v1.0 (range of 0-100)
- ✅ Overlays functionality
- ✅ Benchmark truth system
- ✅ Attribution sections
- ✅ Current table outputs
- ✅ All existing tabs and metrics

### 3. No Breaking Changes
- ✅ No renaming of existing tabs
- ✅ No removal of existing metrics
- ✅ No disruption to existing production features

## Documentation

- **Technical Details**: See [APP_V2_CANDIDATE_DOCS.md](./APP_V2_CANDIDATE_DOCS.md)
- **Quick Start Guide**: See [APP_V2_CANDIDATE_QUICKSTART.md](./APP_V2_CANDIDATE_QUICKSTART.md)
- **Production App Documentation**: Refer to existing production documentation

## Status

- ✅ Extracted from PR #73
- ✅ Renamed to `app_v2_candidate.py`
- ✅ Production `app.py` preserved unchanged
- ✅ Documentation updated
- ⏸️ Awaiting verification testing
- ⏸️ NOT approved for production use
- ⏸️ NOT recommended for deployment

## Next Steps

1. **Testing**: Conduct comprehensive testing of both versions
2. **Comparison**: Generate side-by-side metric comparisons
3. **Screenshots**: Capture verification screenshots as required
4. **Decision**: Determine if V2 candidate meets requirements for production
5. **Integration**: If approved, create migration plan

## Safety Notes

This approach ensures:
- ✅ Zero risk to existing production console
- ✅ Production features remain fully intact
- ✅ Users can test experimental version safely
- ✅ Easy rollback if issues are found
- ✅ Clear separation of production vs experimental code

---

**Created**: December 26, 2025  
**Source**: PR #73 (copilot/create-production-app-file branch)  
**Status**: EXPERIMENTAL - NOT FOR PRODUCTION USE
