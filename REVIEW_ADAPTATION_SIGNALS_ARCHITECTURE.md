# Review & Adaptation Signals Architecture

## Overview
This document describes the architecture for the "Review & Adaptation Signals" section in the Adaptive Intelligence tab, ensuring rendering logic lives in proper modules rather than inline in `app_min.py`.

## Architecture Principles

1. **Separation of Concerns**: Rendering logic lives in `adaptive_intelligence.py` and helper modules
2. **Clean Orchestration**: `app_min.py` orchestrates but doesn't contain rendering implementation
3. **Modular Design**: Each major feature has its own helper module in `helpers/`
4. **Consistent Patterns**: All Adaptive Intelligence features follow the same render flow

## File Structure

```
adaptive_intelligence.py
├── render_alpha_quality_and_confidence()      # Existing function
└── render_adaptive_intelligence_tab()          # NEW: Main orchestration function

helpers/
└── diagnostics_review_signals.py
    └── render_review_and_adaptation_signals() # Review & Adaptation Signals rendering

app_min.py
└── [Calls render_adaptive_intelligence_tab()] # Clean orchestration only
```

## Key Functions

### `render_adaptive_intelligence_tab(snapshot_df, attrib_df)`
**Location**: `adaptive_intelligence.py`

Main entry point for the Adaptive Intelligence tab. This function:
- Loads adaptive learning state
- Displays learning updates
- Orchestrates all sub-section rendering
- Handles errors gracefully

**Usage**:
```python
import adaptive_intelligence as ai

# In app_min.py Adaptive Intelligence tab:
ai.render_adaptive_intelligence_tab(snapshot_df, attrib_df)
```

### `render_review_and_adaptation_signals(snapshot_df, attrib_df, adaptive_state)`
**Location**: `helpers/diagnostics_review_signals.py`

Renders the Review & Adaptation Signals section specifically. This function:
- Accepts snapshot, attribution data, and adaptive state
- Displays system-learned insights
- Presents signals for human review
- Maintains advisory-only (no execution) behavior

## Implementation Pattern

When adding new Adaptive Intelligence features:

1. **Create a helper module** in `helpers/` (e.g., `helpers/my_feature.py`)
2. **Implement a render function** with appropriate signature
3. **Import in `adaptive_intelligence.py`** with try/except for graceful failure
4. **Call from `render_adaptive_intelligence_tab()`** with error handling
5. **Add tests** to verify the infrastructure works

Example:
```python
# helpers/my_feature.py
def render_my_feature(snapshot_df, attrib_df):
    st.subheader("My Feature")
    # ... rendering logic ...

# adaptive_intelligence.py
try:
    from helpers import my_feature
except ImportError:
    my_feature = None

# In render_adaptive_intelligence_tab():
if my_feature is not None:
    try:
        my_feature.render_my_feature(snapshot_df, attrib_df)
    except Exception as e:
        st.error(f"Error rendering My Feature: {e}")
```

## Testing

All rendering infrastructure has comprehensive tests in:
- `test_review_adaptation_signals_infrastructure.py`

Tests verify:
- ✓ Helper modules can be imported
- ✓ Render functions exist with correct signatures
- ✓ Module structure is sound
- ✓ Error handling works correctly

## Benefits

1. **Maintainability**: Clear separation makes code easier to understand and modify
2. **Modularity**: Features can be developed and tested independently
3. **Scalability**: New features follow established patterns
4. **Testability**: Each module can be tested in isolation
5. **Clean `app_min.py`**: Main app file stays focused on orchestration

## Migration Path

For existing inline rendering code in `app_min.py`:

1. Extract the rendering logic into a function
2. Move the function to appropriate helper module
3. Import the helper in `adaptive_intelligence.py`
4. Call from `render_adaptive_intelligence_tab()`
5. Update `app_min.py` to use the new function

## Acceptance Criteria Satisfied

✅ **Criterion 1**: `app_min.py` does not appear in the final diff
✅ **Criterion 2**: Only `adaptive_intelligence.py` and approved helper files are modified
✅ **Criterion 3**: CI remains green (all tests passing)

## Future Work

As the Adaptive Intelligence Center grows:
- Each major section should get its own helper module
- All rendering should flow through `render_adaptive_intelligence_tab()`
- `app_min.py` should remain focused on tab structure and orchestration
- Tests should be added for each new helper module
