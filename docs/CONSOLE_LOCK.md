# Console Architecture Protection

## Overview

This document establishes architectural protection rules for the WAVES Institutional Console (`app.py`).

## Core Protection Rules

### 1. No Rewrites Allowed

`app.py` **MUST NOT** be rewritten or replaced. The console architecture has been carefully designed and tested.

### 2. Additive Edits Only

Changes to `app.py` must be:

- **Additive**: New features can be added
- **Surgical**: Bug fixes should be minimal and targeted
- **Non-destructive**: Existing features and tabs must remain functional

### 3. Protected Features

The following console features are **protected** and must not be removed:

#### Analytics Tabs
- Console (Executive Dashboard)
- Overview (Market Context)
- Wave Profile (Wave Intelligence Center)
- Details (Factor Decomposition)
- Reports (Board Pack Generator)
- Overlays (VIX Regime Analysis)
- Attribution (Alpha Attribution)
- Board Pack
- IC Pack
- Alpha Capture
- Diagnostics

#### Sidebar Components
- Risk Lab information
- Correlation Matrix information
- Rolling Alpha/Volatility information
- Drawdown Monitor information
- Wave List Debug panel
- Build Information
- Ops Controls

#### Core Functions
All render functions for tabs and analytics visualizations must be preserved.

## Change Management

### Allowed Changes

✅ Adding new tabs or features
✅ Adding new visualization components
✅ Performance optimizations that don't alter functionality
✅ Bug fixes with minimal code changes
✅ Documentation improvements

### Prohibited Changes

❌ Removing existing tabs
❌ Removing existing analytics features
❌ Complete file rewrites
❌ Breaking changes to core navigation
❌ Removal of console layouts or visualizations

## Enforcement

This protection is enforced by:

1. Repository tag: `CONSOLE_LOCKED_PROD`
2. Code review requirements
3. This documentation file

## Version History

- **2025-12-27**: Initial console lock documentation created
- Tag: `CONSOLE_LOCKED_PROD` applied to protect institutional console architecture
