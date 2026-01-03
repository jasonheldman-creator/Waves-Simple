# Multi-Tab Console UI - Tab Structure Documentation

## Visual Tab Layout

This document provides a "screenshot" view of the complete tab structure in the Waves-Simple application.

## Tab Configuration: ENABLE_WAVE_PROFILE = True (17 visible tabs)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Institutional Console - Executive Layer v2                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌───────┐┌───────┐┌───────┐┌──────┐┌───────┐┌───────┐┌────────┐┌───────────┐ │
│  │Overv. ││Overv. ││Consol.││ Wave ││Detail.││Report.││Overlay.││Attribution│ │
│  │(Clean)││       ││       ││      ││       ││       ││        ││           │ │
│  └───────┘└───────┘└───────┘└──────┘└───────┘└───────┘└────────┘└───────────┘ │
│                                                                                  │
│  ┌──────┐┌───────┐┌──────┐┌──────┐┌──────┐┌────────┐┌────────┐┌───────┐       │
│  │Board ││IC Pack││Alpha ││Wave  ││Plan B││Wave Int││Govern. ││Diagno.│       │
│  │Pack  ││       ││Capt. ││Monit.││Monit.││(Plan B)││& Audit ││       │       │
│  └──────┘└───────┘└──────┘└──────┘└──────┘└────────┘└────────┘└───────┘       │
│                                                                                  │
│  ┌──────────┐                                                                   │
│  │Wave Overv││                                                                   │
│  │(New)     ││                                                                   │
│  └──────────┘                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tab Listing (17 tabs when ENABLE_WAVE_PROFILE=True):

1. **Overview (Clean)** - First tab, clean demo-ready overview
2. **Overview** - Unified system overview with performance and context
3. **Console** - Executive console with core functionality
4. **Wave** - Wave Profile with hero card (Wave Intelligence Center)
5. **Details** - Factor decomposition and detailed analytics
6. **Reports** - Risk lab and reporting
7. **Overlays** - Correlation and overlay analysis
8. **Attribution** - Rolling diagnostics and attribution
9. **Board Pack** - Board pack generation
10. **IC Pack** - Investment Committee pack
11. **Alpha Capture** - Alpha capture analytics
12. **Wave Monitor** - Individual wave analytics (ROUND 7 Phase 5)
13. **Plan B Monitor** - Plan B canonical metrics
14. **Wave Intelligence (Plan B)** - Proxy-based analytics for all 28 waves
15. **Governance & Audit** - Governance and transparency layer
16. **Diagnostics** - Health/diagnostics and system monitoring
17. **Wave Overview (New)** - Comprehensive all-waves overview

## Tab Configuration: ENABLE_WAVE_PROFILE = False (16 visible tabs)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Institutional Console - Executive Layer v2                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐┌────────┐┌───────────┐          │
│  │Overv. ││Overv. ││Consol.││Detail.││Report.││Overlay.││Attribution│          │
│  │(Clean)││       ││       ││       ││       ││        ││           │          │
│  └───────┘└───────┘└───────┘└───────┘└───────┘└────────┘└───────────┘          │
│                                                                                  │
│  ┌──────┐┌───────┐┌──────┐┌──────┐┌──────┐┌────────┐┌────────┐┌───────┐       │
│  │Board ││IC Pack││Alpha ││Wave  ││Plan B││Wave Int││Govern. ││Diagno.│       │
│  │Pack  ││       ││Capt. ││Monit.││Monit.││(Plan B)││& Audit ││       │       │
│  └──────┘└───────┘└──────┘└──────┘└──────┘└────────┘└────────┘└───────┘       │
│                                                                                  │
│  ┌──────────┐                                                                   │
│  │Wave Overv││                                                                   │
│  │(New)     ││                                                                   │
│  └──────────┘                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tab Listing (16 tabs when ENABLE_WAVE_PROFILE=False):

1. **Overview (Clean)** - First tab, clean demo-ready overview
2. **Overview** - Unified system overview
3. **Console** - Executive console
4. **Details** - Factor decomposition
5. **Reports** - Risk lab
6. **Overlays** - Correlation analysis
7. **Attribution** - Attribution analytics
8. **Board Pack** - Board pack
9. **IC Pack** - IC pack
10. **Alpha Capture** - Alpha capture
11. **Wave Monitor** - Wave monitoring
12. **Plan B Monitor** - Plan B metrics
13. **Wave Intelligence (Plan B)** - Proxy analytics
14. **Governance & Audit** - Governance layer
15. **Diagnostics** - System diagnostics
16. **Wave Overview (New)** - All-waves overview

**Note:** The "Wave" tab (Wave Intelligence Center) is only visible when `ENABLE_WAVE_PROFILE=True`

## Tab Configuration: Safe Mode Fallback (16 visible tabs)

When `wave_ic_has_errors=True` and `use_safe_mode=True`, the Wave Intelligence Center tab is excluded:

Same layout as ENABLE_WAVE_PROFILE=False configuration.

## Implementation Details

### Render Function to Tab Mapping

| Render Function | Visible in Normal Mode | Visible in Wave Profile Mode | Visible in Safe Mode |
|----------------|------------------------|------------------------------|----------------------|
| render_overview_clean_tab | ✓ (1st tab) | ✓ (1st tab) | ✓ (1st tab) |
| render_executive_brief_tab | ✓ (as "Overview") | ✓ (as "Overview") | - |
| render_executive_tab | ✓ (as "Console") | ✓ (as "Console") | ✓ (as "Console") |
| render_overview_tab | ✓ | ✓ | ✓ |
| render_wave_intelligence_center_tab | - | ✓ (as "Wave") | - |
| render_details_tab | ✓ | ✓ | ✓ |
| render_reports_tab | ✓ | ✓ | ✓ |
| render_overlays_tab | ✓ | ✓ | ✓ |
| render_attribution_tab | ✓ | ✓ | ✓ |
| render_board_pack_tab | ✓ | ✓ | ✓ |
| render_ic_pack_tab | ✓ | ✓ | ✓ |
| render_alpha_capture_tab | ✓ | ✓ | ✓ |
| render_wave_monitor_tab | ✓ | ✓ | ✓ |
| render_planb_monitor_tab | ✓ | ✓ | ✓ |
| render_wave_intelligence_planb_tab | ✓ | ✓ | ✓ |
| render_governance_audit_tab | ✓ | ✓ | ✓ |
| render_diagnostics_tab | ✓ | ✓ | ✓ |
| render_wave_overview_new_tab | ✓ | ✓ | ✓ |

**Total:** 18 render functions → 16-17 visible tabs (depending on configuration)

## Error Handling

All tabs use the `safe_component()` wrapper which:
- Catches exceptions during rendering
- Displays user-friendly warning messages
- Prevents full-page crashes
- Allows the application to continue running with degraded functionality

This ensures compliance with Hard Constraint #2: "If data required for a tab is missing or broken, retain the tab and display a warning message inside the tab instead of removing it."

## Additional UI Components

Beyond the tab structure, the application includes:

1. **Selected Wave Banner** - Displays at the top showing current wave selection and mode
2. **Mission Control** - Dashboard showing system status
3. **Reality Panel** - Single source of truth for price data (PRICE_BOOK metadata)
4. **Sidebar** - Wave selection, settings, and controls
5. **Bottom Ticker Bar** - Institutional rail (when enabled)

---

**Document Generated:** 2026-01-03  
**Status:** ✅ Complete Multi-Tab Console UI Verified
