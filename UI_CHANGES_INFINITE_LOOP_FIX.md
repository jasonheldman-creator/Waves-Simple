# UI Changes - Visual Guide

## Overview
This document describes the user-visible UI changes implemented to prevent infinite loading loops.

## 1. Run ID Display (Top of Page)

**Location:** Very top of the main app, before any other content

**Appearance:**
```
🔄 Run ID: 5 | Trigger: button: planb_rebuild
```

**What it shows:**
- Current run ID (increments on each rerun)
- What triggered this rerun (initial_load, auto_refresh, user_interaction, or specific button)

**Purpose:**
- Debugging tool to understand why app is rerunning
- Helps identify unexpected reruns

---

## 2. SAFE DEMO MODE Toggle (Sidebar)

**Location:** Sidebar, under Feature Settings section

**Appearance:**
```
☐ 🛡️ SAFE DEMO MODE (NO NETWORK / NO ENGINE RECOMPUTE)
```

When enabled:
```
☑ 🛡️ SAFE DEMO MODE (NO NETWORK / NO ENGINE RECOMPUTE)
ℹ️ SAFE DEMO MODE ACTIVE - Using cached data only
```

**What it does:**
- When ON: Prevents ALL network calls and compute operations
- App uses only cached snapshot files
- Immediate interactivity, no loading delays

**Purpose:**
- Emergency kill switch for infinite loops
- Demo mode for presentations without network
- Testing cached data paths

---

## 3. Build Status Indicators (Plan B Tab)

**Location:** Plan B (Wave Intelligence) tab, below diagnostics metrics

**Possible States:**

### Build In Progress
```
ℹ️ Plan B Build Running - Please wait...
```
or
```
ℹ️ Engine Build Running - Please wait...
```

### SAFE DEMO MODE Active
```
✅ SAFE DEMO MODE - All builds suppressed
```

### Build Lock Active
```
⚠️ Build Lock Active: Last build 3.2m ago. Must wait 6.8m more.
```

**Purpose:**
- Shows current build state
- Prevents confusion during builds
- Explains why builds are suppressed

---

## 4. Diagnostics Expander (Plan B Tab)

**Location:** Plan B tab, expandable section after Safe Mode banner

**Appearance:**
```
▶ 🔍 Diagnostics: Why is it rerunning?
```

When expanded:
```
▼ 🔍 Diagnostics: Why is it rerunning?

### Rerun & Build Diagnostics

[App Rerun Status]              [Plan B Build Status]
Run ID: 12                       Build In Progress: False
Trigger: user_interaction        Last Build Attempt: 2026-01-01 09:35:22
SAFE DEMO MODE: 🔴 OFF          Last Build Success: True
Plan B Safe Mode: 🔴 OFF        Minutes Since Last: 8.5
                                 Last Build Run ID: 10

### Engine Build Status

[Column 1]                       [Column 2]
Build In Progress: False         Minutes Since Last: 12.3
Last Build Attempt: Never        Last Build Run ID: N/A
Last Build Success: N/A

### Last Build Summary
Total Waves Processed: 28
Successful Fetches: 26
Failed Fetches: 2
Build Duration: 12.4s
Timeout Exceeded: False
```

**What it shows:**
- Current run ID and what triggered it
- SAFE DEMO MODE status (global and Plan B-specific)
- Build state for both Plan B and Engine snapshots
- Last build attempt times
- Success/failure status
- Cooldown information
- Build performance metrics

**Purpose:**
- Comprehensive troubleshooting information
- Understand why app is or isn't rebuilding
- Identify performance issues
- Debug infinite loop scenarios

---

## 5. Rebuild Button Behavior (Plan B Tab)

**Location:** Plan B tab, action buttons row

**Normal State:**
```
🔄 Rebuild Snapshot Now
```

**Disabled States:**

When SAFE MODE is ON:
```
🔄 Rebuild Snapshot Now (disabled - grayed out)
```

When build is in progress:
```
🔄 Rebuild Snapshot Now (disabled - grayed out)
```

**After Clicking:**
1. Button sets `_last_button_clicked` for tracking
2. Build starts with explicit_button_click=True
3. Status changes to "Build In Progress"
4. Spinner shows: "Rebuilding proxy snapshot (max 15s timeout)..."
5. On success: "✅ Snapshot rebuilt with 28 waves"
6. Build status clears

**Purpose:**
- Clear feedback on button state
- Explicit user intent bypasses cooldown
- Progress indication during build

---

## UI Flow Example

### Scenario: User Opens App

1. **Initial Load:**
   ```
   🔄 Run ID: 0 | Trigger: initial_load
   ```

2. **Snapshot Check:**
   - If fresh: No build
   - If stale: Build (if no cooldown)
   - If SAFE MODE: No build

3. **Build Status:**
   ```
   ℹ️ Engine Build Running - Please wait...
   ```

4. **After Build:**
   ```
   ✅ Snapshot is fresh (age: 0.1 min)
   ```

### Scenario: User Enables SAFE DEMO MODE

1. **Toggle ON:**
   ```
   ☑ 🛡️ SAFE DEMO MODE (NO NETWORK / NO ENGINE RECOMPUTE)
   ℹ️ SAFE DEMO MODE ACTIVE - Using cached data only
   ```

2. **Build Status:**
   ```
   ✅ SAFE DEMO MODE - All builds suppressed
   ```

3. **Rebuild Button:**
   - Grayed out/disabled
   - No network calls possible

4. **App Behavior:**
   - Loads from cached snapshots
   - Immediate interactivity
   - No timeouts or delays

### Scenario: User Clicks Rebuild

1. **Button Click:**
   ```
   🔄 Run ID: 5 | Trigger: button: planb_rebuild
   ```

2. **Build Starts:**
   ```
   ℹ️ Plan B Build Running - Please wait...
   🔄 Rebuilding proxy snapshot (max 15s timeout)...
   ```

3. **Diagnostics Update:**
   ```
   Build In Progress: True
   Last Build Run ID: 5
   ```

4. **Build Complete:**
   ```
   ✅ Snapshot rebuilt with 28 waves
   ```

5. **Cooldown Active:**
   ```
   ⚠️ Build Lock Active: Last build 0.1m ago. Must wait 9.9m more.
   ```

---

## Accessibility Notes

- All status indicators use emoji + text for clarity
- Color coding: 🟢 Green (active), 🔴 Red (off), 🟡 Yellow (warning)
- Clear action labels on all buttons
- Disabled states are visually distinct
- Progress indicators during async operations
- Help text on checkboxes and buttons

---

## Responsive Behavior

- Run ID display: Always visible at top
- Sidebar controls: Persistent across tabs
- Build status: Context-sensitive to current tab
- Diagnostics: Collapsible to save space
- Mobile: All controls remain accessible

---

## Summary

The UI changes provide:
1. ✅ Clear visibility into app state
2. ✅ User control over build behavior
3. ✅ Comprehensive diagnostics
4. ✅ Visual feedback on all actions
5. ✅ Prevention of accidental infinite loops
