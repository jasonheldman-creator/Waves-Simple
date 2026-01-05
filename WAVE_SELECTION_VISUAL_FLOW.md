# Wave Selection Context Resolver - Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WAVE SELECTION FLOW (NEW)                        │
└─────────────────────────────────────────────────────────────────────┘

1. USER INTERACTION
   ┌───────────────┐
   │  Sidebar UI   │
   │  Selectbox    │
   │  "Gold Wave"  │
   └───────┬───────┘
           │
           ▼
   
2. STATE UPDATE (Single authoritative key)
   ┌──────────────────────────────────────┐
   │  st.session_state["selected_wave_id"]│
   │  = "wave_gold"                       │
   │                                      │
   │  ✅ ONLY wave_id stored              │
   │  ❌ Display name NOT stored          │
   └──────────────────┬───────────────────┘
                      │
                      ▼
   
3. CONTEXT RESOLUTION (Canonical resolver)
   ┌──────────────────────────────────────┐
   │  ctx = resolve_app_context()         │
   │                                      │
   │  Returns:                            │
   │  {                                   │
   │    'selected_wave_id': 'wave_gold', │
   │    'selected_wave_name': 'Gold Wave'│
   │    'mode': 'Standard',               │
   │    'context_key': 'Standard:wave_gold'│
   │  }                                   │
   └──────────────────┬───────────────────┘
                      │
                      ▼
   
4. RENDER PIPELINE (Uses context everywhere)
   ┌──────────────────────────────────────┐
   │  render_banner(ctx["selected_wave_name"])│
   │  render_tabs(ctx["selected_wave_id"])│
   │  cache.get(ctx["context_key"])       │
   └──────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    RERUN LOOP PREVENTION                             │
└─────────────────────────────────────────────────────────────────────┘

OLD (Could cause infinite loops):
   ┌─────────────────┐
   │ Widget changes  │──▶ State update ──▶ Widget re-renders ──┐
   └─────────────────┘                                          │
           ▲                                                    │
           └────────────────────────────────────────────────────┘
                         LOOP! ⚠️

NEW (Prevents loops):
   ┌──────────────────────┐
   │ Widget changes       │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ State change detection:          │
   │ if current != new:               │
   │     update state                 │
   │ else:                            │
   │     skip update ✅                │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Widget renders once  │
   │ No loop! ✅          │
   └──────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    CACHE KEY NORMALIZATION                           │
└─────────────────────────────────────────────────────────────────────┘

Format: {mode}:{wave_id or 'PORTFOLIO'}

Examples:
  Portfolio View:       "Standard:PORTFOLIO"
  Gold Wave:            "Standard:wave_gold"
  Income Wave:          "Standard:wave_income"
  Aggressive Mode:      "Aggressive:wave_gold"

Benefits:
  ✅ Consistent caching across reruns
  ✅ Proper cache invalidation
  ✅ Context isolation
  ✅ Easy debugging


┌─────────────────────────────────────────────────────────────────────┐
│                    STATE ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────┘

SESSION STATE (st.session_state):
  ┌─────────────────────────────────┐
  │ selected_wave_id: "wave_gold"   │ ◀─ AUTHORITATIVE (stored)
  │ mode: "Standard"                │ ◀─ AUTHORITATIVE (stored)
  └─────────────────────────────────┘

DERIVED (via resolve_app_context()):
  ┌─────────────────────────────────┐
  │ selected_wave_name: "Gold Wave" │ ◀─ DERIVED (not stored)
  │ context_key: "Standard:wave_gold"│ ◀─ DERIVED (not stored)
  └─────────────────────────────────┘

PRINCIPLE: Store IDs, derive names
  ✅ Single source of truth
  ✅ No sync issues
  ✅ Always consistent


┌─────────────────────────────────────────────────────────────────────┐
│                    KEY SAFEGUARDS                                    │
└─────────────────────────────────────────────────────────────────────┘

1. UNIQUE WIDGET KEYS
   wave_selector_unique_key ──▶ Prevents widget ID conflicts

2. STATE CHANGE DETECTION
   if old_value != new_value: ──▶ Only update when changed
       update_state()

3. AUTO-REFRESH OFF
   DEFAULT_AUTO_REFRESH_ENABLED = False ──▶ No automatic reruns

4. RUN GUARD COUNTER
   if run_count > 3: ──▶ Stop infinite loops
       halt_execution()

5. CONTEXT RESOLVER
   ctx = resolve_app_context() ──▶ Single call per render
   (not called multiple times)


┌─────────────────────────────────────────────────────────────────────┐
│                    BEFORE vs AFTER                                   │
└─────────────────────────────────────────────────────────────────────┘

BEFORE:
  • Multiple state keys (selected_wave, selected_wave_display_name)
  • Display names stored in state (sync issues)
  • No change detection (unnecessary updates)
  • Auto-refresh ON by default (infinite loops)
  • Non-unique widget keys (conflicts)
  • Context scattered across codebase

AFTER:
  • Single state key (selected_wave_id)
  • Display names derived (no sync issues)
  • Change detection (prevents updates)
  • Auto-refresh OFF by default (no loops)
  • Unique widget keys (no conflicts)
  • Context centralized (resolve_app_context)

RESULT: 
  ✅ Stable
  ✅ Predictable
  ✅ Maintainable
  ✅ No infinite loops
```
