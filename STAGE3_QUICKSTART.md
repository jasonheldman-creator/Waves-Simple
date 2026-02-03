# Stage 3 Adaptive Intelligence - Quick Start Guide

## Overview
Stage 3 of the Adaptive Intelligence Center provides narrative and causal intelligence through signal clustering, change detection, and priority ranking. This guide shows you how to use the new features.

## Accessing Stage 3

1. Navigate to the **🧠 Adaptive Intelligence Center** tab in the application
2. The tab will automatically show the Stage 3 governance banner
3. Stage 3 features are displayed in three main sections

## UI Sections

### 1. Today's Intelligence Summary (Top Priority)
Located at the very top, this section shows the **top 3 most critical insights** requiring your attention.

**What you'll see:**
- **Rank badges** (#1, #2, #3)
- **Severity indicators**:
  - 🔴 Critical (75-100)
  - 🟠 High (50-74)
  - 🟡 Medium (25-49)
  - 🔵 Low (0-24)
- **Affected wave count**
- **Full narrative** explaining the issue
- **Priority justification** explaining why this is ranked where it is

**Example:**
```
🔴 #1: Regime Mismatch Cluster
Severity: Critical (90/100)
Affected Waves: 2
Priority Rank: #1 of 3

Regime Mismatch Alert: 2 waves operating in non-LIVE data regimes 
(1 in SANDBOX, 1 in UNAVAILABLE). These waves may be using stale, 
simulated, or unavailable data...

Ranked #1 due to: critical severity, affects multiple waves (2), 
highly persistent issue, data quality concern
```

### 2. Signal Clusters (Full Detail)
This section shows **all detected clusters** with full details.

**Features:**
- **Change summary** (if prior snapshot exists):
  - 🆕 New clusters
  - ⬆️ Escalating clusters (getting worse)
  - ⬇️ Improving clusters (getting better)
  - ✅ Resolved clusters (no longer an issue)

- **Expandable cluster cards**:
  - High-severity clusters (≥50) are auto-expanded
  - Low-severity clusters are collapsed by default
  - Each card includes:
    - Severity badge and change indicator
    - Wave count, severity score, persistence
    - Full narrative
    - List of affected waves

**Example:**
```
🟠 Beta Drift Cluster ⬆️ - 3 waves
▼ Click to expand/collapse

Severity: High (65/100)
Affected Waves: 3
Persistence: 70%

Narrative:
Beta Drift Detected: 3 waves showing tracking error vs target beta...

Affected Waves:
sp500_wave, tech_wave, crypto_l1_growth_wave

Change: Beta Drift Cluster escalating: severity ↑15 points, waves ↑1
```

### 3. Wave Health Monitor (Unchanged from Stage 2)
Standard wave health metrics table.

### 4. Regime Intelligence (Unchanged from Stage 2)
Volatility regime and alignment metrics.

### 5. Learning Signals (Unchanged from Stage 2)
Individual signal details grouped by severity.

## Understanding Cluster Types

Stage 3 creates five types of clusters:

### 1. Beta Drift Cluster
**What it means:** Waves are deviating from their target beta (market exposure)

**Why it matters:** Portfolio allocation may be different from what you intended

**What to do:** Review rebalancing thresholds, consider tactical adjustments

### 2. Regime Mismatch Cluster
**What it means:** Waves are using non-LIVE data (SANDBOX, UNAVAILABLE, HYBRID)

**Why it matters:** Data quality issues can lead to incorrect decisions

**What to do:** Verify data pipeline health, exclude waves until fixed

### 3. Alpha Decay Cluster
**What it means:** Multiple waves underperforming their benchmarks for 30+ days

**Why it matters:** Strategy may not be working or market conditions adverse

**What to do:** Review strategy assumptions, factor exposures, consider defensive positioning

### 4. Concentration Risk Cluster
**What it means:** Some waves have extreme exposure (>98% or <50%)

**Why it matters:** Either too much risk (high exposure) or underutilized capital (low exposure)

**What to do:** Review allocation strategy, adjust exposure levels

### 5. High Drawdown Cluster
**What it means:** Waves experiencing significant losses over 60 days

**Why it matters:** Extended drawdowns increase recovery time

**What to do:** Review risk management, consider stop-loss policies or hedging

## Understanding Change Indicators

When you refresh the page, Stage 3 compares the current state to the prior snapshot:

- **🆕 New**: Cluster just appeared (wasn't there before)
- **⬆️ Escalating**: Cluster got worse (severity increased ≥10 points or more waves affected)
- **⬇️ Improving**: Cluster got better (severity decreased or fewer waves affected)
- **✅ Resolved**: Cluster disappeared (issue no longer exists)

## Understanding Priority Ranking

The top 3 insights are ranked using a weighted formula:

**Components:**
1. **Severity (40%)**: How serious is the issue?
2. **Wave Count (30%)**: How many waves are affected?
3. **Regime Sensitivity (20%)**: Is this a data quality issue?
4. **Persistence (10%)**: How long has this been going on?

**Example Rankings:**
- **Regime mismatch with 8 waves**: Typically #1 (data quality + high wave count)
- **Beta drift with high severity**: Typically #2-3 (tracking concern)
- **Concentration risk with 2 waves**: Typically #3 (lower severity)

## Interpreting Narratives

All narratives are **template-based** (no AI/LLM), so they're:
- **Deterministic**: Same input = same narrative
- **Reproducible**: Can be audited
- **Transparent**: You can see exactly how conclusions were reached

**Narrative structure:**
1. **What**: Brief description of the cluster
2. **How many**: Number of affected waves and key metrics
3. **Why**: What this indicates about your portfolio
4. **What to do**: Suggested actions or considerations

## Best Practices

### Daily Workflow
1. **Check Today's Intelligence Summary first** - This tells you what matters most
2. **Review change indicators** - See what's new or changing
3. **Expand high-severity clusters** - Understand the details
4. **Check affected waves list** - Know which specific waves need attention
5. **Read narratives** - Understand the "why" behind each cluster

### When to Act
- **Critical (🔴) severity**: Review immediately, may require intervention
- **High (🟠) severity**: Monitor closely, action may be needed soon
- **Medium (🟡) severity**: Watch for escalation
- **Low (🔵) severity**: Informational only

### Change Detection Tips
- **New clusters**: Investigate why they appeared
- **Escalating clusters**: Determine if trend is likely to continue
- **Improving clusters**: Verify improvements are sustainable
- **Resolved clusters**: Confirm issue is truly fixed, not just data lag

## Troubleshooting

### "No clusters detected"
This is actually good news! It means all waves are operating within normal parameters.

### "All clusters marked as new"
This happens on first load when there's no prior snapshot. This is expected.

### "Priority insights showing same clusters as before"
If the same issues persist, they'll keep appearing. Check the change indicators - they might be "improving" even if still present.

### "Change detection shows no changes"
If nothing changed between refreshes, this is correct behavior.

## Technical Notes

### Data Source
All clusters are derived from TruthFrame data only. No external data sources.

### Read-Only Compliance
Stage 3 never modifies:
- Trading logic
- Portfolio parameters
- Execution behavior
- Benchmark definitions
- TruthFrame data

### Deterministic Behavior
Stage 3 calculations are fully deterministic:
- No randomness
- No LLM/AI models
- Same input always produces same output
- Reproducible for audits

### Session State
The prior snapshot is stored in `st.session_state['ai_prior_snapshot']` and persists across tab switches but resets on page refresh.

## API Usage (For Developers)

If you're integrating Stage 3 programmatically:

```python
from analytics_truth import get_truth_frame
from adaptive_intelligence import get_adaptive_intelligence_snapshot

# Load TruthFrame
truth_df = get_truth_frame(safe_mode=True)

# Get full snapshot (includes Stage 2 + Stage 3)
snapshot = get_adaptive_intelligence_snapshot(truth_df, prior_snapshot=None)

# Access Stage 3 features
clusters = snapshot['signal_clusters']
changes = snapshot['cluster_changes']
insights = snapshot['priority_insights']

# Individual functions also available
from adaptive_intelligence import cluster_signals, detect_cluster_changes, get_priority_insights

signals = detect_learning_signals(truth_df)
clusters = cluster_signals(signals, truth_df)
changes = detect_cluster_changes(clusters, prior_clusters)
insights = get_priority_insights(clusters)
```

## FAQ

**Q: How often should I check the Adaptive Intelligence tab?**
A: Daily is recommended. The intelligence updates as your TruthFrame data updates.

**Q: Can I export the cluster information?**
A: Not in Stage 3. This may be added in future stages.

**Q: Will Stage 3 make trading decisions for me?**
A: No. Stage 3 is strictly read-only monitoring. All decisions remain with you.

**Q: Are the narratives generated by AI?**
A: No. All narratives are template-based, deterministic, and fully reproducible.

**Q: What's the difference between signals and clusters?**
A: Signals are individual issues (e.g., "Wave A has beta drift"). Clusters group related signals into broader themes (e.g., "Beta Drift Cluster affecting 3 waves").

**Q: Why do some clusters auto-expand and others don't?**
A: High-severity clusters (≥50) auto-expand so you don't miss critical issues. Lower-severity clusters are collapsed to reduce noise.

**Q: Can I customize the priority ranking weights?**
A: Not in Stage 3. This may be configurable in future stages.

## Next Steps

- Review the full documentation: `STAGE3_ADAPTIVE_INTELLIGENCE_SUMMARY.md`
- Run the test suite: `python test_adaptive_intelligence_stage3.py`
- Explore the code: `adaptive_intelligence.py` (functions are well-documented)

## Support

For issues or questions:
1. Check this guide first
2. Review the full documentation
3. Check the test suite for examples
4. Review the code comments in `adaptive_intelligence.py`
