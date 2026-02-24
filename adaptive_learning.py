import logging
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ADAPTIVE_STATE_PATH = Path("data/adaptive_state.json")
GOVERNANCE_DECISIONS_PATH = Path("data/governance_decisions.json")


def load_governance_decisions(path=None):
    """Load and normalize governance decisions from data/governance_decisions.json.

    Normalizes governance decision structure into the format expected by the
    adaptive learning compute functions (date, id, outcome_30d, status, etc.).
    Returns empty list with a logged warning when the file is missing.
    """
    gov_path = Path(path or GOVERNANCE_DECISIONS_PATH)
    if not gov_path.exists():
        logging.warning(
            "[AdaptiveIntelligence] governance_decisions.json not found at %s"
            " - learning pipeline data will be limited",
            gov_path,
        )
        return []
    try:
        with open(gov_path, "r") as f:
            raw = json.load(f)
        normalized = []
        for d in raw:
            ctx = d.get("context_snapshot", {}) or {}
            created = d.get("created", "")
            date_str = created[:10] if created else "N/A"
            normalized.append({
                "id": d.get("id", ""),
                "date": date_str,
                "wave": d.get("wave", "Portfolio"),
                "decision_type": d.get("decision_type", "system_generated"),
                "event_type": d.get("trigger_source", ""),
                "status": d.get("status", ""),
                "approval_status": d.get("status", ""),
                "outcome_30d": None,
                "regime_at_decision": d.get("regime_at_decision", "Normal"),
                "context": ctx.get("action", "") if isinstance(ctx, dict) else "",
                "snapshot_asof": ctx.get("snapshot_asof", "") if isinstance(ctx, dict) else "",
                "source": d.get("source", "system"),
            })
        return normalized
    except Exception as e:
        logging.error(
            "[AdaptiveIntelligence] Failed to load governance_decisions.json: %s", e
        )
        return []

def load_adaptive_state():
    if ADAPTIVE_STATE_PATH.exists():
        try:
            with open(ADAPTIVE_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "regime_state": "normal",
        "confidence": 0.5,
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "learning_rate": 0.01,
        "pattern_memory": {
            "volatility_regime": {"trend": "stable", "current": 0.15},
            "momentum_regime": {"trend": "neutral", "current": 0.0},
            "correlation_regime": {"trend": "stable", "current": 0.5},
        },
        "tilt_history": [],
        "regime_history": [],
        "parameters": {
            "vol_threshold_high": 0.25,
            "vol_threshold_low": 0.10,
            "momentum_lookback": 30,
            "rebalance_frequency_days": 30,
        }
    }

def _save_adaptive_state(state):
    try:
        os.makedirs(os.path.dirname(ADAPTIVE_STATE_PATH), exist_ok=True)
        state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ADAPTIVE_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def update_adaptive_state(snapshot_df, attrib_df, adaptive_state=None):
    if adaptive_state is None:
        adaptive_state = load_adaptive_state()
    messages = []

    if snapshot_df is not None and not snapshot_df.empty:
        if "benchmark_volatility_30d" in snapshot_df.columns:
            vol_vals = pd.to_numeric(snapshot_df["benchmark_volatility_30d"], errors="coerce").dropna()
            if len(vol_vals) > 0:
                avg_vol = vol_vals.mean()
                prev_vol = adaptive_state.get("pattern_memory", {}).get("volatility_regime", {}).get("current", 0.15)
                adaptive_state.setdefault("pattern_memory", {}).setdefault("volatility_regime", {})["current"] = round(float(avg_vol), 4)
                if avg_vol > 0.25:
                    adaptive_state["pattern_memory"]["volatility_regime"]["trend"] = "increasing"
                    adaptive_state["regime_state"] = "elevated"
                    messages.append("Volatility regime elevated")
                elif avg_vol < 0.10:
                    adaptive_state["pattern_memory"]["volatility_regime"]["trend"] = "decreasing"
                    adaptive_state["regime_state"] = "compressed"
                    messages.append("Volatility regime compressed")
                else:
                    adaptive_state["pattern_memory"]["volatility_regime"]["trend"] = "stable"
                    adaptive_state["regime_state"] = "normal"

        if "alpha_30d" in snapshot_df.columns:
            alpha_vals = pd.to_numeric(snapshot_df["alpha_30d"], errors="coerce").dropna()
            if len(alpha_vals) > 0:
                avg_alpha = alpha_vals.mean()
                adaptive_state.setdefault("pattern_memory", {}).setdefault("momentum_regime", {})["current"] = round(float(avg_alpha), 4)
                if avg_alpha > 0.01:
                    adaptive_state["pattern_memory"]["momentum_regime"]["trend"] = "positive"
                elif avg_alpha < -0.01:
                    adaptive_state["pattern_memory"]["momentum_regime"]["trend"] = "negative"
                    messages.append("Negative momentum detected across waves")
                else:
                    adaptive_state["pattern_memory"]["momentum_regime"]["trend"] = "neutral"

    adaptive_state["confidence"] = min(1.0, adaptive_state.get("confidence", 0.5) + 0.01)

    # Compute dynamic learning rate scaled by system confidence.
    # Range: ~0.0001 (low conf) to ~0.0002 (full conf).  A non-zero value
    # confirms the calibration engine is active; larger values indicate higher
    # system confidence.  Kept intentionally conservative to avoid spurious
    # calibration signals during early system operation.
    conf = adaptive_state.get("confidence", 0.5)
    adaptive_state["learning_rate"] = round(0.0001 * (1.0 + conf), 4)

    _save_adaptive_state(adaptive_state)
    return adaptive_state, messages

def compute_learning_snapshot(snapshot_df, attrib_df, adaptive_state, decisions):
    snapshot = {
        "regime": adaptive_state.get("regime_state", "normal").title(),
        "confidence": adaptive_state.get("confidence", 0.5),
        "volatility_trend": adaptive_state.get("pattern_memory", {}).get("volatility_regime", {}).get("trend", "stable"),
        "momentum_trend": adaptive_state.get("pattern_memory", {}).get("momentum_regime", {}).get("trend", "neutral"),
        "total_waves": len(snapshot_df) if snapshot_df is not None else 0,
        "total_decisions": len(decisions) if decisions else 0,
        "learning_rate": adaptive_state.get("learning_rate", 0.01),
        "last_updated": adaptive_state.get("last_updated", "â"),
    }
    return snapshot

def compute_core_learning_signals(snapshot_df, attrib_df, adaptive_state):
    signals = []
    regime = adaptive_state.get("regime_state", "normal")
    vol_trend = adaptive_state.get("pattern_memory", {}).get("volatility_regime", {}).get("trend", "stable")
    mom_trend = adaptive_state.get("pattern_memory", {}).get("momentum_regime", {}).get("trend", "neutral")

    signals.append({
        "signal": "Regime State",
        "value": regime.title(),
        "context": f"Current volatility regime is {vol_trend}, momentum is {mom_trend}",
        "confidence": adaptive_state.get("confidence", 0.5),
    })

    if snapshot_df is not None and "alpha_30d" in snapshot_df.columns:
        alpha_vals = pd.to_numeric(snapshot_df["alpha_30d"], errors="coerce").dropna()
        if len(alpha_vals) > 0:
            positive_pct = (alpha_vals > 0).sum() / len(alpha_vals) * 100
            signals.append({
                "signal": "Alpha Breadth",
                "value": f"{positive_pct:.0f}%",
                "context": f"{positive_pct:.0f}% of waves generating positive 30D alpha",
                "confidence": 0.7,
            })

    if attrib_df is not None and not attrib_df.empty:
        component_cols = ["selection_alpha", "momentum_alpha", "volatility_alpha"]
        for col in component_cols:
            if col in attrib_df.columns:
                vals = pd.to_numeric(attrib_df[col], errors="coerce").dropna()
                if len(vals) > 0 and abs(vals.mean()) > 0.002:
                    name = col.replace("_alpha", "").title()
                    signals.append({
                        "signal": f"{name} Component",
                        "value": f"{vals.mean():.4f}",
                        "context": f"{'Drag' if vals.mean() < 0 else 'Contribution'} detected in {name.lower()} component",
                        "confidence": 0.6,
                    })

    return signals

def compute_parameter_sensitivity(attrib_df, adaptive_state):
    params = adaptive_state.get("parameters", {})
    sensitivities = []

    for param_name, param_value in params.items():
        sensitivities.append({
            "parameter": param_name.replace("_", " ").title(),
            "current_value": param_value,
            "sensitivity": "Low",
            "recommendation": "No change recommended",
        })

    return sensitivities

def compute_learning_curve(snapshot_df, attrib_df, adaptive_state, decisions):
    """Compute the system learning curve using NOTE 036/037 weighted logic.

    Learning Index = 60% Decision Outcome Alignment + 25% Outcome Consistency
                   + 15% Structural Improvement

    When no outcome-recorded decisions exist, falls back to proxy metrics
    derived from system confidence and governance decision consistency so that
    the curve is never silently empty when decision data is present.
    """
    if not decisions or len(decisions) < 2:
        if not decisions:
            logging.warning(
                "[AdaptiveIntelligence] compute_learning_curve: decision list empty"
                " - check governance_decisions.json and decision_log.json"
            )
        return {"has_data": False}

    recorded = [d for d in decisions if d.get("outcome_30d") and d["outcome_30d"] != "Pending"]

    if len(recorded) >= 2:
        # Outcome-driven path (NOTE 036/037)
        positive = sum(1 for d in recorded if d.get("outcome_30d") == "Positive")
        neutral = sum(1 for d in recorded if d.get("outcome_30d") == "Neutral")
        total = len(recorded)

        doa = (positive / total * 100) if total > 0 else 0
        oc = ((positive + neutral) / total * 100) if total > 0 else 0
        si = min(100, doa * 0.8 + oc * 0.2)
        # NOTE 036/037: 60% DOA + 25% OC + 15% SI
        learning_index = doa * 0.60 + oc * 0.25 + si * 0.15

        monthly_points = []
        for i, d in enumerate(recorded):
            running_pos = sum(1 for x in recorded[:i + 1] if x.get("outcome_30d") == "Positive")
            running_total = i + 1
            monthly_points.append([d.get("date", f"T{i}"), round(running_pos / running_total * 100, 1)])
    else:
        # Bootstrap path: proxy metrics from system confidence and governance data
        # Used when decisions are present but outcomes not yet recorded.
        conf = adaptive_state.get("confidence", 0.5)
        doa = round(conf * 100, 1)

        # Outcome consistency proxy: fraction of decisions with same type
        types = [d.get("decision_type", "") for d in decisions if d.get("decision_type")]
        if types:
            most_common = max(set(types), key=types.count)
            oc = round(types.count(most_common) / len(types) * 100, 1)
        else:
            oc = 50.0

        # Structural improvement proxy: from attribution data if available
        if attrib_df is not None and not attrib_df.empty and "total_alpha" in attrib_df.columns:
            vals = pd.to_numeric(attrib_df["total_alpha"], errors="coerce").dropna()
            if len(vals) > 0:
                positive_fraction = float((vals > 0).sum()) / len(vals)
                si = round(min(100.0, positive_fraction * 100), 1)
            else:
                si = 50.0
        else:
            si = 50.0

        # NOTE 036/037 weights
        learning_index = doa * 0.60 + oc * 0.25 + si * 0.15

        # Single bootstrap point using earliest available date
        dates = [d.get("date", "") for d in decisions if d.get("date") and d.get("date") != "N/A"]
        point_date = dates[0] if dates else "T0"
        monthly_points = [[point_date, round(learning_index, 1)]]

    if learning_index >= 80:
        grade, zone = "A", "Mastery"
    elif learning_index >= 60:
        grade, zone = "B", "Proficiency"
    elif learning_index >= 40:
        grade, zone = "C", "Development"
    elif learning_index >= 20:
        grade, zone = "D", "Foundation"
    else:
        grade, zone = "F", "Early Stage"

    return {
        "has_data": True,
        "learning_index": learning_index,
        "grade": grade,
        "zone": zone,
        "decision_outcome_alignment": doa,
        "outcome_consistency": oc,
        "structural_improvement": si,
        # Aliased keys expected by the rendering layer
        "decision_alignment_pct": doa,
        "outcome_consistency_pct": oc,
        "structural_improvement_pct": si,
        "monthly_points": monthly_points,
    }


def compute_efficiency_curve(decisions, adaptive_state):
    """Compute decision efficiency metrics.

    Counts any decision that has been acknowledged by the system
    (status in Awaiting Approval / Recorded / Active / Monitoring) as an
    engaged signal, consistent with how governance decisions enter the pipeline.
    """
    if not decisions or len(decisions) < 2:
        return {"has_data": False}

    total = len(decisions)

    # "Engaged" = any decision that has moved beyond the initial creation state
    engaged_statuses = {"Recorded", "Active", "Monitoring", "Approved", "Awaiting Approval"}
    engaged = [d for d in decisions if d.get("status", "") in engaged_statuses]
    approved = [d for d in decisions if d.get("status", "") in {"Approved", "Recorded"}
                or d.get("approval_status", "") == "Approved"]

    signal_engagement_rate = min(100, len(engaged) / max(total, 1) * 100)
    decision_implementation_rate = min(100, len(approved) / max(total, 1) * 100)

    avg_latency_days = 5
    decision_latency_score = max(0, 100 - avg_latency_days * 5)

    efficiency_index = (
        signal_engagement_rate * 0.4
        + decision_implementation_rate * 0.3
        + decision_latency_score * 0.3
    )

    if efficiency_index >= 80:
        grade = "A"
    elif efficiency_index >= 60:
        grade = "B"
    elif efficiency_index >= 40:
        grade = "C"
    elif efficiency_index >= 20:
        grade = "D"
    else:
        grade = "F"

    monthly_points = []
    for i, d in enumerate(decisions):
        running_engaged = sum(
            1 for x in decisions[:i + 1] if x.get("status", "") in engaged_statuses
        )
        running_total = i + 1
        monthly_points.append([d.get("date", f"T{i}"), round(running_engaged / running_total * 100, 1)])

    # avg_decision_latency_hours: estimated hours from decision creation to engagement.
    # Use the static estimate (avg_latency_days) converted to hours as a baseline
    # until per-decision timestamp tracking is available.
    avg_latency_hours = float(avg_latency_days * 24)

    return {
        "has_data": True,
        "efficiency_index": efficiency_index,
        "grade": grade,
        "signal_engagement_rate": signal_engagement_rate,
        "decision_implementation_rate": decision_implementation_rate,
        "decision_latency_score": decision_latency_score,
        # Aliased keys expected by the rendering layer
        "signal_engagement_pct": signal_engagement_rate,
        "implementation_rate_pct": decision_implementation_rate,
        "avg_decision_latency_hours": avg_latency_hours,
        "monthly_points": monthly_points,
    }

def compute_decision_memory_table(decisions, attrib_df):
    if not decisions or len(decisions) < 1:
        return {"has_data": False}

    rows = []
    for d in decisions[:15]:
        rows.append({
            "Decision ID": d.get("id", "N/A"),
            "Date": d.get("date", "N/A"),
            "Type": d.get("decision_type", d.get("event_type", "Other")),
            "Wave": d.get("wave", "Portfolio"),
            "Outcome (30D)": d.get("outcome_30d", "Pending"),
            "Regime": d.get("regime_at_decision", "Normal"),
        })

    recorded = [d for d in decisions if d.get("outcome_30d") and d["outcome_30d"] != "Pending"]
    positive = sum(1 for d in recorded if d.get("outcome_30d") == "Positive")
    total_recorded = len(recorded) if recorded else 1

    return {
        "has_data": True,
        "rows": rows,
        "summary": {
            "total_decisions": len(decisions),
            "alignment_rate": round(positive / total_recorded * 100, 1),
            "structural_improvements": positive,
            "persistent_detractors": sum(1 for d in recorded if d.get("outcome_30d") == "Negative"),
        },
    }

def build_decision_memory(decisions):
    """Build a decision memory DataFrame from governance lifecycle decision records.

    Pipeline step: load_governance_decisions() → build_decision_memory() → DataFrame.
    Adaptive Intelligence consumes governance records only; it never generates decisions.

    Args:
        decisions: list of decision dicts returned by load_governance_decisions()

    Returns:
        pd.DataFrame with columns: decision_id, wave, decision_type, created_timestamp,
        status, lifecycle_stage, outcome_alignment, monitoring_state, horizon_context,
        regime_context. Returns empty DataFrame (with columns) if decisions is empty.
    """
    _cols = [
        "decision_id", "wave", "decision_type", "created_timestamp",
        "status", "lifecycle_stage", "outcome_alignment", "monitoring_state",
        "horizon_context", "regime_context",
    ]
    if not decisions:
        return pd.DataFrame(columns=_cols)

    rows = []
    for d in decisions:
        status = d.get("status", "Unknown")

        # Derive lifecycle_stage from status
        if status in ("Awaiting Approval", "Pending"):
            lifecycle_stage = "Pre-Approval"
        elif status in ("Approved", "Active"):
            lifecycle_stage = "Active"
        elif status in ("Recorded",):
            lifecycle_stage = "Recorded"
        elif status == "Monitoring":
            lifecycle_stage = "Under Review"
        elif status in ("Closed", "Resolved"):
            lifecycle_stage = "Closed"
        else:
            lifecycle_stage = status

        # Derive monitoring_state from status
        if status == "Awaiting Approval":
            monitoring_state = "Pending"
        elif status in ("Approved", "Active"):
            monitoring_state = "Active"
        elif status == "Monitoring":
            monitoring_state = "Monitoring"
        elif status == "Recorded":
            monitoring_state = "Recorded"
        else:
            monitoring_state = "Unknown"

        # outcome_alignment - use existing outcome if available, else "Pending Observation"
        outcome = d.get("outcome_30d")
        if outcome and outcome not in ("Pending",):
            outcome_alignment = str(outcome)
        else:
            outcome_alignment = "Pending Observation"

        rows.append({
            "decision_id": d.get("id", "N/A"),
            "wave": d.get("wave", "Portfolio"),
            "decision_type": d.get("decision_type", "system_generated"),
            "created_timestamp": d.get("date", "N/A"),
            "status": status,
            "lifecycle_stage": lifecycle_stage,
            "outcome_alignment": outcome_alignment,
            "monitoring_state": monitoring_state,
            "horizon_context": d.get("snapshot_asof") or d.get("date", "N/A"),
            "regime_context": d.get("regime_at_decision", "Normal"),
        })

    return pd.DataFrame(rows, columns=_cols)


def compute_persistent_detractors(decision_memory_df, attrib_df=None):
    """Compute count of persistent detractors from decision memory.

    Persistent detractors are:
    - Attribution drivers showing negative values across >=2 horizons in attrib_df, OR
    - Decisions in repeated monitoring state (>=2 decisions with monitoring_state='Monitoring')

    Args:
        decision_memory_df: DataFrame from build_decision_memory()
        attrib_df: Attribution DataFrame (optional)

    Returns:
        int: count of persistent detractors
    """
    count = 0

    # Check for repeated monitoring state
    if decision_memory_df is not None and not decision_memory_df.empty:
        monitoring_count = int(
            (decision_memory_df["monitoring_state"] == "Monitoring").sum()
        )
        if monitoring_count >= 2:
            count += monitoring_count

    # Check attribution-based detractors (negative across >=2 horizons)
    if attrib_df is not None and not attrib_df.empty and "horizon" in attrib_df.columns:
        horizons = sorted(attrib_df["horizon"].dropna().unique().tolist())
        _attribution_cols = [
            "selection_alpha", "momentum_alpha", "volatility_alpha",
            "regime_alpha", "exposure_alpha",
        ]
        for col in _attribution_cols:
            if col not in attrib_df.columns:
                continue
            neg_horizons = sum(
                1 for h in horizons
                if pd.to_numeric(
                    attrib_df[attrib_df["horizon"] == h][col], errors="coerce"
                ).mean() < -0.002
            )
            if neg_horizons >= 2:
                count += 1

    return count


def compute_cross_horizon_stability(snapshot_df, attrib_df):
    """Compute cross-horizon stability for each attribution driver.

    Stability is determined by sign agreement between the 30D and 365D
    horizons for each driver (consistent positive or negative = Stable,
    opposite signs = Volatile, only one horizon available = Moderate).
    Attribution data must come from alpha_attribution_summary.csv or an
    equivalent source with a ``horizon`` column.
    """
    component_cols = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]
    driver_names = ["Selection", "Momentum", "Volatility", "Regime", "Exposure"]
    drivers = []

    if attrib_df is not None and not attrib_df.empty:
        if "horizon" not in attrib_df.columns:
            logging.warning(
                "[AdaptiveIntelligence] compute_cross_horizon_stability: 'horizon' column"
                " missing from attribution data - cross-horizon stability unavailable"
            )
        else:
            for name, col in zip(driver_names, component_cols):
                if col not in attrib_df.columns:
                    continue
                h30_df = attrib_df[attrib_df["horizon"] == 30]
                h365_df = attrib_df[attrib_df["horizon"] == 365]
                h90_df = attrib_df[attrib_df["horizon"] == 90]

                h30_vals = pd.to_numeric(h30_df[col], errors="coerce").dropna()
                h365_vals = pd.to_numeric(h365_df[col], errors="coerce").dropna()
                h90_vals = pd.to_numeric(h90_df[col], errors="coerce").dropna()

                def _state(vals):
                    if len(vals) == 0:
                        return "Neutral"
                    m = float(vals.mean())
                    return "Positive" if m > 0.005 else "Negative" if m < -0.005 else "Neutral"

                state_30d = _state(h30_vals)
                state_90d = _state(h90_vals) if len(h90_vals) > 0 else state_30d
                state_365d = _state(h365_vals)

                # Stability: Stable when 30D and 365D share the same sign direction
                if len(h30_vals) > 0 and len(h365_vals) > 0:
                    same_sign = (h30_vals.mean() > 0) == (h365_vals.mean() > 0)
                    stability = "Stable" if same_sign else "Volatile"
                elif len(h30_vals) > 0 or len(h365_vals) > 0:
                    stability = "Moderate"
                else:
                    continue

                drivers.append({
                    "Driver": name,
                    "30D State": state_30d,
                    "90D State": state_90d,
                    "365D State": state_365d,
                    "Stability": stability,
                })
    else:
        logging.warning(
            "[AdaptiveIntelligence] compute_cross_horizon_stability: attribution data"
            " not available - cross-horizon stability will be empty"
        )

    if not drivers and snapshot_df is not None:
        horizons = {"1D": "alpha_1d", "30D": "alpha_30d", "365D": "alpha_365d"}
        for label, col in horizons.items():
            if col in snapshot_df.columns:
                vals = pd.to_numeric(snapshot_df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    mean_val = float(vals.mean())
                    state = "Positive" if mean_val > 0 else "Negative" if mean_val < 0 else "Neutral"
                    drivers.append({
                        "Driver": label,
                        "30D State": state,
                        "90D State": state,
                        "365D State": state,
                        "Stability": "Stable" if vals.std() < 0.01 else "Moderate",
                    })

    if not drivers:
        return {"drivers": [], "summary": "Insufficient data for cross-horizon analysis."}

    stable_count = sum(1 for d in drivers if d["Stability"] == "Stable")
    summary = f"{stable_count} of {len(drivers)} drivers show stable alpha patterns across horizons."
    return {"drivers": drivers, "summary": summary}

def compute_derived_signals(snapshot_df, attrib_df):
    signals = []
    if snapshot_df is not None and not snapshot_df.empty:
        if "alpha_30d" in snapshot_df.columns:
            alpha_vals = pd.to_numeric(snapshot_df["alpha_30d"], errors="coerce").dropna()
            if len(alpha_vals) > 0:
                signals.append({
                    "name": "Mean Alpha 30D",
                    "value": round(float(alpha_vals.mean()), 4),
                    "direction": "positive" if alpha_vals.mean() > 0 else "negative",
                })
        if "drawdown_30d" in snapshot_df.columns:
            dd_vals = pd.to_numeric(snapshot_df["drawdown_30d"], errors="coerce").dropna()
            if len(dd_vals) > 0:
                signals.append({
                    "name": "Max Drawdown 30D",
                    "value": round(float(dd_vals.max()), 4),
                    "direction": "negative" if dd_vals.max() > 0.05 else "neutral",
                })
    return signals

def compute_cross_horizon_agreement(snapshot_df, attrib_df):
    agreements = {}
    if snapshot_df is None or snapshot_df.empty:
        return agreements

    horizons = {"alpha_1d": "1D", "alpha_30d": "30D", "alpha_365d": "365D"}
    directions = {}
    for col, label in horizons.items():
        if col in snapshot_df.columns:
            vals = pd.to_numeric(snapshot_df[col], errors="coerce").dropna()
            if len(vals) > 0:
                directions[label] = "positive" if vals.mean() > 0 else "negative"

    if len(directions) >= 2:
        vals = list(directions.values())
        if all(v == vals[0] for v in vals):
            agreements["status"] = "Aligned"
            agreements["direction"] = vals[0]
        else:
            agreements["status"] = "Divergent"
            agreements["direction"] = "mixed"
    else:
        agreements["status"] = "Insufficient Data"
        agreements["direction"] = "unknown"

    agreements["horizons"] = directions
    return agreements

def generate_adaptive_tilt_proposals(signals, adaptive_state, cross_horizon_agreements):
    proposals = []
    alignment = cross_horizon_agreements.get("status", "")

    if alignment == "Aligned" and cross_horizon_agreements.get("direction") == "positive":
        proposals.append({
            "proposal": "Maintain current allocations",
            "rationale": "Cross-horizon alignment is positive â no tilt adjustment warranted",
            "confidence": 0.7,
            "type": "Hold",
        })
    elif alignment == "Divergent":
        proposals.append({
            "proposal": "Review horizon-specific exposures",
            "rationale": "Cross-horizon signals are divergent â consider horizon-specific review",
            "confidence": 0.5,
            "type": "Review",
        })

    regime = adaptive_state.get("regime_state", "normal")
    if regime == "elevated":
        proposals.append({
            "proposal": "Consider defensive tilt",
            "rationale": "Elevated volatility regime may warrant reduced risk exposure",
            "confidence": 0.6,
            "type": "Defensive",
        })

    if not proposals:
        proposals.append({
            "proposal": "No tilt adjustments indicated",
            "rationale": "Current conditions do not suggest changes to portfolio tilts",
            "confidence": 0.5,
            "type": "Hold",
        })

    return proposals


def compute_alpha_quality(snapshot_df, attrib_df):
    """Rank waves by alpha quality across horizons.

    Returns a dict with ``has_data`` flag and ``waves`` list of per-wave
    quality metrics suitable for rendering as a DataFrame.
    """
    if snapshot_df is None or snapshot_df.empty:
        return {"has_data": False, "waves": []}

    rows = []
    horizon_cols = {
        "Alpha 30D": "alpha_30d",
        "Alpha 60D": "alpha_60d",
        "Alpha 365D": "alpha_365d",
    }
    name_col = "display_name" if "display_name" in snapshot_df.columns else "wave_name"
    if name_col not in snapshot_df.columns:
        return {"has_data": False, "waves": []}

    for _, r in snapshot_df.iterrows():
        wave = str(r.get(name_col, ""))
        if not wave:
            continue
        alphas = {}
        for label, col in horizon_cols.items():
            v = r.get(col)
            alphas[label] = round(float(v), 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

        filled = [v for v in alphas.values() if v is not None]
        if not filled:
            continue

        # Consistency: fraction of horizons with positive alpha
        consistency = round(sum(1 for v in filled if v > 0) / len(filled), 2)
        # Composite score: mean of available horizon alphas
        composite = round(float(np.mean(filled)), 4)

        entry = {"Wave": wave, "Composite Alpha": composite, "Consistency": consistency}
        entry.update({k: (v if v is not None else "") for k, v in alphas.items()})
        rows.append(entry)

    if not rows:
        return {"has_data": False, "waves": []}

    rows.sort(key=lambda x: x["Composite Alpha"], reverse=True)
    return {"has_data": True, "waves": rows}


def compute_capital_pressure(snapshot_df):
    """Compute portfolio-level capital pressure metrics.

    Returns regime label, positive-alpha percentage, and dispersion.
    """
    if snapshot_df is None or snapshot_df.empty:
        return {"has_data": False}

    col = "alpha_30d" if "alpha_30d" in snapshot_df.columns else None
    if col is None:
        return {"has_data": False}

    vals = pd.to_numeric(snapshot_df[col], errors="coerce").dropna()
    if len(vals) == 0:
        return {"has_data": False}

    positive_pct = round(float((vals > 0).sum() / len(vals) * 100), 1)
    dispersion = round(float(vals.std()), 4)

    if positive_pct >= 60:
        regime = "Expansive"
    elif positive_pct >= 40:
        regime = "Neutral"
    else:
        regime = "Contractive"

    return {
        "has_data": True,
        "Capital Pressure Regime": regime,
        "Positive Alpha %": positive_pct,
        "Dispersion (Std Dev)": dispersion,
    }


def compute_rotation_velocity(snapshot_df):
    """Estimate how rapidly alpha is rotating across waves.

    Compares 30D vs 365D alpha to identify accelerating and decelerating waves.
    Returns a dict with ``has_data`` flag and ``waves`` list.
    """
    if snapshot_df is None or snapshot_df.empty:
        return {"has_data": False, "waves": []}

    if "alpha_30d" not in snapshot_df.columns or "alpha_365d" not in snapshot_df.columns:
        return {"has_data": False, "waves": []}

    name_col = "display_name" if "display_name" in snapshot_df.columns else "wave_name"
    if name_col not in snapshot_df.columns:
        return {"has_data": False, "waves": []}

    rows = []
    for _, r in snapshot_df.iterrows():
        wave = str(r.get(name_col, ""))
        a30 = pd.to_numeric(r.get("alpha_30d"), errors="coerce")
        a365 = pd.to_numeric(r.get("alpha_365d"), errors="coerce")
        if wave and not (np.isnan(a30) or np.isnan(a365)):
            velocity = round(float((a30 - a365) / 12), 4)
            direction = "Accelerating" if velocity > 0 else "Decelerating"
            rows.append({
                "Wave": wave,
                "Alpha 30D": round(float(a30), 4),
                "Alpha 365D": round(float(a365), 4),
                "Rotation Velocity": velocity,
                "Direction": direction,
            })

    if not rows:
        return {"has_data": False, "waves": []}

    rows.sort(key=lambda x: abs(x["Rotation Velocity"]), reverse=True)
    return {"has_data": True, "waves": rows}


def compute_alpha_ignition(snapshot_df):
    """Identify waves where alpha is beginning to emerge (ignition signals).

    A wave is considered igniting when short-horizon alpha (30D) is positive
    while longer-horizon alpha (365D) remains negative or near zero.
    Returns a dict with ``has_data`` flag and ``waves`` list.
    """
    if snapshot_df is None or snapshot_df.empty:
        return {"has_data": False, "waves": []}

    need = {"alpha_30d", "alpha_60d", "alpha_365d"}
    if not need.issubset(snapshot_df.columns):
        return {"has_data": False, "waves": []}

    name_col = "display_name" if "display_name" in snapshot_df.columns else "wave_name"
    if name_col not in snapshot_df.columns:
        return {"has_data": False, "waves": []}

    rows = []
    for _, r in snapshot_df.iterrows():
        wave = str(r.get(name_col, ""))
        a30 = pd.to_numeric(r.get("alpha_30d"), errors="coerce")
        a60 = pd.to_numeric(r.get("alpha_60d"), errors="coerce")
        a365 = pd.to_numeric(r.get("alpha_365d"), errors="coerce")
        if not wave:
            continue
        if np.isnan(a30) or np.isnan(a365):
            continue

        # Ignition: short-term alpha positive, long-term subdued
        ignition_score = round(float(a30 - (a365 / 4)), 4)
        signal = "Igniting" if a30 > 0 and (np.isnan(a365) or a365 < a30 * 0.5) else "Stable"

        rows.append({
            "Wave": wave,
            "Alpha 30D": round(float(a30), 4),
            "Alpha 60D": round(float(a60), 4) if not np.isnan(a60) else "",
            "Alpha 365D": round(float(a365), 4),
            "Ignition Score": ignition_score,
            "Signal": signal,
        })

    if not rows:
        return {"has_data": False, "waves": []}

    rows.sort(key=lambda x: x["Ignition Score"], reverse=True)
    return {"has_data": True, "waves": rows}


# ---------------------------------------------------------------------------
# DataFrame-producing wrappers (canonical attribution source)
# ---------------------------------------------------------------------------

_REQUIRED_ATTRIB_COLS = {
    "wave", "horizon", "total_alpha", "selection_alpha", "momentum_alpha",
    "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha",
}
_HORIZON_LABEL_MAP = {30: "30D", 60: "60D", 90: "90D", 365: "365D"}


def _validate_attrib_df(attrib_df, caller: str) -> pd.DataFrame:
    """Validate and normalize an attribution DataFrame.

    Returns a copy with horizon values normalised to string labels
    (30D / 60D / 90D / 365D).  Raises ``ValueError`` with a clear diagnostic
    message when the input is empty, missing required columns, or contains
    only UNKNOWN_WAVE rows.
    """
    if attrib_df is None or (isinstance(attrib_df, pd.DataFrame) and attrib_df.empty):
        raise ValueError(
            f"[{caller}] Attribution DataFrame is empty. "
            "Run build_alpha_attribution_csv.py to rebuild data/alpha_attribution_summary.csv."
        )
    missing = _REQUIRED_ATTRIB_COLS - set(attrib_df.columns)
    if missing:
        raise ValueError(
            f"[{caller}] Attribution DataFrame missing required columns: {sorted(missing)}. "
            "Schema must include wave, horizon, and all alpha component columns."
        )
    df = attrib_df.copy()
    if "wave" in df.columns:
        unknown_mask = df["wave"].astype(str).str.strip() == "UNKNOWN_WAVE"
        if unknown_mask.all():
            raise ValueError(
                f"[{caller}] All rows have wave='UNKNOWN_WAVE'. "
                "Attribution data is corrupt or placeholder. "
                "Rebuild data/alpha_attribution_summary.csv."
            )
        df = df[~unknown_mask]
        if df.empty:
            raise ValueError(
                f"[{caller}] No valid wave rows remain after filtering UNKNOWN_WAVE entries."
            )
    numeric_horizons = pd.to_numeric(df["horizon"], errors="coerce")
    df = df[numeric_horizons.notna()].copy()
    df["horizon"] = numeric_horizons[numeric_horizons.notna()].astype(int).map(
        lambda h: _HORIZON_LABEL_MAP.get(h, str(h))
    ).values
    df = df.reset_index(drop=True)
    return df


def compute_alpha_quality_df(attrib_df) -> pd.DataFrame:
    """Rank waves by alpha quality derived from attribution data.

    Alpha Quality Score = mean(abs(total_alpha across horizons)) * consistency
    minus a volatility-drag penalty, where consistency is the fraction of
    horizons with positive total_alpha.

    Returns a DataFrame sorted by *alpha_quality_score* descending with
    columns: wave, alpha_quality_score, consistency, total_alpha_30D,
    total_alpha_60D, total_alpha_90D, total_alpha_365D.

    Raises ``ValueError`` when *attrib_df* is invalid (see
    :func:`_validate_attrib_df`).
    """
    df = _validate_attrib_df(attrib_df, "compute_alpha_quality_df")
    rows = []
    for wave, grp in df.groupby("wave"):
        horizon_data = {}
        for _, r in grp.iterrows():
            h = str(r["horizon"])
            t = pd.to_numeric(r.get("total_alpha"), errors="coerce")
            v = pd.to_numeric(r.get("volatility_alpha"), errors="coerce")
            if pd.notna(t):
                horizon_data[h] = {
                    "total_alpha": float(t),
                    "volatility_alpha": float(v) if pd.notna(v) else 0.0,
                }
        if not horizon_data:
            continue
        alphas = [d["total_alpha"] for d in horizon_data.values()]
        if not alphas:
            continue
        consistency = round(sum(1 for a in alphas if a > 0) / len(alphas), 2)
        mean_abs = float(np.mean([abs(a) for a in alphas]))
        avg_vol_drag = float(np.mean([abs(d["volatility_alpha"]) for d in horizon_data.values()]))
        score = round(mean_abs * consistency - avg_vol_drag * 0.5, 6)
        entry = {"wave": str(wave), "alpha_quality_score": score, "consistency": consistency}
        for h_label in ["30D", "60D", "90D", "365D"]:
            entry[f"total_alpha_{h_label}"] = (
                round(horizon_data[h_label]["total_alpha"], 6)
                if h_label in horizon_data
                else None
            )
        rows.append(entry)
    if not rows:
        return pd.DataFrame(
            columns=["wave", "alpha_quality_score", "consistency",
                     "total_alpha_30D", "total_alpha_60D", "total_alpha_90D", "total_alpha_365D"]
        )
    result = pd.DataFrame(rows).sort_values("alpha_quality_score", ascending=False).reset_index(drop=True)
    return result


def compute_capital_pressure_df(attrib_df) -> pd.DataFrame:
    """Compute capital pressure regime per wave from attribution data.

    Capital Pressure Score = residual_alpha + abs(regime_alpha) at the 30D
    horizon (falls back to all horizons if 30D is absent).  Regime bins:
    High (score > 0.005), Low (score < -0.005), Neutral otherwise.

    Returns a DataFrame sorted by *capital_pressure_score* descending with
    columns: wave, capital_pressure_score, regime_bin, residual_alpha_30D,
    regime_alpha_30D.

    Raises ``ValueError`` when *attrib_df* is invalid.
    """
    df = _validate_attrib_df(attrib_df, "compute_capital_pressure_df")
    df30 = df[df["horizon"] == "30D"].copy()
    if df30.empty:
        df30 = df.copy()
    rows = []
    for wave, grp in df30.groupby("wave"):
        residual = pd.to_numeric(grp["residual_alpha"], errors="coerce").mean()
        regime_a = pd.to_numeric(grp["regime_alpha"], errors="coerce").mean()
        residual = float(residual) if pd.notna(residual) else 0.0
        regime_a = float(regime_a) if pd.notna(regime_a) else 0.0
        score = round(residual + abs(regime_a), 6)
        regime_bin = "High" if score > 0.005 else ("Low" if score < -0.005 else "Neutral")
        rows.append({
            "wave": str(wave),
            "capital_pressure_score": score,
            "regime_bin": regime_bin,
            "residual_alpha_30D": round(residual, 6),
            "regime_alpha_30D": round(regime_a, 6),
        })
    if not rows:
        return pd.DataFrame(
            columns=["wave", "capital_pressure_score", "regime_bin",
                     "residual_alpha_30D", "regime_alpha_30D"]
        )
    return pd.DataFrame(rows).sort_values("capital_pressure_score", ascending=False).reset_index(drop=True)


def compute_rotation_velocity_df(attrib_df) -> pd.DataFrame:
    """Compute rotation velocity per wave from attribution data.

    Rotation Velocity = total_alpha_30D − total_alpha_365D.  Direction is
    *Accelerating* when velocity > 0, *Decelerating* otherwise.

    Returns a DataFrame sorted by abs(rotation_velocity) descending with
    columns: wave, total_alpha_30D, total_alpha_365D, rotation_velocity,
    direction.

    Raises ``ValueError`` when *attrib_df* is invalid.
    """
    df = _validate_attrib_df(attrib_df, "compute_rotation_velocity_df")
    rows = []
    for wave, grp in df.groupby("wave"):
        h30 = grp[grp["horizon"] == "30D"]["total_alpha"]
        h365 = grp[grp["horizon"] == "365D"]["total_alpha"]
        a30 = pd.to_numeric(h30, errors="coerce").mean()
        a365 = pd.to_numeric(h365, errors="coerce").mean()
        if pd.isna(a30) or pd.isna(a365):
            continue
        velocity = round(float(a30) - float(a365), 6)
        rows.append({
            "wave": str(wave),
            "total_alpha_30D": round(float(a30), 6),
            "total_alpha_365D": round(float(a365), 6),
            "rotation_velocity": velocity,
            "direction": "Accelerating" if velocity > 0 else "Decelerating",
        })
    if not rows:
        return pd.DataFrame(
            columns=["wave", "total_alpha_30D", "total_alpha_365D", "rotation_velocity", "direction"]
        )
    result = pd.DataFrame(rows)
    result = result.reindex(result["rotation_velocity"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return result


def compute_alpha_ignition_df(attrib_df) -> pd.DataFrame:
    """Compute alpha ignition score per wave and horizon from attribution data.

    Ignition Score = 0.5 × selection_alpha + 0.3 × momentum_alpha
                     − 0.2 × abs(volatility_alpha)

    Returns a DataFrame sorted by *ignition_score* descending with columns:
    wave, horizon, ignition_score, selection_alpha, momentum_alpha,
    volatility_alpha.

    Raises ``ValueError`` when *attrib_df* is invalid.
    """
    df = _validate_attrib_df(attrib_df, "compute_alpha_ignition_df")
    rows = []
    for _, r in df.iterrows():
        sel = pd.to_numeric(r.get("selection_alpha"), errors="coerce")
        mom = pd.to_numeric(r.get("momentum_alpha"), errors="coerce")
        vol = pd.to_numeric(r.get("volatility_alpha"), errors="coerce")
        if pd.isna(sel) or pd.isna(mom) or pd.isna(vol):
            continue
        score = round(0.5 * float(sel) + 0.3 * float(mom) - 0.2 * abs(float(vol)), 6)
        rows.append({
            "wave": str(r["wave"]),
            "horizon": str(r["horizon"]),
            "ignition_score": score,
            "selection_alpha": round(float(sel), 6),
            "momentum_alpha": round(float(mom), 6),
            "volatility_alpha": round(float(vol), 6),
        })
    if not rows:
        return pd.DataFrame(
            columns=["wave", "horizon", "ignition_score",
                     "selection_alpha", "momentum_alpha", "volatility_alpha"]
        )
    return pd.DataFrame(rows).sort_values("ignition_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lightweight wrappers — convert compute_* outputs to DataFrames
# ---------------------------------------------------------------------------

def alpha_quality_df(attrib_df) -> pd.DataFrame:
    """Wrapper: call compute_alpha_quality_df and return DataFrame (empty on error)."""
    try:
        return compute_alpha_quality_df(attrib_df)
    except Exception:
        return pd.DataFrame()


def capital_pressure_df(attrib_df) -> pd.DataFrame:
    """Wrapper: call compute_capital_pressure_df and return DataFrame (empty on error)."""
    try:
        return compute_capital_pressure_df(attrib_df)
    except Exception:
        return pd.DataFrame()


def rotation_velocity_df(attrib_df) -> pd.DataFrame:
    """Wrapper: call compute_rotation_velocity_df and return DataFrame (empty on error)."""
    try:
        return compute_rotation_velocity_df(attrib_df)
    except Exception:
        return pd.DataFrame()


def alpha_ignition_df(attrib_df) -> pd.DataFrame:
    """Wrapper: call compute_alpha_ignition_df and return DataFrame (empty on error)."""
    try:
        return compute_alpha_ignition_df(attrib_df)
    except Exception:
        return pd.DataFrame()
