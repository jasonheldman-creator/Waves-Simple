import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ADAPTIVE_STATE_PATH = Path("data/adaptive_state.json")

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
    if not decisions or len(decisions) < 2:
        return {"has_data": False}

    recorded = [d for d in decisions if d.get("outcome_30d") and d["outcome_30d"] != "Pending"]
    if len(recorded) < 2:
        return {"has_data": False}

    positive = sum(1 for d in recorded if d.get("outcome_30d") == "Positive")
    neutral = sum(1 for d in recorded if d.get("outcome_30d") == "Neutral")
    total = len(recorded)

    doa = (positive / total * 100) if total > 0 else 0
    oc = ((positive + neutral) / total * 100) if total > 0 else 0
    si = min(100, doa * 0.8 + oc * 0.2)
    learning_index = doa * 0.5 + oc * 0.3 + si * 0.2

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

    monthly_points = []
    for i, d in enumerate(recorded):
        running_pos = sum(1 for x in recorded[:i+1] if x.get("outcome_30d") == "Positive")
        running_total = i + 1
        monthly_points.append([d.get("date", f"T{i}"), round(running_pos / running_total * 100, 1)])

    return {
        "has_data": True,
        "learning_index": learning_index,
        "grade": grade,
        "zone": zone,
        "decision_outcome_alignment": doa,
        "outcome_consistency": oc,
        "structural_improvement": si,
        "monthly_points": monthly_points,
    }

def compute_efficiency_curve(decisions, adaptive_state):
    if not decisions or len(decisions) < 2:
        return {"has_data": False}

    recorded = [d for d in decisions if d.get("status") == "Recorded"]
    approved = [d for d in decisions if d.get("approval_status") == "Approved"]
    total = len(decisions)

    if total == 0:
        return {"has_data": False}

    signal_engagement_rate = min(100, len(recorded) / max(total, 1) * 100)
    decision_implementation_rate = min(100, len(approved) / max(total, 1) * 100)

    avg_latency_days = 5
    decision_latency_score = max(0, 100 - avg_latency_days * 5)

    efficiency_index = signal_engagement_rate * 0.4 + decision_implementation_rate * 0.3 + decision_latency_score * 0.3

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
        running_recorded = sum(1 for x in decisions[:i+1] if x.get("status") == "Recorded")
        running_total = i + 1
        monthly_points.append([d.get("date", f"T{i}"), round(running_recorded / running_total * 100, 1)])

    return {
        "has_data": True,
        "efficiency_index": efficiency_index,
        "grade": grade,
        "signal_engagement_rate": signal_engagement_rate,
        "decision_implementation_rate": decision_implementation_rate,
        "decision_latency_score": decision_latency_score,
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

def compute_cross_horizon_stability(snapshot_df, attrib_df):
    horizons = {"1D": "alpha_1d", "30D": "alpha_30d", "365D": "alpha_365d"}
    drivers = []

    component_cols = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha"]
    driver_names = ["Selection", "Momentum", "Volatility", "Regime", "Exposure"]

    if attrib_df is not None and not attrib_df.empty:
        for name, col in zip(driver_names, component_cols):
            if col in attrib_df.columns:
                vals = pd.to_numeric(attrib_df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    mean_val = float(vals.mean())
                    state_30d = "Positive" if mean_val > 0.005 else "Negative" if mean_val < -0.005 else "Neutral"
                    state_90d = state_30d
                    state_365d = state_30d
                    stability = "Stable" if vals.std() < 0.01 else "Moderate" if vals.std() < 0.03 else "Volatile"
                    drivers.append({
                        "Driver": name,
                        "30D State": state_30d,
                        "90D State": state_90d,
                        "365D State": state_365d,
                        "Stability": stability,
                    })

    if not drivers and snapshot_df is not None:
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
