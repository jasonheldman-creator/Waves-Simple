# NOTE 077 — Forward Intelligence Signals
# Observational awareness layer only.
# Non-executing. No governance authority.

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd


_REQUIRED_COLUMNS = [
    "signal_id",
    "wave",
    "signal_type",
    "signal_title",
    "observation",
    "confidence",
    "horizon",
    "created_at",
]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_REQUIRED_COLUMNS)


def generate_forward_intelligence_signals(
    attrib_df: pd.DataFrame,
    governance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate observational forward intelligence signals.

    This function is read-only and observational. It does not create
    governance decisions, modify allocations, execute logic, or change
    existing workflows.

    Parameters
    ----------
    attrib_df:
        Alpha attribution summary dataframe loaded from
        ``data/alpha_attribution_summary.csv``.
    governance_df:
        Governance decisions dataframe from
        ``helpers.governance_lifecycle.load_governance_decisions_df``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``signal_id``, ``wave``, ``signal_type``,
        ``signal_title``, ``observation``, ``confidence``, ``horizon``,
        ``created_at``.  Returns an empty DataFrame (not None) when no
        signals are detected.
    """
    signals: list[dict] = []
    now = datetime.now(tz=timezone.utc).isoformat()

    if attrib_df is not None and not attrib_df.empty:
        df = attrib_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        has_wave = "wave" in df.columns
        has_horizon = "horizon" in df.columns
        has_vol = "volatility_alpha" in df.columns
        has_total = "total_alpha" in df.columns
        has_momentum = "momentum_alpha" in df.columns

        if has_wave and has_horizon and has_vol and has_total:
            # ----------------------------------------------------------------
            # Rule A — Earnings Preparation Window
            # If volatility_alpha increasing AND 30D alpha deteriorating
            # ----------------------------------------------------------------
            try:
                for wave_name, wave_df in df.groupby("wave"):
                    wave_df = wave_df.sort_values("horizon")
                    horizons = wave_df["horizon"].tolist()

                    if 30 not in horizons:
                        continue

                    row_30 = wave_df[wave_df["horizon"] == 30]
                    total_30 = float(row_30["total_alpha"].iloc[0])
                    vol_30 = float(row_30["volatility_alpha"].iloc[0])

                    # Compare to a longer horizon to detect "increasing" vol alpha
                    longer = [h for h in horizons if h > 30]
                    if not longer:
                        continue
                    row_long = wave_df[wave_df["horizon"] == min(longer)]
                    vol_long = float(row_long["volatility_alpha"].iloc[0])

                    vol_increasing = vol_30 > vol_long
                    alpha_deteriorating = total_30 < 0

                    if vol_increasing and alpha_deteriorating:
                        signals.append({
                            "signal_id": str(uuid.uuid4()),
                            "wave": wave_name,
                            "signal_type": "Earnings Conditioning",
                            "signal_title": "Earnings Preparation Window",
                            "observation": (
                                f"{wave_name}: volatility alpha rising "
                                f"({vol_30:.4f} vs {vol_long:.4f} at longer horizon) "
                                f"with deteriorating 30-day alpha ({total_30:.4f}). "
                                "Earnings conditioning conditions present."
                            ),
                            "confidence": "Moderate",
                            "horizon": "30D",
                            "created_at": now,
                        })
            except Exception:
                pass

        if has_wave and has_horizon and has_total:
            # ----------------------------------------------------------------
            # Rule B — Position Conditioning
            # If long-term alpha positive AND short-term alpha negative
            # ----------------------------------------------------------------
            try:
                for wave_name, wave_df in df.groupby("wave"):
                    wave_df = wave_df.sort_values("horizon")
                    horizons = sorted(wave_df["horizon"].tolist())

                    short_horizons = [h for h in horizons if h <= 30]
                    long_horizons = [h for h in horizons if h >= 60]

                    if not short_horizons or not long_horizons:
                        continue

                    row_short = wave_df[wave_df["horizon"] == min(short_horizons)]
                    row_long = wave_df[wave_df["horizon"] == max(long_horizons)]

                    alpha_short = float(row_short["total_alpha"].iloc[0])
                    alpha_long = float(row_long["total_alpha"].iloc[0])

                    if alpha_long > 0 and alpha_short < 0:
                        signals.append({
                            "signal_id": str(uuid.uuid4()),
                            "wave": wave_name,
                            "signal_type": "Short-Term Dislocation",
                            "signal_title": "Position Conditioning",
                            "observation": (
                                f"{wave_name}: long-term alpha positive "
                                f"({alpha_long:.4f} at {max(long_horizons)}D) "
                                f"diverging from negative short-term alpha "
                                f"({alpha_short:.4f} at {min(short_horizons)}D). "
                                "Short-term dislocation observed."
                            ),
                            "confidence": "High",
                            "horizon": f"{min(short_horizons)}D–{max(long_horizons)}D",
                            "created_at": now,
                        })
            except Exception:
                pass

        if has_wave and has_horizon and has_total:
            # ----------------------------------------------------------------
            # Rule C — Governance Pressure Build
            # If ≥2 waves show same attribution deterioration
            # ----------------------------------------------------------------
            try:
                deteriorating_waves = []
                for wave_name, wave_df in df.groupby("wave"):
                    wave_df = wave_df.sort_values("horizon")
                    horizons = sorted(wave_df["horizon"].tolist())
                    if len(horizons) < 2:
                        continue
                    first_alpha = float(
                        wave_df[wave_df["horizon"] == horizons[0]]["total_alpha"].iloc[0]
                    )
                    last_alpha = float(
                        wave_df[wave_df["horizon"] == horizons[-1]]["total_alpha"].iloc[0]
                    )
                    if first_alpha < last_alpha < 0:
                        deteriorating_waves.append(wave_name)

                if len(deteriorating_waves) >= 2:
                    signals.append({
                        "signal_id": str(uuid.uuid4()),
                        "wave": ", ".join(deteriorating_waves[:3])
                        + ("..." if len(deteriorating_waves) > 3 else ""),
                        "signal_type": "Cluster Formation",
                        "signal_title": "Governance Pressure Build",
                        "observation": (
                            f"{len(deteriorating_waves)} waves show correlated attribution "
                            "deterioration across horizons. Cluster formation observed: "
                            + ", ".join(deteriorating_waves[:3])
                            + ("..." if len(deteriorating_waves) > 3 else "")
                            + "."
                        ),
                        "confidence": "Moderate",
                        "horizon": "Multi-Horizon",
                        "created_at": now,
                    })
            except Exception:
                pass

        if has_wave and has_horizon and has_momentum:
            # ----------------------------------------------------------------
            # Rule D — Emerging Momentum Shift
            # If momentum_alpha flips sign between horizons
            # ----------------------------------------------------------------
            try:
                for wave_name, wave_df in df.groupby("wave"):
                    wave_df = wave_df.sort_values("horizon")
                    horizons = sorted(wave_df["horizon"].tolist())
                    if len(horizons) < 2:
                        continue

                    mom_values = []
                    for h in horizons:
                        row = wave_df[wave_df["horizon"] == h]
                        mom_values.append((h, float(row["momentum_alpha"].iloc[0])))

                    # Check for sign flip across consecutive horizons
                    for i in range(len(mom_values) - 1):
                        h1, m1 = mom_values[i]
                        h2, m2 = mom_values[i + 1]
                        if m1 != 0 and m2 != 0 and (m1 > 0) != (m2 > 0):
                            signals.append({
                                "signal_id": str(uuid.uuid4()),
                                "wave": wave_name,
                                "signal_type": "Momentum Transition",
                                "signal_title": "Emerging Momentum Shift",
                                "observation": (
                                    f"{wave_name}: momentum alpha flips from "
                                    f"{m1:.4f} at {h1}D to {m2:.4f} at {h2}D. "
                                    "Momentum transition signal detected."
                                ),
                                "confidence": "Low",
                                "horizon": f"{h1}D→{h2}D",
                                "created_at": now,
                            })
                            break
            except Exception:
                pass

    if not signals:
        return _empty_df()

    return pd.DataFrame(signals, columns=_REQUIRED_COLUMNS)
