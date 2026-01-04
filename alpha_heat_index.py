# alpha_heat_index.py
# WAVES Intelligence™ — Alpha Heat Index (AHI)
# Console-layer analytic only (NO engine changes)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AHIConfig:
    timeframes: Tuple[str, ...] = ("Intraday", "30D", "60D")
    z_clip: float = 2.5
    alpha_col_candidates: Dict[str, Tuple[str, ...]] = None

    def __post_init__(self):
        if self.alpha_col_candidates is None:
            self.alpha_col_candidates = {
                "Intraday": (
                    "alpha_intraday", "alpha_intra", "intraday_alpha", "alpha_today",
                    "Alpha (Intraday)", "Alpha Intraday", "Alpha Captured (Intraday)",
                    "alpha_capture_intraday", "alpha_captured_intraday",
                ),
                "30D": (
                    "alpha_30d", "alpha_30D", "alpha_30", "30d_alpha",
                    "Alpha (30D)", "Alpha 30D", "Alpha Captured (30D)",
                    "alpha_capture_30d", "alpha_captured_30d",
                ),
                "60D": (
                    "alpha_60d", "alpha_60D", "alpha_60", "60d_alpha",
                    "Alpha (60D)", "Alpha 60D", "Alpha Captured (60D)",
                    "alpha_capture_60d", "alpha_captured_60d",
                ),
                "1Y": (
                    "alpha_1y", "alpha_1Y", "alpha_252d", "alpha_12m", "alpha_1yr",
                    "Alpha (1Y)", "Alpha 1Y", "Alpha Captured (1Y)",
                    "alpha_capture_1y", "alpha_captured_1y",
                ),
            }


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        lc = str(c).lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _coerce_float(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return float("nan")
    s = s.replace("%", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _robust_z(series: pd.Series) -> pd.Series:
    x = series.astype(float)
    med = np.nanmedian(x.values)
    mad = np.nanmedian(np.abs(x.values - med))
    if mad == 0 or np.isnan(mad):
        std = np.nanstd(x.values)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(x)), index=series.index)
        return (x - np.nanmean(x.values)) / std
    return (x - med) / (1.4826 * mad)


def _to_heat(z: pd.Series, z_clip: float) -> pd.Series:
    zc = z.clip(-z_clip, z_clip)
    return 100.0 / (1.0 + np.exp(-zc))


def build_alpha_heat_index(
    overview_df: pd.DataFrame,
    wave_name_col: Optional[str] = None,
    cfg: Optional[AHIConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = AHIConfig()

    if overview_df is None or len(overview_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df = overview_df.copy()

    if wave_name_col is None:
        for c in ("Wave", "wave", "Wave Name", "Wave_Name", "name"):
            if c in df.columns:
                wave_name_col = c
                break
    if wave_name_col is None:
        wave_name_col = df.columns[0]

    raw = pd.DataFrame(index=df[wave_name_col].astype(str).values)

    for tf in cfg.timeframes:
        col = _find_col(df, cfg.alpha_col_candidates.get(tf, ()))
        if col is None:
            raw[tf] = np.nan
        else:
            raw[tf] = df[col].apply(_coerce_float).astype(float)

    heat = pd.DataFrame(index=raw.index)
    for tf in raw.columns:
        z = _robust_z(raw[tf])
        heat[tf] = _to_heat(z, cfg.z_clip)

    heat = heat.dropna(axis=1, how="all")
    raw = raw[heat.columns]

    return heat.round(1), raw


def describe_ahi_cell(alpha_value: float, heat_value: float, tf_label: str) -> str:
    if alpha_value is None or (isinstance(alpha_value, float) and math.isnan(alpha_value)):
        return f"{tf_label}: alpha not available."
    direction = "positive" if alpha_value > 0 else "negative" if alpha_value < 0 else "flat"
    strength = (
        "very strong" if heat_value >= 80 else
        "strong" if heat_value >= 65 else
        "neutral" if heat_value >= 45 else
        "weak" if heat_value >= 30 else
        "very weak"
    )
    return (
        f"{tf_label}: raw alpha {alpha_value:.6f} ({direction}); "
        f"relative heat {heat_value:.1f}/100 ({strength} vs peer Waves)."
    )