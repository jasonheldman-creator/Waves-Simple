import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from urllib.parse import quote_plus

# Try to import your existing engine; app still works if not present
try:
    import waves_engine  # type: ignore
    HAS_ENGINE = True
except Exception:
    waves_engine = None
    HAS_ENGINE = False


# =========================
#  CONFIG & CONSTANTS
# =========================

APP_TITLE = "WAVES Intelligence™ Institutional Console — Vector1"
APP_SUBTITLE = "Adaptive Portfolio Waves™ • Alpha-Minus-Beta • Private Logic™ • SmartSafe™"

LOGS_POSITIONS_DIR = os.path.join("logs", "positions")
LOGS_PERFORMANCE_DIR = os.path.join("logs", "performance")
HUMAN_OVERRIDE_DIR = os.path.join("logs", "human_overrides")

ALPHA_CAPTURE_CANDIDATES = [
    os.path.join(LOGS_PERFORMANCE_DIR, "alpha_capture_matrix.csv"),
    "alpha_capture_matrix.csv",
]

WAVESCORE_CANDIDATES = [
    os.path.join(LOGS_PERFORMANCE_DIR, "wavescore_summary.csv"),
    "wavescore_summary.csv",
]

WAVE_WEIGHTS_PATH = "wave_weights.csv"

# Optional metadata to make UI nicer
WAVE_METADATA: Dict[str, Dict[str, str]] = {
    "S&P 500 Wave": {
        "category": "Core Equity",
        "benchmark": "SPY",
        "tagline": "Core S&P 500 exposure with adaptive overlays.",
    },
    "Growth Wave": {
        "category": "Growth Equity",
        "benchmark": "QQQ",
        "tagline": "High-growth exposure tuned for volatility and drawdowns.",
    },
    "Small Cap Growth Wave": {
        "category": "Small Cap",
        "benchmark": "IWM",
        "tagline": "Adaptive small-cap engine with disciplined risk gates.",
    },
    "Small–Mid Cap Growth Wave": {
        "category": "SMID Growth",
        "benchmark": "IJH",
        "tagline": "Blended small-mid growth with AI-driven factor tilts.",
    },
    "Future Power & Energy Wave": {
        "category": "Energy & Transition",
        "benchmark": "XLE",
        "tagline": "Future power, energy transition and infrastructure.",
    },
    "Crypto Income Wave": {
        "category": "Digital Assets",
        "benchmark": "BTC-USD",
        "tagline": "Crypto income overlay with disciplined risk throttles.",
    },
    "Quantum Computing Wave": {
       