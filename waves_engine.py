# waves_engine.py — WAVES Intelligence™ Engine Wrapper (SAFE)
# Purpose:
# 1) Preserve your current working engine (moved to waves_engine_prod.py)
# 2) Register NEW Waves (like the Demas Small–Mid Cap Value Acceleration Wave)
# 3) Expose the same API the console expects, without breaking Standard / AMB
#
# How it works:
# - Imports everything from waves_engine_prod.py (your frozen, working engine)
# - Then patches in the Demas wave into whichever registry dicts exist
# - If a registry key doesn't exist in your engine, it safely no-ops

from __future__ import annotations

import importlib
from typing import Any, Dict

PROD_MODULE_NAME = "waves_engine_prod"
DEMAS_WAVE_ID = "DEMAS_SMID_VALUE_ACCEL"
DEMAS_WAVE_NAME = "Demas Small–Mid Cap Value Acceleration Wave"

# Benchmark suggestion:
# - Russell 2000 proxy is typically IWM
# - If your engine supports blended benchmarks, we can later do IWM/IJR blend
DEMAS_BENCHMARK = {
    "type": "ETF",
    "tickers": {"IWM": 1.0},
    "desc": "IWM (Russell 2000 proxy)",
}

# Optional metadata (won’t break anything if unused)
DEMAS_META = {
    "wave_id": DEMAS_WAVE_ID,
    "name": DEMAS_WAVE_NAME,
    "category": "US Equities",
    "style": "SMID Value + Fundamental Acceleration",
    "signals": {
        "min_qoq_revenue_growth": 0.20,  # 20%
        "min_qoq_earnings_growth": 0.25, # 25%
        "max_pe": 12.0,
    },
    "benchmark": DEMAS_BENCHMARK,
}

def _safe_patch_registry(ns: Dict[str, Any]) -> None:
    """
    Patch the Demas wave into common registry variable names without assumptions.
    This will NOT crash if your prod engine uses different structures.
    """

    # 1) Common registry dict names used across engines
    registry_names = [
        "WAVES",
        "WAVE_REGISTRY",
        "WAVE_DEFS",
        "WAVE_DEFINITIONS",
        "WAVE_CONFIG",
        "WAVE_CONFIGS",
    ]

    for name in registry_names:
        reg = ns.get(name)
        if isinstance(reg, dict):
            # If registry is keyed by wave_id
            if DEMAS_WAVE_ID not in reg:
                # Try to match the shape of existing entries (best-effort)
                exemplar = None
                try:
                    exemplar = next(iter(reg.values()))
                except Exception:
                    exemplar = None

                if isinstance(exemplar, dict):
                    entry = dict(exemplar)  # shallow clone of structure
                    # Overwrite obvious fields
                    for k in ["wave_id", "id", "key"]:
                        if k in entry:
                            entry[k] = DEMAS_WAVE_ID
                    for k in ["name", "wave_name", "title"]:
                        if k in entry:
                            entry[k] = DEMAS_WAVE_NAME

                    # Benchmarks (support multiple possible keys)
                    if "benchmark" in entry:
                        entry["benchmark"] = DEMAS_BENCHMARK
                    if "benchmark_desc" in entry:
                        entry["benchmark_desc"] = DEMAS_BENCHMARK["desc"]
                    if "benchmark_tickers" in entry:
                        entry["benchmark_tickers"] = DEMAS_BENCHMARK["tickers"]
                    if "benchmark_weights" in entry:
                        entry["benchmark_weights"] = DEMAS_BENCHMARK["tickers"]

                    # Add meta if there is a place
                    if "meta" in entry and isinstance(entry["meta"], dict):
                        entry["meta"] = {**entry["meta"], **DEMAS_META}

                    reg[DEMAS_WAVE_ID] = entry
                else:
                    # If we can’t infer structure, store minimal metadata
                    reg[DEMAS_WAVE_ID] = {"wave_id": DEMAS_WAVE_ID, "name": DEMAS_WAVE_NAME, "benchmark": DEMAS_BENCHMARK}

            return  # patched one registry; enough

    # 2) Some engines keep a list of wave names/ids
    list_names = ["WAVE_IDS", "WAVES_LIST", "WAVE_LIST", "ALL_WAVES"]
    for name in list_names:
        lst = ns.get(name)
        if isinstance(lst, list) and DEMAS_WAVE_ID not in lst and DEMAS_WAVE_NAME not in lst:
            # Prefer IDs if list appears to be IDs
            if lst and isinstance(lst[0], str) and ("_" in lst[0] or lst[0].isupper()):
                lst.append(DEMAS_WAVE_ID)
            else:
                lst.append(DEMAS_WAVE_NAME)

    # 3) Benchmark mappings sometimes live in a separate dict
    bench_names = ["BENCHMARKS", "WAVE_BENCHMARKS", "BENCHMARK_MAP"]
    for name in bench_names:
        bm = ns.get(name)
        if isinstance(bm, dict) and DEMAS_WAVE_ID not in bm:
            bm[DEMAS_WAVE_ID] = DEMAS_BENCHMARK.get("tickers", {"IWM": 1.0})


# ----------------------------
# Load PROD engine and re-export
# ----------------------------

_prod = importlib.import_module(PROD_MODULE_NAME)

# Export EVERYTHING from prod module into this module’s namespace
globals().update({k: getattr(_prod, k) for k in dir(_prod) if not k.startswith("__")})

# Patch registries after import
_safe_patch_registry(globals())

# Optional: helper for debugging (won’t be used unless called)
def waves_wrapper_status() -> Dict[str, Any]:
    """
    Returns a small status blob to confirm wrapper + wave registration worked.
    Safe to call from console/debug.
    """
    reg = None
    for n in ["WAVES", "WAVE_REGISTRY", "WAVE_DEFS", "WAVE_CONFIG", "WAVE_CONFIGS"]:
        v = globals().get(n)
        if isinstance(v, dict):
            reg = (n, v)
            break

    return {
        "prod_module": PROD_MODULE_NAME,
        "demas_wave_id": DEMAS_WAVE_ID,
        "demas_wave_name": DEMAS_WAVE_NAME,
        "registry_found": reg[0] if reg else None,
        "demas_registered": (DEMAS_WAVE_ID in reg[1]) if reg else False,
    }