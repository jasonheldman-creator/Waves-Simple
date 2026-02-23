"""Architecture packet export - stub implementation."""
import json
from datetime import datetime, timezone

_VERSION = "1.0.0"
_MODULES = [
    "governance_lifecycle",
    "strategy_instructions",
    "alpha_igniters",
    "governance_action_builder",
    "decision_intent",
    "daily_cycle_engine",
    "ai_action_bus",
    "governance_replay",
    "decision_continuity",
    "policy_hash",
    "holding_intelligence",
    "market_data",
    "market_briefing",
    "market_news",
    "strategy_security_optimizer",
    "decision_constructor",
    "escalation_engine",
]
_DATA_SOURCES = [
    "data/governance_decisions.json",
    "data/strategy_instructions.json",
    "data/live_snapshot.csv",
    "data/cache/prices_cache.parquet",
    "wave_config.csv",
    "wave_weights.csv",
]


def get_architecture_summary():
    """Return a summary of the current architecture."""
    return {
        "version": _VERSION,
        "modules": _MODULES,
        "data_sources": _DATA_SOURCES,
    }


def export_architecture_packet(format="json"):
    """Export the architecture packet in the requested format.

    Parameters
    ----------
    format : str
        "json" returns a dict; "bytes" returns UTF-8 encoded bytes.
    """
    packet = {
        **get_architecture_summary(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    if format == "bytes":
        return json.dumps(packet, indent=2).encode("utf-8")
    return packet
