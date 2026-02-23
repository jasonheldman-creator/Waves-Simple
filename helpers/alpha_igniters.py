"""Alpha igniters context - stub implementation."""

_DEFAULT_CONTEXT = {
    "alpha_signal": "Neutral",
    "primary_driver": "Observational placeholder",
    "supporting_signals": ["No live signals available"],
    "confidence": "Low",
}

_WAVE_HINTS = {
    "Growth": {"alpha_signal": "Positive", "primary_driver": "Earnings momentum",
               "confidence": "Medium"},
    "Income": {"alpha_signal": "Neutral", "primary_driver": "Yield environment",
               "confidence": "Medium"},
    "Defensive": {"alpha_signal": "Neutral", "primary_driver": "Volatility regime",
                  "confidence": "Low"},
    "SP500": {"alpha_signal": "Positive", "primary_driver": "Broad market trend",
              "confidence": "Medium"},
}


def get_alpha_context_for_wave(wave_name):
    """Return alpha context dict for a given wave name."""
    base = {**_DEFAULT_CONTEXT, "wave": wave_name, "supporting_signals": list(_DEFAULT_CONTEXT["supporting_signals"])}
    hint = _WAVE_HINTS.get(wave_name, {})
    base.update(hint)
    return base


def get_all_igniters():
    """Return list of alpha context dicts for all known waves."""
    waves = list(_WAVE_HINTS.keys()) or ["Growth", "Income", "Defensive", "SP500"]
    return [get_alpha_context_for_wave(w) for w in waves]
