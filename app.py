def _truthframe_mtime() -> float:
    try:
        return os.path.getmtime("data/truthframe.json")
    except Exception:
        return 0.0

def get_truthframe_safe() -> Dict[str, Any]:
    """ Defensive wrapper for optional use ONLY. Does not require refactoring existing call sites. """
    try:
        return get_truthframe()
    except Exception:
        logger.warning("[TruthFrame] Recovering gracefully from access failure")
        return {"waves": {}}