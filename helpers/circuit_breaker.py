def reset_all_circuit_breakers(circuit_breakers):
    """
    Resets all existing circuit breakers by calling their reset() method.
    This function is intended for manual invocation only.
    """
    for breaker in circuit_breakers:
        breaker.reset()
