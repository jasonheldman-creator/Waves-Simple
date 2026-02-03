// Modified system health logic to handle partial-horizon validation when markets are closed.

function checkHealth(waves) {
    const validHorizons = waves.filter(wave => wave.isValid);
    if (validHorizons.length > 0) {
        return 'HEALTHY'; // At least one valid horizon exists
    }
    return 'DEGRADED'; // No valid horizons exist
}

// Other existing code remains unchanged...