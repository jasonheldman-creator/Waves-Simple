def wave_performance_diagnostics(wave_data):
    """
    Isolates and records reasons for validation issues at the wave level.
    It does this without impacting global health computations.
    """
    issues = []

    # Check for null or invalid values
    if wave_data is None:
        issues.append('No data provided')
        return issues

    # Example validation checks
    if not isinstance(wave_data.get('amplitude'), (int, float)):
        issues.append('Invalid amplitude value')
    if not isinstance(wave_data.get('frequency'), (int, float)):
        issues.append('Invalid frequency value')

    # Additional checks can be implemented following the tiered health logic
    #...

    return issues

# Integration with global health computation (pseudocode)
def compute_global_health(wave_list):
    """
    Computes global health based on validated wave data.
    Substitutes wave performance diagnostics as needed.
    """
    for wave in wave_list:
        validation_issues = wave_performance_diagnostics(wave)
        if validation_issues:
            record_validation_issues(wave, validation_issues)
    # Proceed with global health calculation


