"""
Ticker normalization utilities.

Provides canonical ticker normalization to ensure consistent handling
across all CSV files (wave_weights.csv, prices.csv, etc.) and before
any merge/grouping operations.
"""


def normalize_ticker(t: str) -> str:
    """
    Normalize ticker symbol to canonical format.
    
    This function ensures all ticker keys are normalized consistently:
    - Handles None values
    - Strips whitespace
    - Converts to uppercase
    - Replaces various dash/hyphen Unicode characters with standard hyphen
    - Replaces dots with hyphens
    
    Args:
        t: Ticker symbol to normalize (can be None)
        
    Returns:
        Normalized ticker symbol as uppercase string with standard formatting.
        Returns empty string if input is None.
        
    Examples:
        >>> normalize_ticker("aapl")
        'AAPL'
        >>> normalize_ticker("BRK.B")
        'BRK-B'
        >>> normalize_ticker("BRK–B")  # en-dash
        'BRK-B'
        >>> normalize_ticker(None)
        ''
        >>> normalize_ticker("  MSFT  ")
        'MSFT'
    """
    if t is None:
        return ""
    return (
        str(t).strip()
        .upper()
        .replace("–", "-")  # en-dash (U+2013)
        .replace("—", "-")  # em-dash (U+2014)
        .replace("‐", "-")  # hyphen (U+2010)
        .replace("−", "-")  # minus sign (U+2212)
        .replace(".", "-")  # dot to hyphen
    )
