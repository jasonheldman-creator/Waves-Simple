"""
Alpha Contracts Module - Governance Rules for Wave Performance

This module defines governance "contracts" for each Wave, specifying
performance expectations, risk limits, and alpha source requirements.

Each contract includes:
- max_dd_limit: Maximum drawdown tolerance
- vol_budget: Volatility budget/target
- min_sharpe: Minimum acceptable Sharpe ratio
- expected_alpha_sources: List of expected alpha contributors
- exposure_range: Acceptable exposure range (min, max)

Contract evaluation produces color-coded status:
- GREEN: All criteria met
- YELLOW: Some criteria at warning threshold
- RED: Critical criteria violated
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum


class ContractStatus(Enum):
    """Contract compliance status."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class AlphaContract:
    """
    Governance contract for a Wave.
    
    Attributes:
        wave_name: Name of the Wave
        max_dd_limit: Maximum acceptable drawdown (decimal, e.g., -0.15 for -15%)
        vol_budget: Target/maximum volatility (annualized, decimal)
        min_sharpe: Minimum acceptable Sharpe ratio
        expected_alpha_sources: List of alpha sources that should contribute
        exposure_range: Tuple of (min_exposure, max_exposure) in decimal form
        description: Human-readable description of contract
    """
    wave_name: str
    max_dd_limit: float
    vol_budget: float
    min_sharpe: float
    expected_alpha_sources: List[str]
    exposure_range: Tuple[float, float]
    description: str = ""


# Default contracts for each Wave
DEFAULT_CONTRACTS = {
    "Growth Wave": AlphaContract(
        wave_name="Growth Wave",
        max_dd_limit=-0.15,  # -15% max drawdown
        vol_budget=0.20,  # 20% annualized vol
        min_sharpe=0.5,
        expected_alpha_sources=["selection", "overlay"],
        exposure_range=(0.6, 1.0),
        description="Growth-focused equity wave with moderate risk tolerance"
    ),
    "Crypto Income Wave": AlphaContract(
        wave_name="Crypto Income Wave",
        max_dd_limit=-0.25,  # -25% max drawdown (crypto volatility)
        vol_budget=0.40,  # 40% annualized vol
        min_sharpe=0.3,
        expected_alpha_sources=["selection", "overlay"],
        exposure_range=(0.5, 0.9),
        description="Crypto-focused income wave with higher volatility tolerance"
    ),
    "SP500 Wave": AlphaContract(
        wave_name="SP500 Wave",
        max_dd_limit=-0.12,  # -12% max drawdown
        vol_budget=0.18,  # 18% annualized vol
        min_sharpe=0.7,
        expected_alpha_sources=["selection", "overlay"],
        exposure_range=(0.7, 1.0),
        description="Large-cap equity wave with lower risk tolerance"
    ),
    "Russell 3000 Wave": AlphaContract(
        wave_name="Russell 3000 Wave",
        max_dd_limit=-0.13,  # -13% max drawdown
        vol_budget=0.19,  # 19% annualized vol
        min_sharpe=0.6,
        expected_alpha_sources=["selection", "overlay"],
        exposure_range=(0.7, 1.0),
        description="Broad market equity wave"
    )
}


def get_contract(wave_name: str) -> Optional[AlphaContract]:
    """
    Get the alpha contract for a specific wave.
    
    Args:
        wave_name: Name of the Wave
        
    Returns:
        AlphaContract if available, None otherwise
    """
    return DEFAULT_CONTRACTS.get(wave_name)


def evaluate_contract(
    wave_name: str,
    metrics: Dict[str, Any],
    contract: Optional[AlphaContract] = None
) -> Tuple[ContractStatus, List[str]]:
    """
    Evaluate Wave performance against its alpha contract.
    
    Args:
        wave_name: Name of the Wave
        metrics: Dictionary of computed metrics
        contract: Optional contract override (uses default if None)
        
    Returns:
        Tuple of (status: ContractStatus, reasons: List[str])
    """
    # Get contract
    if contract is None:
        contract = get_contract(wave_name)
    
    # If no contract defined, return GREEN with info message
    if contract is None:
        return (ContractStatus.GREEN, [f"No contract defined for {wave_name}"])
    
    reasons = []
    violations = 0
    warnings = 0
    
    # Check max drawdown
    max_dd = metrics.get("max_drawdown")
    if max_dd is not None:
        if max_dd < contract.max_dd_limit:
            violations += 1
            reasons.append(f"Max DD violation: {max_dd*100:.2f}% exceeds limit {contract.max_dd_limit*100:.2f}%")
        elif max_dd < contract.max_dd_limit * 0.8:  # Within 80% of limit
            warnings += 1
            reasons.append(f"Max DD warning: {max_dd*100:.2f}% approaching limit {contract.max_dd_limit*100:.2f}%")
    
    # Check volatility
    vol = metrics.get("volatility")
    if vol is not None:
        if vol > contract.vol_budget:
            violations += 1
            reasons.append(f"Volatility violation: {vol*100:.2f}% exceeds budget {contract.vol_budget*100:.2f}%")
        elif vol > contract.vol_budget * 0.9:  # Within 90% of budget
            warnings += 1
            reasons.append(f"Volatility warning: {vol*100:.2f}% approaching budget {contract.vol_budget*100:.2f}%")
    
    # Check Sharpe ratio
    sharpe = metrics.get("sharpe_ratio")
    if sharpe is not None:
        if sharpe < contract.min_sharpe:
            violations += 1
            reasons.append(f"Sharpe violation: {sharpe:.2f} below minimum {contract.min_sharpe:.2f}")
        elif sharpe < contract.min_sharpe * 1.2:  # Within 120% of minimum
            warnings += 1
            reasons.append(f"Sharpe warning: {sharpe:.2f} near minimum {contract.min_sharpe:.2f}")
    
    # Check exposure range
    exposure = metrics.get("exposure")
    if exposure is not None:
        min_exp, max_exp = contract.exposure_range
        if exposure < min_exp or exposure > max_exp:
            violations += 1
            reasons.append(f"Exposure violation: {exposure*100:.1f}% outside range [{min_exp*100:.1f}%, {max_exp*100:.1f}%]")
    
    # Check alpha sources availability
    missing_sources = []
    if metrics.get("selection_alpha") is None and "selection" in contract.expected_alpha_sources:
        missing_sources.append("selection")
    if metrics.get("overlay_alpha") is None and "overlay" in contract.expected_alpha_sources:
        missing_sources.append("overlay")
    
    if missing_sources:
        warnings += 1
        reasons.append(f"Missing alpha sources: {', '.join(missing_sources)}")
    
    # Determine overall status
    if violations > 0:
        status = ContractStatus.RED
    elif warnings > 0:
        status = ContractStatus.YELLOW
    else:
        status = ContractStatus.GREEN
        reasons.append("All contract criteria met")
    
    return (status, reasons)


def format_contract_summary(contract: AlphaContract) -> str:
    """
    Format a contract as a human-readable summary.
    
    Args:
        contract: AlphaContract to format
        
    Returns:
        Formatted string summary
    """
    min_exp, max_exp = contract.exposure_range
    
    summary = f"""
**{contract.wave_name} - Alpha Contract**

{contract.description}

**Risk Limits:**
- Max Drawdown: {contract.max_dd_limit*100:.1f}%
- Volatility Budget: {contract.vol_budget*100:.1f}%
- Min Sharpe Ratio: {contract.min_sharpe:.2f}

**Exposure Range:** {min_exp*100:.1f}% - {max_exp*100:.1f}%

**Expected Alpha Sources:** {', '.join(contract.expected_alpha_sources)}
"""
    return summary.strip()


def get_contract_status_color(status: ContractStatus) -> str:
    """
    Get the display color for a contract status.
    
    Args:
        status: ContractStatus enum value
        
    Returns:
        Color string for UI display
    """
    if status == ContractStatus.GREEN:
        return "🟢 GREEN"
    elif status == ContractStatus.YELLOW:
        return "🟡 YELLOW"
    else:  # RED
        return "🔴 RED"
