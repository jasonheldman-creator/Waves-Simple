"""
Executive Summary Narrative Generator

Generates natural language executive summaries from snapshot data and market context.
Includes:
- Top outperformers of the day
- Waves needing user attention
- Market regime implications
- High-level reasons for alpha capture/loss
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd


def _format_pct(value: float) -> str:
    """Format a decimal value as a percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:+.2f}%"


def _format_value(value: float, decimals: int = 2) -> str:
    """Format a numeric value."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def identify_top_performers(snapshot_df, n: int = 3) -> List[Dict[str, Any]]:
    """
    Identify top performing waves by 1D return.
    
    Args:
        snapshot_df: Snapshot DataFrame
        n: Number of top performers to return
        
    Returns:
        List of dicts with wave info
    """
    if snapshot_df is None or snapshot_df.empty:
        return []
    
    if "Return_1D" not in snapshot_df.columns:
        return []
    
    # Sort by 1D return descending
    top = snapshot_df.nlargest(n, "Return_1D")
    
    performers = []
    for _, row in top.iterrows():
        performers.append({
            "name": row.get("Display_Name", "Unknown"),
            "return_1d": row.get("Return_1D", 0),
            "alpha_1d": row.get("Alpha_1D", 0),
            "category": row.get("Category", "N/A"),
        })
    
    return performers


def identify_attention_waves(snapshot_df, threshold: float = -0.02) -> List[Dict[str, Any]]:
    """
    Identify waves needing attention (negative returns, low coverage, etc.).
    
    Excludes SmartSafe cash waves from attention checks.
    
    Args:
        snapshot_df: Snapshot DataFrame
        threshold: Return threshold for flagging (default -2%)
        
    Returns:
        List of dicts with wave info
    """
    if snapshot_df is None or snapshot_df.empty:
        return []
    
    attention_waves = []
    
    for _, row in snapshot_df.iterrows():
        # Skip SmartSafe cash waves - they are always stable
        wave_id = row.get("Wave_ID", "")
        flags = row.get("Flags", "")
        if "SmartSafe Cash Wave" in flags or wave_id in ["smartsafe_treasury_cash_wave", "smartsafe_tax_free_money_market_wave"]:
            continue
        
        reasons = []
        
        # Check 1D return
        return_1d = row.get("Return_1D", 0)
        if return_1d < threshold:
            reasons.append(f"1D return {_format_pct(return_1d)}")
        
        # Check data status
        status = row.get("Data_Regime_Tag", "")
        if status in ["Unavailable", "Operational"]:
            reasons.append(f"Data status: {status}")
        
        # Check coverage
        coverage = row.get("Coverage_Percent", 100)
        if coverage < 80:
            reasons.append(f"Coverage: {coverage:.0f}%")
        
        if reasons:
            attention_waves.append({
                "name": row.get("Display_Name", "Unknown"),
                "reasons": reasons,
                "return_1d": return_1d,
                "status": status,
            })
    
    return attention_waves


def get_market_regime_summary(vix_level: Optional[float] = None, spy_return: Optional[float] = None) -> str:
    """
    Generate market regime summary text.
    
    Args:
        vix_level: Current VIX level
        spy_return: SPY 1D return
        
    Returns:
        Text description of market regime
    """
    regime_parts = []
    
    if vix_level is not None:
        if vix_level < 15:
            regime_parts.append("low volatility environment (VIX < 15)")
        elif vix_level < 20:
            regime_parts.append("moderate volatility (VIX 15-20)")
        elif vix_level < 30:
            regime_parts.append("elevated volatility (VIX 20-30)")
        else:
            regime_parts.append("high volatility regime (VIX > 30)")
    
    if spy_return is not None:
        if spy_return > 0.01:
            regime_parts.append("strong market advance")
        elif spy_return > 0:
            regime_parts.append("modest market gains")
        elif spy_return > -0.01:
            regime_parts.append("modest market decline")
        else:
            regime_parts.append("significant market pullback")
    
    if regime_parts:
        return "Market conditions reflect " + " with ".join(regime_parts) + "."
    return "Market regime data not available."


def analyze_alpha_drivers(snapshot_df) -> Dict[str, Any]:
    """
    Analyze what's driving alpha across the platform.
    
    Args:
        snapshot_df: Snapshot DataFrame
        
    Returns:
        Dictionary with alpha analysis
    """
    if snapshot_df is None or snapshot_df.empty or "Alpha_1D" not in snapshot_df.columns:
        return {"positive_alpha_count": 0, "negative_alpha_count": 0, "avg_alpha": 0}
    
    positive_alpha = snapshot_df[snapshot_df["Alpha_1D"] > 0]
    negative_alpha = snapshot_df[snapshot_df["Alpha_1D"] < 0]
    
    analysis = {
        "positive_alpha_count": len(positive_alpha),
        "negative_alpha_count": len(negative_alpha),
        "avg_alpha": snapshot_df["Alpha_1D"].mean(),
        "top_alpha_category": None,
        "bottom_alpha_category": None,
    }
    
    # Analyze by category if available
    if "Category" in snapshot_df.columns:
        category_alpha = snapshot_df.groupby("Category")["Alpha_1D"].mean().sort_values(ascending=False)
        if not category_alpha.empty:
            analysis["top_alpha_category"] = category_alpha.index[0]
            analysis["bottom_alpha_category"] = category_alpha.index[-1]
    
    return analysis


def generate_executive_summary(
    snapshot_df,
    market_data: Optional[Dict[str, float]] = None
) -> str:
    """
    Generate comprehensive executive summary narrative.
    
    Args:
        snapshot_df: Snapshot DataFrame with wave metrics
        market_data: Optional dict with market context (VIX, SPY, QQQ returns, etc.)
        
    Returns:
        Formatted executive summary text
    """
    if snapshot_df is None or snapshot_df.empty:
        return "**Executive Summary**\n\nSnapshot data not available. Please refresh to generate summary."
    
    lines = []
    lines.append("**Executive Summary**")
    lines.append("")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    lines.append(f"*As of {timestamp}*")
    lines.append("")
    
    # Platform overview
    total_waves = len(snapshot_df)
    lines.append(f"**Platform Status:** {total_waves} waves actively monitored.")
    
    # Data quality summary
    if "Data_Regime_Tag" in snapshot_df.columns:
        status_counts = snapshot_df["Data_Regime_Tag"].value_counts()
        full_count = status_counts.get("Full", 0)
        partial_count = status_counts.get("Partial", 0)
        operational_count = status_counts.get("Operational", 0)
        unavailable_count = status_counts.get("Unavailable", 0)
        
        lines.append(f"**Data Coverage:** {full_count} Full, {partial_count} Partial, {operational_count} Operational, {unavailable_count} Unavailable")
    
    lines.append("")
    
    # Market regime
    if market_data:
        vix = market_data.get("VIX")
        spy_return = market_data.get("SPY_1D")
        qqq_return = market_data.get("QQQ_1D")
        
        regime_text = get_market_regime_summary(vix, spy_return)
        lines.append(f"**Market Regime:** {regime_text}")
        
        if vix is not None:
            lines.append(f"- VIX Level: {vix:.2f}")
        if spy_return is not None:
            lines.append(f"- SPY 1D: {_format_pct(spy_return)}")
        if qqq_return is not None:
            lines.append(f"- QQQ 1D: {_format_pct(qqq_return)}")
        
        lines.append("")
    
    # Top performers
    top_performers = identify_top_performers(snapshot_df, n=3)
    if top_performers:
        lines.append("**Top Outperformers Today:**")
        for i, perf in enumerate(top_performers, 1):
            lines.append(f"{i}. **{perf['name']}**: {_format_pct(perf['return_1d'])} return, {_format_pct(perf['alpha_1d'])} alpha ({perf['category']})")
        lines.append("")
    
    # Attention needed
    attention_waves = identify_attention_waves(snapshot_df, threshold=-0.02)
    if attention_waves:
        lines.append(f"**Waves Needing Attention ({len(attention_waves)}):**")
        for wave in attention_waves[:5]:  # Show top 5
            reasons_str = ", ".join(wave["reasons"])
            lines.append(f"- **{wave['name']}**: {reasons_str}")
        if len(attention_waves) > 5:
            lines.append(f"- *...and {len(attention_waves) - 5} more*")
        lines.append("")
    
    # Alpha analysis
    alpha_analysis = analyze_alpha_drivers(snapshot_df)
    lines.append("**Alpha Performance:**")
    
    avg_alpha = alpha_analysis.get("avg_alpha", 0)
    if avg_alpha > 0:
        lines.append(f"- Platform generating positive alpha: {_format_pct(avg_alpha)} average")
    elif avg_alpha < 0:
        lines.append(f"- Platform alpha challenged: {_format_pct(avg_alpha)} average")
    else:
        lines.append(f"- Platform alpha neutral")
    
    pos_count = alpha_analysis.get("positive_alpha_count", 0)
    neg_count = alpha_analysis.get("negative_alpha_count", 0)
    lines.append(f"- {pos_count} waves with positive alpha, {neg_count} with negative alpha")
    
    if alpha_analysis.get("top_alpha_category"):
        lines.append(f"- Strongest category: {alpha_analysis['top_alpha_category']}")
    if alpha_analysis.get("bottom_alpha_category"):
        lines.append(f"- Weakest category: {alpha_analysis['bottom_alpha_category']}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test summary generation
    import numpy as np
    
    # Create sample snapshot data
    sample_data = {
        "Wave_ID": ["sp500_wave", "crypto_l1_growth_wave", "gold_wave", "income_wave"],
        "Display_Name": ["S&P 500 Wave", "Crypto L1 Growth Wave", "Gold Wave", "Income Wave"],
        "Category": ["Equity", "Crypto", "Commodity", "Fixed Income"],
        "Return_1D": [0.015, 0.05, -0.01, 0.002],
        "Alpha_1D": [0.005, 0.03, -0.005, 0.001],
        "Data_Regime_Tag": ["Full", "Partial", "Full", "Full"],
        "Coverage_Percent": [100, 85, 100, 100],
    }
    
    df = pd.DataFrame(sample_data)
    
    market_data = {
        "VIX": 16.5,
        "SPY_1D": 0.01,
        "QQQ_1D": 0.012,
    }
    
    summary = generate_executive_summary(df, market_data)
    print(summary)
