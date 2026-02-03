"""
Observational Intelligence Layers
Three observational-only layers for adaptive intelligence monitoring.
No execution logic, recommendations, parameter changes, or automation.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, List


# =============================================================================
# NOTE 007: Review & Adaptation Signals
# =============================================================================
def render_review_and_adaptation_signals(
    snapshot_df: pd.DataFrame,
    selected_wave: str,
    return_cols: Dict[str, str],
    alpha_cols: Dict[str, str],
):
    """
    NOTE 007: Review & Adaptation Signals
    
    Observational layer showing performance consistency and signal quality
    across time horizons. Degrades gracefully with missing data.
    
    Args:
        snapshot_df: Portfolio snapshot data
        selected_wave: Currently selected wave name
        return_cols: Mapping of horizon labels to return column names
        alpha_cols: Mapping of horizon labels to alpha column names
    """
    st.subheader("📊 NOTE 007 — Review & Adaptation Signals")
    st.caption("Observational layer · Performance consistency monitoring")
    
    if snapshot_df is None or snapshot_df.empty:
        st.info("⚠️ No data available for review signals.")
        return
    
    # Get wave data
    wave_data = snapshot_df[snapshot_df["display_name"] == selected_wave]
    if wave_data.empty:
        st.warning(f"⚠️ No data found for wave: {selected_wave}")
        return
    
    wave_row = wave_data.iloc[0]
    
    # Calculate signal metrics (observational only)
    signals = []
    
    # 1. Return Consistency Signal
    returns = []
    for horizon, col in return_cols.items():
        if col in wave_row.index:
            val = wave_row[col]
            if pd.notna(val):
                returns.append(val)
    
    if len(returns) >= 2:
        return_std = np.std(returns)
        return_mean = np.mean(returns)
        consistency_score = 1.0 - min(return_std, 1.0) if return_mean != 0 else 0.5
        signals.append({
            "Signal": "Return Consistency",
            "Value": f"{consistency_score:.2f}",
            "Status": "Stable" if consistency_score > 0.7 else "Variable" if consistency_score > 0.4 else "Volatile"
        })
    else:
        signals.append({
            "Signal": "Return Consistency",
            "Value": "—",
            "Status": "Insufficient Data"
        })
    
    # 2. Alpha Stability Signal
    alphas = []
    for horizon, col in alpha_cols.items():
        if col in wave_row.index:
            val = wave_row[col]
            if pd.notna(val):
                alphas.append(val)
    
    if len(alphas) >= 2:
        alpha_std = np.std(alphas)
        alpha_mean = np.mean(alphas)
        stability_score = 1.0 - min(alpha_std, 1.0)
        signals.append({
            "Signal": "Alpha Stability",
            "Value": f"{stability_score:.2f}",
            "Status": "Stable" if stability_score > 0.7 else "Variable" if stability_score > 0.4 else "Volatile"
        })
    else:
        signals.append({
            "Signal": "Alpha Stability",
            "Value": "—",
            "Status": "Insufficient Data"
        })
    
    # 3. Horizon Alignment Signal (do short and long term agree?)
    if len(returns) >= 3:
        short_term = returns[0] if len(returns) > 0 else 0
        long_term = returns[-1] if len(returns) > 0 else 0
        
        if short_term != 0 and long_term != 0:
            alignment = 1.0 if (short_term > 0) == (long_term > 0) else 0.0
            signals.append({
                "Signal": "Horizon Alignment",
                "Value": "Aligned" if alignment > 0.5 else "Divergent",
                "Status": "Consistent" if alignment > 0.5 else "Mixed"
            })
        else:
            signals.append({
                "Signal": "Horizon Alignment",
                "Value": "—",
                "Status": "Insufficient Data"
            })
    else:
        signals.append({
            "Signal": "Horizon Alignment",
            "Value": "—",
            "Status": "Insufficient Data"
        })
    
    # 4. Data Completeness Signal
    total_expected = len(return_cols) + len(alpha_cols)
    available = 0
    for col in list(return_cols.values()) + list(alpha_cols.values()):
        if col in wave_row.index and pd.notna(wave_row[col]):
            available += 1
    
    completeness = available / total_expected if total_expected > 0 else 0
    signals.append({
        "Signal": "Data Completeness",
        "Value": f"{completeness:.0%}",
        "Status": "Complete" if completeness > 0.9 else "Partial" if completeness > 0.5 else "Limited"
    })
    
    # Display signals
    if signals:
        signals_df = pd.DataFrame(signals)
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
    else:
        st.info("⚠️ No signals could be calculated.")
    
    # Interpretation (observational only, no recommendations)
    st.markdown("""
    **Interpretation Guide**
    
    • **Return Consistency**: Measures variation in returns across time horizons  
    • **Alpha Stability**: Tracks consistency of alpha generation over time  
    • **Horizon Alignment**: Observes whether short and long-term trends agree  
    • **Data Completeness**: Indicates availability of required data points
    
    *This layer is observational only and does not provide recommendations.*
    """)


# =============================================================================
# NOTE 002: Decision Outcomes & Results Summary
# =============================================================================
def render_decision_outcomes_summary(
    snapshot_df: pd.DataFrame,
    selected_wave: str,
    return_cols: Dict[str, str],
    alpha_cols: Dict[str, str],
):
    """
    NOTE 002: Decision Outcomes & Results Summary
    
    Observational layer showing historical performance outcomes across
    different time windows. Degrades gracefully with missing data.
    
    Args:
        snapshot_df: Portfolio snapshot data
        selected_wave: Currently selected wave name
        return_cols: Mapping of horizon labels to return column names
        alpha_cols: Mapping of horizon labels to alpha column names
    """
    st.subheader("📈 NOTE 002 — Decision Outcomes & Results Summary")
    st.caption("Observational layer · Historical performance outcomes")
    
    if snapshot_df is None or snapshot_df.empty:
        st.info("⚠️ No data available for outcomes summary.")
        return
    
    # Get wave data
    wave_data = snapshot_df[snapshot_df["display_name"] == selected_wave]
    if wave_data.empty:
        st.warning(f"⚠️ No data found for wave: {selected_wave}")
        return
    
    wave_row = wave_data.iloc[0]
    
    # Calculate outcome metrics (observational only)
    outcomes = []
    
    for horizon, return_col in return_cols.items():
        alpha_col = alpha_cols.get(horizon)
        
        return_val = None
        alpha_val = None
        
        if return_col in wave_row.index:
            return_val = wave_row[return_col]
        
        if alpha_col and alpha_col in wave_row.index:
            alpha_val = wave_row[alpha_col]
        
        # Determine outcome status
        if pd.notna(return_val):
            return_pct = f"{return_val * 100:.2f}%"
            return_status = "Positive" if return_val > 0 else "Negative" if return_val < 0 else "Flat"
        else:
            return_pct = "—"
            return_status = "No Data"
        
        if pd.notna(alpha_val):
            alpha_pct = f"{alpha_val * 100:.2f}%"
            alpha_status = "Positive" if alpha_val > 0 else "Negative" if alpha_val < 0 else "Flat"
        else:
            alpha_pct = "—"
            alpha_status = "No Data"
        
        outcomes.append({
            "Horizon": horizon,
            "Return": return_pct,
            "Return Status": return_status,
            "Alpha": alpha_pct,
            "Alpha Status": alpha_status,
        })
    
    # Display outcomes
    if outcomes:
        outcomes_df = pd.DataFrame(outcomes)
        st.dataframe(outcomes_df, use_container_width=True, hide_index=True)
    else:
        st.info("⚠️ No outcomes could be calculated.")
    
    # Portfolio-level summary (observational)
    st.markdown("#### Portfolio-Level Outcomes")
    
    portfolio_outcomes = []
    for horizon, return_col in return_cols.items():
        alpha_col = alpha_cols.get(horizon)
        
        # Calculate portfolio averages
        if return_col in snapshot_df.columns:
            portfolio_return = snapshot_df[return_col].mean(skipna=True)
            if pd.notna(portfolio_return):
                portfolio_outcomes.append({
                    "Horizon": horizon,
                    "Avg Return": f"{portfolio_return * 100:.2f}%",
                    "Wave Count": int(snapshot_df[return_col].notna().sum()),
                })
            else:
                portfolio_outcomes.append({
                    "Horizon": horizon,
                    "Avg Return": "—",
                    "Wave Count": 0,
                })
        else:
            portfolio_outcomes.append({
                "Horizon": horizon,
                "Avg Return": "—",
                "Wave Count": 0,
            })
    
    if portfolio_outcomes:
        portfolio_df = pd.DataFrame(portfolio_outcomes)
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
    
    # Interpretation (observational only)
    st.markdown("""
    **Interpretation Guide**
    
    • **Return**: Observed total return over the specified horizon  
    • **Alpha**: Observed excess return relative to benchmark  
    • **Portfolio-Level**: Equal-weighted average across all waves  
    • **Wave Count**: Number of waves with available data for each horizon
    
    *This layer is observational only and does not provide recommendations.*
    """)


# =============================================================================
# NOTE 010: Volatility Stress Probability Indicator
# =============================================================================
def render_volatility_stress_indicator(
    snapshot_df: pd.DataFrame,
    selected_wave: str,
    return_cols: Dict[str, str],
):
    """
    NOTE 010: Volatility Stress Probability Indicator
    
    Observational layer showing volatility patterns and stress indicators
    based on return dispersion. Degrades gracefully with missing data.
    
    Args:
        snapshot_df: Portfolio snapshot data
        selected_wave: Currently selected wave name
        return_cols: Mapping of horizon labels to return column names
    """
    st.subheader("⚡ NOTE 010 — Volatility Stress Probability Indicator")
    st.caption("Observational layer · Volatility pattern monitoring")
    
    if snapshot_df is None or snapshot_df.empty:
        st.info("⚠️ No data available for volatility indicators.")
        return
    
    # Get wave data
    wave_data = snapshot_df[snapshot_df["display_name"] == selected_wave]
    if wave_data.empty:
        st.warning(f"⚠️ No data found for wave: {selected_wave}")
        return
    
    wave_row = wave_data.iloc[0]
    
    # Calculate volatility indicators (observational only)
    indicators = []
    
    # 1. Return Dispersion (proxy for realized volatility)
    returns = []
    for horizon, col in return_cols.items():
        if col in wave_row.index:
            val = wave_row[col]
            if pd.notna(val):
                returns.append((horizon, val))
    
    if len(returns) >= 2:
        return_values = [r[1] for r in returns]
        dispersion = np.std(return_values)
        
        # Classify dispersion level
        if dispersion > 0.15:
            stress_level = "High"
        elif dispersion > 0.08:
            stress_level = "Moderate"
        else:
            stress_level = "Low"
        
        indicators.append({
            "Indicator": "Return Dispersion",
            "Value": f"{dispersion:.4f}",
            "Level": stress_level,
        })
    else:
        indicators.append({
            "Indicator": "Return Dispersion",
            "Value": "—",
            "Level": "Insufficient Data",
        })
    
    # 2. Drawdown Observation (largest negative return)
    if len(returns) > 0:
        max_drawdown = min([r[1] for r in returns])
        drawdown_horizon = [r[0] for r in returns if r[1] == max_drawdown][0]
        
        if max_drawdown < -0.10:
            drawdown_severity = "Severe"
        elif max_drawdown < -0.05:
            drawdown_severity = "Moderate"
        elif max_drawdown < 0:
            drawdown_severity = "Minor"
        else:
            drawdown_severity = "None"
        
        indicators.append({
            "Indicator": f"Max Drawdown ({drawdown_horizon})",
            "Value": f"{max_drawdown * 100:.2f}%",
            "Level": drawdown_severity,
        })
    else:
        indicators.append({
            "Indicator": "Max Drawdown",
            "Value": "—",
            "Level": "Insufficient Data",
        })
    
    # 3. Portfolio Volatility Context
    portfolio_returns = []
    for horizon, col in return_cols.items():
        if col in snapshot_df.columns:
            wave_returns = snapshot_df[col].dropna()
            if len(wave_returns) > 0:
                portfolio_std = wave_returns.std()
                if pd.notna(portfolio_std):
                    portfolio_returns.append((horizon, portfolio_std))
    
    if len(portfolio_returns) > 0:
        avg_portfolio_vol = np.mean([r[1] for r in portfolio_returns])
        
        if avg_portfolio_vol > 0.12:
            portfolio_stress = "High"
        elif avg_portfolio_vol > 0.06:
            portfolio_stress = "Moderate"
        else:
            portfolio_stress = "Low"
        
        indicators.append({
            "Indicator": "Portfolio Volatility",
            "Value": f"{avg_portfolio_vol:.4f}",
            "Level": portfolio_stress,
        })
    else:
        indicators.append({
            "Indicator": "Portfolio Volatility",
            "Value": "—",
            "Level": "Insufficient Data",
        })
    
    # 4. Relative Volatility (wave vs portfolio)
    if len(returns) >= 2 and len(portfolio_returns) > 0:
        wave_vol = np.std([r[1] for r in returns])
        portfolio_vol = np.mean([r[1] for r in portfolio_returns])
        
        if portfolio_vol > 0:
            rel_vol = wave_vol / portfolio_vol
            
            if rel_vol > 1.5:
                rel_status = "Higher than Portfolio"
            elif rel_vol > 0.8:
                rel_status = "Similar to Portfolio"
            else:
                rel_status = "Lower than Portfolio"
            
            indicators.append({
                "Indicator": "Relative Volatility",
                "Value": f"{rel_vol:.2f}x",
                "Level": rel_status,
            })
        else:
            indicators.append({
                "Indicator": "Relative Volatility",
                "Value": "—",
                "Level": "Cannot Calculate",
            })
    else:
        indicators.append({
            "Indicator": "Relative Volatility",
            "Value": "—",
            "Level": "Insufficient Data",
        })
    
    # Display indicators
    if indicators:
        indicators_df = pd.DataFrame(indicators)
        st.dataframe(indicators_df, use_container_width=True, hide_index=True)
    else:
        st.info("⚠️ No indicators could be calculated.")
    
    # Interpretation (observational only)
    st.markdown("""
    **Interpretation Guide**
    
    • **Return Dispersion**: Variation in returns across time horizons (volatility proxy)  
    • **Max Drawdown**: Largest observed negative return in any horizon  
    • **Portfolio Volatility**: Average return variation across all waves  
    • **Relative Volatility**: Wave volatility compared to portfolio average
    
    **Stress Levels** (observational thresholds):
    - **High**: Significant variation observed, may indicate turbulent conditions
    - **Moderate**: Normal market variation observed
    - **Low**: Minimal variation observed, stable conditions
    
    *This layer is observational only and does not provide recommendations.*
    """)
