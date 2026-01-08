#!/usr/bin/env python3
"""
Demo script showing crypto overlay diagnostics in action.

This script demonstrates:
1. Crypto wave identification
2. Overlay field extraction
3. Diagnostic formatting for UI display
"""

import sys
import pandas as pd

# Import crypto diagnostics helpers
from helpers.crypto_overlay_diagnostics import (
    get_crypto_overlay_diagnostics,
    format_crypto_regime,
    format_crypto_exposure,
    format_crypto_volatility,
    format_crypto_overlay_status,
    get_crypto_overlay_minimum_exposure
)

# Import waves engine
try:
    import waves_engine as we
except ImportError:
    print("❌ waves_engine not available")
    sys.exit(1)


def demo_crypto_wave_identification():
    """Demo: Identify crypto waves."""
    print("\n" + "="*80)
    print("CRYPTO WAVE IDENTIFICATION")
    print("="*80)
    
    test_waves = [
        "Crypto L1 Growth Wave",
        "Crypto DeFi Growth Wave",
        "Crypto Income Wave",
        "S&P 500 Wave",
        "Income Wave"
    ]
    
    for wave in test_waves:
        is_crypto = we._is_crypto_wave(wave)
        is_growth = we._is_crypto_growth_wave(wave)
        is_income = we._is_crypto_income_wave(wave)
        min_exp = get_crypto_overlay_minimum_exposure(wave)
        
        status = "🔷 CRYPTO" if is_crypto else "📊 EQUITY"
        category = ""
        if is_growth:
            category = "Growth"
        elif is_income:
            category = "Income"
        
        print(f"\n{status} {wave}")
        if is_crypto:
            print(f"  Category: {category}")
            print(f"  Min Exposure: {min_exp * 100:.0f}%")


def demo_crypto_overlay_diagnostics():
    """Demo: Show crypto overlay diagnostics."""
    print("\n" + "="*80)
    print("CRYPTO OVERLAY DIAGNOSTICS")
    print("="*80)
    
    crypto_waves = [
        "Crypto L1 Growth Wave",
        "Crypto Income Wave"
    ]
    
    for wave_name in crypto_waves:
        print(f"\n{'─'*80}")
        print(f"Wave: {wave_name}")
        print(f"{'─'*80}")
        
        try:
            # Compute wave history with diagnostics
            result = we.compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                days=90,
                include_diagnostics=True
            )
            
            if result.empty:
                print("⚠️  No data available (expected in test environment)")
                continue
            
            # Extract diagnostics
            diag = get_crypto_overlay_diagnostics(wave_name, result)
            
            if diag is None:
                print("⚠️  Not a crypto wave")
                continue
            
            # Display overlay status
            print(f"\nOverlay Status: {format_crypto_overlay_status(diag)}")
            
            if diag.get('overlay_active'):
                print(f"\n📊 Diagnostics:")
                print(f"  Regime:     {format_crypto_regime(diag)}")
                print(f"  Exposure:   {format_crypto_exposure(diag)}")
                print(f"  Volatility: {format_crypto_volatility(diag)}")
                
                # Show raw values
                print(f"\n🔧 Raw Values:")
                for key, value in diag.items():
                    if key not in ['is_crypto', 'is_crypto_growth', 'is_crypto_income', 'overlay_active']:
                        print(f"  {key}: {value}")
            else:
                reason = diag.get('reason', 'unknown')
                print(f"⚠️  Overlay inactive: {reason}")
                
        except Exception as e:
            print(f"❌ Error computing diagnostics: {e}")


def demo_regime_classification():
    """Demo: Show regime classification logic."""
    print("\n" + "="*80)
    print("REGIME CLASSIFICATION")
    print("="*80)
    
    print("\n📈 Crypto Trend Regimes:")
    test_trends = [0.25, 0.12, 0.02, -0.08, -0.22]
    for trend in test_trends:
        regime = we._crypto_trend_regime(trend)
        print(f"  {trend:+.2f} → {regime}")
    
    print("\n🌊 Crypto Volatility States:")
    test_vols = [0.25, 0.40, 0.60, 1.00, 1.50]
    for vol in test_vols:
        state = we._crypto_volatility_state(vol)
        print(f"  {vol:.2f} → {state}")
    
    print("\n💧 Crypto Liquidity States:")
    test_liquidity = [2.5, 1.5, 0.8]
    for liq in test_liquidity:
        state = we._crypto_liquidity_state(liq)
        print(f"  {liq:.1f}x → {state}")


def demo_ui_integration():
    """Demo: Show how to integrate with UI."""
    print("\n" + "="*80)
    print("UI INTEGRATION EXAMPLE")
    print("="*80)
    
    print(r"""
In your Streamlit app.py, add crypto overlay diagnostics to wave panels:

```python
from helpers.crypto_overlay_diagnostics import (
    get_crypto_overlay_diagnostics,
    format_crypto_regime,
    format_crypto_exposure,
    format_crypto_volatility
)

# In your wave panel rendering function:
def render_wave_panel(wave_name: str, mode: str):
    # ... existing code ...
    
    # Add crypto overlay diagnostics
    if we._is_crypto_wave(wave_name):
        wave_history = we.compute_history_nav(
            wave_name=wave_name,
            mode=mode,
            days=90,
            include_diagnostics=True
        )
        
        crypto_diag = get_crypto_overlay_diagnostics(wave_name, wave_history)
        
        if crypto_diag and crypto_diag.get('overlay_active'):
            st.subheader("🔷 Crypto Overlay Status")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Vol Regime", format_crypto_regime(crypto_diag))
            
            with col2:
                st.metric("Exposure", format_crypto_exposure(crypto_diag))
            
            with col3:
                st.metric("Volatility", format_crypto_volatility(crypto_diag))
```

This displays:
- Current volatility regime (e.g., "↗️ Uptrend")
- Current exposure level (e.g., "75%")
- Volatility state (e.g., "🟡 Normal (60.0%)")
""")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("CRYPTO VOLATILITY OVERLAY DIAGNOSTICS DEMO")
    print("="*80)
    
    try:
        demo_crypto_wave_identification()
        demo_regime_classification()
        demo_crypto_overlay_diagnostics()
        demo_ui_integration()
        
        print("\n" + "="*80)
        print("✅ Demo complete!")
        print("="*80)
        print("\nNext steps:")
        print("1. Run tests: python3 -m pytest test_crypto_volatility_overlay.py -v")
        print("2. Read guide: cat CRYPTO_OVERLAY_INTEGRATION_GUIDE.md")
        print("3. Integrate into app.py using helpers/crypto_overlay_diagnostics.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
