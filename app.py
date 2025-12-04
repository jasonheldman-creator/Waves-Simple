import streamlit as st
import pandas as pd

from waves_equity_universe_v2 import run_equity_waves, WAVES_CONFIG


# -------------------------------------------------------------------
# Static documentation for how each Wave is managed
# (You can tweak wording any time without touching the engine.)
# -------------------------------------------------------------------

WAVE_DOCS = {
    "SPX": {
        "objective": "Track the S&P 500 with extremely tight tracking error while harvesting small, persistent alpha from better execution and rebalancing.",
        "benchmark": "SPY (S&P 500 ETF)",
        "universe": "Current S&P 500 constituents (large-cap US equities).",
        "management": [
            "Maintain broad, diversified exposure to all S&P 500 names.",
            "Allow small over/under-weights vs SPY where there is strong leadership or momentum evidence.",
            "Stay fully invested in equities except in extreme volatility regimes (handled at the SmartSafe/Vector layer).",
        ],
        "rebalancing": [
            "Baseline: align with index quarterly or when cumulative drift > 2–3% at the Wave level.",
            "Opportunistic: rebalance earlier when leadership rotation is detected (e.g., sector/factor leaders breaking out).",
            "Minimize turnover; favor partial, incremental trades over full resets.",
        ],
        "alpha_drivers": [
            "Smoother and slightly faster rebalancing than the underlying ETF.",
            "Avoiding forced-seller behavior around index reconstitutions where spreads/impact are wide.",
            "Small tilts toward persistent quality, profitability, and strong balance sheets.",
        ],
        "risk": [
            "Target beta ≈ 1.0 vs S&P 500.",
            "No single stock > 8% of Wave NAV; flag if index weight exceeds that.",
            "Sector weights stay within ±5% of benchmark, unless explicitly overridden by Private Logic mode.",
        ],
    },
    "USMKT": {
        "objective": "Own the full US equity market from mega-cap to micro-cap with simple, rules-based tilts toward quality and profitability.",
        "benchmark": "VTI (Total US Market ETF)",
        "universe": "Total US equity universe, anchored to broad-market ETFs or CRSP/FTSE indices.",
        "management": [
            "Blend S&P 500, mid-cap, small-cap, and micro-cap exposures using your master stock sheet.",
            "Overweight persistent quality names; underweight highly distressed or structurally impaired names.",
            "Maintain liquidity discipline; cap allocation to very illiquid names.",
        ],
        "rebalancing": [
            "Quarterly baseline rebalance; monthly light drift corrections.",
            "Lift weights slowly in segments showing improving breadth and leadership.",
        ],
        "alpha_drivers": [
            "Better handling of small-cap and micro-cap liquidity than vanilla index funds.",
            "Avoidance of names with elevated bankruptcy/earnings blow-up risk.",
        ],
        "risk": [
            "Target beta ≈ 1.0 vs total-market benchmark.",
            "Guardrails on small-cap and micro-cap exposure to avoid liquidity traps.",
        ],
    },
    "LGRW": {
        "objective": "Concentrated exposure to large-cap growth leaders (software, platforms, compounders) with risk-aware drawdown controls.",
        "benchmark": "QQQ or similar large-growth ETF.",
        "universe": "Large-cap US growth stocks; tech, communications, consumer growth leaders.",
        "management": [
            "Focus on durable revenue growth, strong balance sheets, and network effects.",
            "Allow more concentrated positions in top 25–40 names.",
        ],
        "rebalancing": [
            "Faster response to trend and leadership changes than a static growth ETF.",
            "Trim extended winners on parabolic moves; add on controlled pullbacks.",
        ],
        "alpha_drivers": [
            "Momentum/leadership continuation overlays.",
            "Faster removal of broken stories vs index methodology.",
        ],
        "risk": [
            "Beta can run > 1.0 vs S&P 500 but is monitored vs growth benchmark.",
            "Explicit drawdown monitoring; throttle position size after large adverse moves.",
        ],
    },
    "SCG": {
        "objective": "Capture the small-cap growth premium while aggressively filtering out junk and low-quality names.",
        "benchmark": "IWO (Russell 2000 Growth) or similar.",
        "universe": "US small-cap growth stocks, initially seeded from Russell 2000 and related indices.",
        "management": [
            "Favor profitable or near-profitable small caps with real businesses.",
            "Systematically underweight highly dilutive, structurally unprofitable names.",
        ],
        "rebalancing": [
            "Higher turnover allowed versus large-cap Waves; small caps move faster.",
            "Rotate away from deteriorating balance sheets and broken price structures quickly.",
        ],
        "alpha_drivers": [
            "Quality and survivorship screening inside the small-cap growth universe.",
            "Avoidance of the worst “lottery ticket” names that drag index returns.",
        ],
        "risk": [
            "Position-size limits to respect liquidity (e.g., max % of average daily volume).",
            "Wave-level volatility expected to be higher; monitored and throttled by Vector/SmartSafe.",
        ],
    },
    "SMID": {
        "objective": "Bridge small and mid-cap growth exposures with smoother volatility and broader diversification than pure small-cap growth.",
        "benchmark": "IJT or similar small/mid growth ETF.",
        "universe": "US small-mid growth stocks, tilted slightly up-cap from SCG.",
        "management": [
            "Blend higher-quality small caps with stronger, more established mid-caps.",
            "Use this Wave as the core growth complement to SPX and USMKT.",
        ],
        "rebalancing": [
            "Moderate turnover; react when leadership clearly shifts between small and mid-cap cohorts.",
        ],
        "alpha_drivers": [
            "Dynamic tilt between small and mid depending on breadth and factor regimes.",
        ],
        "risk": [
            "Beta around 1.05–1.15 vs S&P 500; watched closely vs growth benchmarks.",
        ],
    },
    "AITECH": {
        "objective": "High-conviction exposure to AI, cloud, data, and next-gen compute leaders.",
        "benchmark": "QQQ or specialized AI/tech ETF basket.",
        "universe": "AI-linked semis, cloud platforms, infra software, data/analytics leaders.",
        "management": [
            "Concentrated portfolio of 20–40 names with strong secular AI tailwinds.",
            "Allow thematic concentration but enforce single-name limits.",
        ],
        "rebalancing": [
            "Aggressive response to regime changes (hardware vs software leadership, etc.).",
        ],
        "alpha_drivers": [
            "Faster capture of new leaders than slow-moving index methodologies.",
        ],
        "risk": [
            "High volatility by design; vector-level controls will cap portfolio-wide exposure to this Wave.",
        ],
    },
    "ROBO": {
        "objective": "Robotics, automation, industrial AI, and advanced manufacturing equities.",
        "benchmark": "BOTZ or similar robotics ETF.",
        "universe": "Robotics, sensors, automation OEMs, enabling software.",
        "management": [
            "Diversify across hardware, software, and enabling supply-chain vendors.",
        ],
        "rebalancing": [
            "Monitor capex cycles and industrial order books; de-risk when macro turns sharply down.",
        ],
        "alpha_drivers": [
            "Avoid overcrowded single-theme baskets; mix cyclical and secular stories.",
        ],
        "risk": [
            "Cyclical drawdown risk; position size scaled based on macro and earnings volatility.",
        ],
    },
    "ENERGYF": {
        "objective": "Future power, clean energy, grid infra, and transition-linked equities.",
        "benchmark": "ICLN or similar clean-energy ETF.",
        "universe": "Renewables developers, grid and storage companies, enabling tech.",
        "management": [
            "Blend volatile pure-play names with more stable grid and infra names.",
        ],
        "rebalancing": [
            "Scale risk up/down based on policy, rate environment, and factor regimes.",
        ],
        "alpha_drivers": [
            "Dynamic sizing across sub-themes (solar, wind, storage, grid, etc.) instead of static weights.",
        ],
        "risk": [
            "High single-name and regulatory risk; tight position limits and stop-loss logic.",
        ],
    },
    "EQINC": {
        "objective": "High-quality global equity income with a focus on dividend growth and resilience.",
        "benchmark": "SCHD or similar dividend-equity ETF.",
        "universe": "Dividend-paying global large/mid-cap stocks with sustainable payout ratios.",
        "management": [
            "Emphasize dividend growth and balance-sheet strength over raw yield.",
            "Underweight yield traps and structurally impaired high-yield names.",
        ],
        "rebalancing": [
            "Gradual re-optimization as dividend cuts, hikes, or buyback policies change.",
        ],
        "alpha_drivers": [
            "Quality screen and dividend-safety overlay vs yield-only strategies.",
        ],
        "risk": [
            "Lower beta target than broad equity (≈0.8–0.9 vs S&P).",
        ],
    },
    "INTL": {
        "objective": "International developed + EM equity exposure with smart country/sector tilts.",
        "benchmark": "VEA + EM blend or a global ex-US ETF basket.",
        "universe": "Non-US equities: developed markets plus selectively EM.",
        "management": [
            "Avoid structurally impaired markets/currencies; overweight higher-quality regimes.",
            "Blend country, sector, and currency views into a coherent global ex-US sleeve.",
        ],
        "rebalancing": [
            "Adjust regional weights as macro, FX, and policy regimes evolve.",
        ],
        "alpha_drivers": [
            "Country/sector selection alpha, not just stock-picking inside a static ex-US index.",
        ],
        "risk": [
            "Explicit tracking of FX and political risk; cap exposure to fragile regimes.",
        ],
    },
}


# -------------------------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence – Equity Waves Console",
    layout="wide",
)

st.title("🌊 Equity Waves Console")
st.caption("Live view of all 10 Equity Waves (prototype)")

st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Controls")
    if st.button("🔁 Run Equity Waves", use_container_width=True):
        with st.spinner("Running WAVES engine…"):
            df_result = run_equity_waves()
        st.session_state["equity_waves_df"] = df_result
        st.success("Run complete.")
    else:
        df_result = st.session_state.get("equity_waves_df")

with col_right:
    st.subheader("Status")
    configured = [w for w in WAVES_CONFIG if w.holdings_csv_url]
    missing = [w for w in WAVES_CONFIG if not w.holdings_csv_url]

    st.markdown(f"**Configured Waves:** {len(configured)} / {len(WAVES_CONFIG)}")
    if configured:
        st.markdown(
            "- " + "\n- ".join(f"`{w.code}` – {w.name} (bench: `{w.benchmark}`)" for w in configured)
        )
    if missing:
        st.markdown("**Missing holdings URLs (not yet live):**")
        st.markdown(
            "- " + "\n- ".join(f"`{w.code}` – {w.name}" for w in missing)
        )

st.divider()

# -------------------------------------------------------------------
# SUMMARY TABLE + ALPHA CHART
# -------------------------------------------------------------------

if df_result is not None and isinstance(df_result, pd.DataFrame) and not df_result.empty:
    # Format for display
    df_disp = df_result.copy()
    for col in ["wave_return", "benchmark_return", "alpha"]:
        if col in df_disp.columns:
            df_disp[col] = (df_disp[col] * 100).round(2)

    df_disp.rename(
        columns={
            "code": "Wave",
            "name": "Name",
            "benchmark": "Benchmark",
            "nav": "NAV ($)",
            "wave_return": "Wave Return (%)",
            "benchmark_return": "Benchmark Return (%)",
            "alpha": "Alpha (%)",
        },
        inplace=True,
    )

    st.subheader("Wave Performance Snapshot")
    st.dataframe(df_disp, use_container_width=True)

    if "alpha" in df_result.columns:
        st.subheader("Alpha by Wave")
        alpha_series = df_result.set_index("code")["alpha"]
        st.bar_chart(alpha_series)
else:
    st.info("No results yet. Click **Run Equity Waves** to generate a snapshot.")


st.divider()

# -------------------------------------------------------------------
# DETAILED MANAGEMENT RULES / PLAYBOOK SECTION
# -------------------------------------------------------------------

st.header("📘 Wave Playbooks – How Each Wave Is Managed")

st.write(
    "Below are the management rules, objectives, and risk constraints for each Equity Wave. "
    "These are the human-readable playbooks that sit on top of the engine logic."
)

for wave in WAVES_CONFIG:
    docs = WAVE_DOCS.get(wave.code, {})
    with st.expander(f"{wave.code} — {wave.name}", expanded=False):
        st.markdown(f"**Benchmark:** `{wave.benchmark}`")

        if "objective" in docs:
            st.markdown(f"**Objective**  \n{docs['objective']}")

        if "universe" in docs:
            st.markdown(f"**Universe**  \n{docs['universe']}")

        def bullet_section(title: str, key: str):
            items = docs.get(key, [])
            if items:
                st.markdown(f"**{title}**")
                st.markdown("- " + "\n- ".join(items))

        bullet_section("Management Style", "management")
        bullet_section("Rebalancing Rules", "rebalancing")
        bullet_section("Alpha Drivers", "alpha_drivers")
        bullet_section("Risk & Guardrails", "risk")

        if not docs:
            st.info("No detailed playbook written yet for this Wave. We can add it anytime.")

st.caption("Prototype console – not investment advice. For internal WAVES Intelligence™ use only.")