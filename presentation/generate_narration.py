#!/usr/bin/env python3
"""
Generate TTS-ready narration files for each slide.
Each file contains the voiceover text for one slide.
"""

import os

# Narration content for each slide
NARRATIONS = {
    "slide_01_title_introduction.txt": """Good afternoon. I'm Vector, your guide to WAVES Intelligence. Over the next seven to nine minutes, I'll walk you through our institutional-grade portfolio analytics platform—designed specifically for sophisticated investors who demand transparency, precision, and governance-ready insights.

WAVES Intelligence represents a fundamental shift in how institutions understand and explain portfolio performance. This isn't just another analytics dashboard. It's a complete decision-making ecosystem built on three core principles: no-predict constraints, deterministic outputs, and architectural transparency.""",

    "slide_02_challenge.txt": """Traditional portfolio analytics platforms present a challenge. They often deliver attribution reports that are opaque, difficult to validate, and unreliable under changing market regimes. Institutional investors need more than backward-looking performance metrics—they need governance-grade truth outputs that explain exactly where returns come from, under what conditions they persist, and what risks threaten their durability.

Most platforms optimize for prediction. WAVES Intelligence takes a different approach. We embrace a no-predict constraint, focusing instead on deterministic, explainable insights that boards and oversight committees can actually use.""",

    "slide_03_solution.txt": """WAVES Intelligence provides four integrated capabilities:

First, Alpha Attribution—our Vector Truth Layer decomposes returns into structural effects and residual strategy returns, with complete regime awareness.

Second, Performance Deep Dive—comprehensive wave performance charts, metrics, and historical tracking with synthetic data support for seamless onboarding.

Third, Decision Ledger—a governance layer that tracks every decision, contract, and oversight action, creating an immutable audit trail.

And fourth, Board-Ready Reporting—institutional PDF board packs that translate complex analytics into clear, actionable narratives.

Each component is built on the Wave ID system, our canonical identifier framework that ensures data integrity across all analytics.""",

    "slide_04_wave_id.txt": """The foundation of WAVES Intelligence is the Wave ID system. Think of Wave IDs as permanent identifiers for every investment strategy or portfolio composition we track.

Each Wave has three data layers: historical performance data, configuration parameters including benchmark definitions, and dynamic weightings that evolve over time.

The system supports both real market data and synthetic placeholder data, marked explicitly with transparency indicators. This means institutions can onboard immediately, with all analytics fully functional, while real data integration happens in the background.

The Wave ID architecture ensures backward compatibility, so legacy data integrates seamlessly. No disruption. No data migration friction.""",

    "slide_05_vector_truth.txt": """Let me explain the Vector Truth Layer—our attribution engine.

Traditional attribution tells you what happened. Vector Truth tells you why it happened, with governance-ready reliability signals.

We separate returns into two categories: structural effects and residual strategy returns.

Structural effects include capital preservation overlays—VIX regime controls, SmartSafe gating—and benchmark construction offsets. These are non-alpha by design. They're governance controls.

Residual strategy returns reflect timing, exposure scaling, volatility control, and regime management after those structural overlays. We don't attribute these to asset selection or static weights. We show them as integrated strategy decisions.

This separation is critical. It prevents alpha inflation, ensures intellectual honesty, and gives boards the transparency they need.""",

    "slide_06_alpha_decomposition.txt": """Here's how it works in practice.

Imagine a Wave delivers eight percent total excess return over its benchmark. Vector Truth decomposes this into:

Security selection alpha of five percent—the exposure-adjusted return from strategic positioning.

Exposure management alpha of two percent—gains from timing and scaling decisions.

Capital preservation effect of one and a half percent—the structural benefit from VIX overlays and regime-aware cash sweeps.

And a benchmark construction offset of negative point five percent—the expected structural drag from benchmark composition choices.

Notice how the structural components largely offset. That's intentional. The residual strategy return of seven point five percent is what's truly alpha-eligible after governance controls.

Vector Truth provides this breakdown with full regime attribution, showing how much alpha was earned in risk-on versus risk-off environments. This tells you whether the strategy is durable across market conditions.""",

    "slide_07_regime_attribution.txt": """Regime attribution is essential for assessing durability.

Vector Truth tracks every return period's market regime—risk-on or risk-off—based on VIX levels and volatility signals.

We then show cumulative alpha earned in each regime. If a strategy delivers four percent alpha in risk-on and three percent in risk-off, that's balanced. It suggests resilience.

But if a strategy shows six percent alpha in risk-on and negative one percent in risk-off, that's a red flag. The strategy is regime-dependent. Its durability is questionable.

We calculate a fragility score from zero to one. Low fragility means the alpha is structurally sound. High fragility means it's vulnerable to regime shifts, dispersion collapse, or volatility suppression.

Boards need to know this. Traditional attribution hides it. Vector Truth surfaces it explicitly.""",

    "slide_08_decision_ledger.txt": """The Decision Ledger provides governance accountability.

Every portfolio decision—entry, exit, rebalance—is logged with timestamp, wave identifier, and rationale. Every alpha contract—the expected return profile and risk parameters—is recorded at inception.

The Ledger tracks outcome reconciliation. Did the wave deliver on its contract? If not, why not? Was it regime conditions? Execution friction? Strategy drift?

This creates an immutable audit trail. Oversight committees can trace any current position back to its original decision rationale. No black boxes. No memory gaps.

The Ledger also supports performance attribution at the decision level. You can see which decisions contributed to returns and which detracted, with full transparency.

This is governance-native design. It treats compliance, transparency, and oversight as first-class requirements, not afterthoughts.""",

    "slide_09_performance_deep_dive.txt": """The Performance Deep Dive console provides real-time visibility into every Wave.

You see historical return charts with regime overlays—shaded regions indicating risk-on versus risk-off periods.

You see rolling alpha metrics, drawdown analysis, and volatility tracking, all benchmarked against governed composite indices.

The system highlights synthetic data with clear visual indicators. No one is misled. If a Wave is using placeholder data while real market integration is in progress, you'll know immediately.

The WaveScore system provides a proprietary performance scoring algorithm that combines return magnitude, consistency, regime balance, and risk efficiency into a single digestible metric.

Every chart, every table, every metric is exportable. CSV downloads, PDF board packs, institutional-ready formatting. No manual data wrangling required.""",

    "slide_10_board_pack.txt": """WAVES Intelligence generates comprehensive PDF board packs automatically.

Each board pack includes an executive summary, wave-by-wave performance analysis, Vector Truth attribution reports, and Decision Ledger highlights.

The system uses deterministic narrative templates. The language is stable, predictable, and governance-appropriate. No marketing fluff. No predictive claims. Just clear, factual reporting.

The board packs integrate seamlessly with existing reporting workflows. They're designed for quarterly board meetings, investment committee reviews, and regulatory filings.

Institutions can customize branding, adjust content sections, and control disclosure levels. But the underlying analytics remain deterministic and auditable.

This isn't just automation. It's institutional-grade documentation at scale.""",

    "slide_11_mission_control.txt": """Mission Control is your real-time risk monitoring hub.

It displays current VIX levels, regime classifications, and exposure adjustments across all Waves.

The VIX Regime Overlay system automatically classifies market conditions as risk-on or risk-off based on trailing volatility. When regimes shift, Mission Control highlights which Waves are affected and how exposure levels adjust.

The SmartSafe gating system is visible here. When volatility spikes above critical thresholds, capital preservation overlays activate. You see this in real time. No surprises.

Mission Control also tracks capital deployment across Waves, identifies concentration risks, and monitors benchmark drift. If a governed benchmark is drifting from its snapshot definition, you're alerted immediately.

This is proactive risk management. Not reactive firefighting.""",

    "slide_12_implementation.txt": """Implementation is designed for minimal friction.

WAVES Intelligence supports multiple data integration pathways. You can connect via CSV file uploads, API endpoints for real-time market data, or leverage our synthetic data seeding for immediate onboarding.

The system is built on Python and Streamlit, with deployment via Vercel for instant cloud availability. No complex infrastructure. No vendor lock-in.

The data pipeline is idempotent and transactional. You can run data updates multiple times safely. Backups are automatic. Rollback is straightforward.

For institutions with existing data warehouses, we provide schema documentation and migration scripts. The Wave ID registry is extensible—add new Waves, define custom benchmarks, configure attribution rules.

Our implementation team works with your data governance and compliance teams to ensure WAVES Intelligence meets your specific institutional requirements.""",

    "slide_13_closing.txt": """Thank you for your time today.

To summarize: WAVES Intelligence delivers governance-grade portfolio attribution, real-time risk monitoring, and board-ready reporting—all built on a transparent, deterministic architecture that respects institutional oversight requirements.

We don't predict. We explain. We don't optimize in black boxes. We surface truth outputs with reliability signals.

If your institution is ready to move beyond traditional attribution and embrace transparent, governance-native analytics, we're ready to help.

Next steps are simple. Schedule a technical deep dive with our implementation team. Review our data integration requirements. And let's discuss how WAVES Intelligence can support your specific investment oversight needs.

I'm Vector. Thank you for listening. And welcome to the future of institutional portfolio intelligence.""",
}


def main():
    """Generate all narration files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    narration_dir = os.path.join(script_dir, "narration")
    
    # Ensure narration directory exists
    os.makedirs(narration_dir, exist_ok=True)
    
    print("Generating TTS-ready narration files...")
    print(f"Output directory: {narration_dir}")
    print()
    
    for filename, content in NARRATIONS.items():
        filepath = os.path.join(narration_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created: {filename}")
    
    print()
    print(f"✓ Generated {len(NARRATIONS)} narration files")
    print()
    print("Files are compatible with TTS tools:")
    print("  • Amazon Polly")
    print("  • Google Cloud Text-to-Speech")
    print("  • Microsoft Azure Speech")
    print("  • ElevenLabs")
    print()
    print("Recommended voice settings:")
    print("  • Tone: Calm, warm, explanatory")
    print("  • Speed: Moderate (95-100%)")
    print("  • Pitch: Neutral to slightly lower")


if __name__ == "__main__":
    main()
