interface WaveCard {
  id: number;
  name: string;
  description: string;
  performance?: string;
  status?: string;
}

interface WaveCardsProps {
  waves?: WaveCard[];
}

export default function WaveCards({ waves }: WaveCardsProps) {
  // Generate institutional-grade waves with unique strategies if not provided
  const defaultWaves: WaveCard[] = [
    {
      id: 1,
      name: "Wave 1 - Core Equity",
      description: "Large-cap quality core portfolio. Capital Role: Portfolio foundation (40-60% allocation). Risk Regime: All-weather positioning. Allocation Purpose: Beta exposure with quality overlay. Portfolio Sleeve: Core/Satellite foundation.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 2,
      name: "Wave 2 - Growth Alpha",
      description: "Momentum-driven growth strategy. Capital Role: Opportunistic allocation (10-20%). Risk Regime: Risk-on positioning. Allocation Purpose: Alpha capture in expansion cycles. Portfolio Sleeve: Satellite growth engine.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 3,
      name: "Wave 3 - Value Recovery",
      description: "Systematic value with mean reversion. Capital Role: Contrarian allocation (10-15%). Risk Regime: Risk-off/recovery phases. Allocation Purpose: Capture mispricing and market dislocations. Portfolio Sleeve: Defensive/Opportunistic blend.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 4,
      name: "Wave 4 - Income Generation",
      description: "Dividend sustainability mandate. Capital Role: Stable income sleeve (15-25%). Risk Regime: Low volatility preference. Allocation Purpose: Cash flow generation and capital preservation. Portfolio Sleeve: Defensive core.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 5,
      name: "Wave 5 - Multi-Factor",
      description: "Value-momentum-quality-volatility blend. Capital Role: Diversified factor exposure (20-30%). Risk Regime: Regime-adaptive positioning. Allocation Purpose: Balanced factor returns across cycles. Portfolio Sleeve: Core diversification.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 6,
      name: "Wave 6 - Sector Rotation",
      description: "Macro-driven sector allocation. Capital Role: Tactical positioning (5-15%). Risk Regime: Cycle-dependent rotation. Allocation Purpose: Economic cycle alpha capture. Portfolio Sleeve: Tactical overlay.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 7,
      name: "Wave 7 - Global Diversification",
      description: "International equity with currency management. Capital Role: Geographic diversification (15-25%). Risk Regime: Global macro-aware. Allocation Purpose: Non-US growth and diversification. Portfolio Sleeve: Geographic expansion.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 8,
      name: "Wave 8 - Small-Mid Cap",
      description: "Size factor with liquidity constraints. Capital Role: Inefficiency capture (10-20%). Risk Regime: Risk-on with higher volatility. Allocation Purpose: Small-cap premium and market inefficiency. Portfolio Sleeve: Opportunistic alpha.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 9,
      name: "Wave 9 - Innovation Thematic",
      description: "Secular technology and innovation trends. Capital Role: Thematic growth (5-15%). Risk Regime: High conviction, growth-oriented. Allocation Purpose: Long-term secular alpha capture. Portfolio Sleeve: Thematic satellite.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 10,
      name: "Wave 10 - Defensive Positioning",
      description: "Low-beta capital preservation. Capital Role: Risk mitigation (10-20%). Risk Regime: Risk-off and volatility spikes. Allocation Purpose: Downside protection and stability. Portfolio Sleeve: Defensive anchor.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 11,
      name: "Wave 11 - ESG Integration",
      description: "Sustainable investing with ESG factors. Capital Role: Values-aligned allocation (10-30%). Risk Regime: Long-term orientation. Allocation Purpose: ESG alpha and impact alignment. Portfolio Sleeve: Core with values overlay.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 12,
      name: "Wave 12 - Volatility Harvesting",
      description: "Options-based volatility strategies. Capital Role: Alternative return stream (5-10%). Risk Regime: Volatility regime-dependent. Allocation Purpose: Uncorrelated alpha and volatility capture. Portfolio Sleeve: Alternative diversifier.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 13,
      name: "Wave 13 - Macro Allocation",
      description: "Multi-asset global macro positioning. Capital Role: Dynamic beta adjustment (10-20%). Risk Regime: All-weather with tactical shifts. Allocation Purpose: Top-down allocation and currency exposure. Portfolio Sleeve: Strategic macro overlay.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 14,
      name: "Wave 14 - Market Neutral",
      description: "Long-short equity with factor neutrality. Capital Role: Absolute return sleeve (5-15%). Risk Regime: Market-neutral positioning. Allocation Purpose: Pure alpha extraction independent of beta. Portfolio Sleeve: Alternative alpha.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 15,
      name: "Wave 15 - Special Situations",
      description: "Event-driven catalyst strategies. Capital Role: Opportunistic events (5-10%). Risk Regime: Event-specific timing. Allocation Purpose: M&A, restructuring, and corporate action alpha. Portfolio Sleeve: Tactical opportunistic.",
      performance: "+0.00%",
      status: "Active",
    },
  ];

  const displayWaves = waves || defaultWaves;

  return (
    <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">
            Investment <span className="text-cyan-400">Waves</span>
          </h2>
          <p className="mt-4 text-lg text-gray-400">
            Explore our portfolio of strategic investment waves
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {displayWaves.map((wave) => (
            <div
              key={wave.id}
              className="group rounded-lg border border-gray-800 bg-gray-900/30 p-6 backdrop-blur-sm transition-all hover:border-cyan-500/50 hover:bg-gray-900/50"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400">
                    {wave.name}
                  </h3>
                  <span className="mt-1 inline-block rounded-full bg-green-500/20 px-2 py-1 text-xs font-medium text-green-400">
                    {wave.status}
                  </span>
                </div>
                {wave.performance && (
                  <div className="text-right">
                    <div className="text-lg font-bold text-green-400">{wave.performance}</div>
                    <div className="text-xs text-gray-500">Performance</div>
                  </div>
                )}
              </div>
              <p className="mt-4 text-sm text-gray-400">{wave.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
