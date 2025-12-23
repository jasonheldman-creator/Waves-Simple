interface WaveCard {
  id: number;
  name: string;
  description: string;
  performance?: string;
  status?: string;
}

interface WaveCardsProps {
  waves?: WaveCard[];
  count?: number;
}

export default function WaveCards({ waves, count: _count = 15 }: WaveCardsProps) {
  // Generate institutional-grade waves with unique strategies if not provided
  const defaultWaves: WaveCard[] = [
    {
      id: 1,
      name: "Wave 1",
      description: "Large-cap equity core portfolio with benchmark integrity controls. Analytics: Quality factor scoring, competitive moat analysis. Role: Portfolio foundation. Users: Institutional asset allocators, pension funds.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 2,
      name: "Wave 2",
      description: "Growth-oriented strategy with momentum risk overlays. Analytics: Earnings trajectory modeling, market leadership scoring. Role: Growth alpha generation. Users: Growth-focused managers, endowments.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 3,
      name: "Wave 3",
      description: "Systematic value with audit trail validation. Analytics: Fundamental mispricing detection, mean-reversion signals. Role: Value factor exposure. Users: Value investors, contrarian allocators.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 4,
      name: "Wave 4",
      description: "Dividend mandate with benchmark integrity on payout ratios. Analytics: Cash flow stability metrics, payout sustainability. Role: Income generation. Users: Income-focused institutions, retirees.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 5,
      name: "Wave 5",
      description: "Multi-factor quantitative with comprehensive risk overlays. Analytics: Value-momentum-quality-volatility scoring. Role: Diversified factor exposure. Users: Quantitative managers, factor allocators.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 6,
      name: "Wave 6",
      description: "Sector rotation with macro regime risk overlays. Analytics: Economic cycle positioning, sector correlation analysis. Role: Tactical allocation. Users: Macro strategists, tactical allocators.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 7,
      name: "Wave 7",
      description: "International equity with currency risk overlays and audit trails. Analytics: Geographic diversification metrics, emerging market exposure. Role: Global diversification. Users: Global allocators, international funds.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 8,
      name: "Wave 8",
      description: "Small-mid cap with benchmark integrity on liquidity constraints. Analytics: Market inefficiency scoring, illiquidity premium capture. Role: Size factor exposure. Users: Small-cap specialists, growth seekers.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 9,
      name: "Wave 9",
      description: "Technology innovation with secular growth risk overlays. Analytics: Digital transformation scoring, innovation pipeline assessment. Role: Thematic growth. Users: Tech-focused investors, growth allocators.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 10,
      name: "Wave 10",
      description: "Defensive positioning with downside risk overlays and audit trails. Analytics: Low-beta validation, volatility regime detection. Role: Capital preservation. Users: Risk-averse institutions, defensive allocators.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 11,
      name: "Wave 11",
      description: "ESG-integrated with governance benchmark integrity controls. Analytics: ESG factor scoring, impact measurement. Role: Sustainable investing. Users: ESG-focused institutions, impact investors.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 12,
      name: "Wave 12",
      description: "Volatility-harvesting with options risk overlays and complete audit trails. Analytics: Volatility regime modeling, dispersion capture. Role: Alternative alpha. Users: Derivatives specialists, volatility traders.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 13,
      name: "Wave 13",
      description: "Macro-driven allocation with multi-asset risk overlays. Analytics: Global macro indicators, currency positioning analytics. Role: Dynamic beta adjustment. Users: Macro allocators, global asset managers.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 14,
      name: "Wave 14",
      description: "Long-short equity with market-neutral risk overlays and benchmark integrity. Analytics: Paired position analysis, factor neutrality validation. Role: Market-neutral alpha. Users: Hedge funds, absolute return seekers.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 15,
      name: "Wave 15",
      description: "Special situations with event-driven risk overlays and governance audit trails. Analytics: Corporate action modeling, catalyst probability scoring. Role: Opportunistic alpha. Users: Event-driven managers, special situations investors.",
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
