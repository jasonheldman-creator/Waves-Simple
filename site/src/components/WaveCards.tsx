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

export default function WaveCards({ waves, count = 15 }: WaveCardsProps) {
  // Generate institutional-grade waves with unique strategies if not provided
  const defaultWaves: WaveCard[] = [
    {
      id: 1,
      name: "Wave 1",
      description: "Large-cap equity core portfolio emphasizing quality factors, sustainable competitive advantages, and operational excellence across developed markets.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 2,
      name: "Wave 2",
      description: "Growth-oriented strategy targeting high-momentum securities with superior earnings trajectory and market leadership in expanding sectors.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 3,
      name: "Wave 3",
      description: "Systematic value approach identifying mispriced securities through fundamental analysis, contrarian positioning, and mean-reversion dynamics.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 4,
      name: "Wave 4",
      description: "Dividend-focused mandate delivering consistent income generation through blue-chip equities with proven payout histories and cash flow stability.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 5,
      name: "Wave 5",
      description: "Multi-factor quantitative strategy combining value, momentum, quality, and low-volatility signals for risk-adjusted alpha generation.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 6,
      name: "Wave 6",
      description: "Sector rotation framework dynamically allocating capital across cyclical and defensive industries based on macroeconomic regime analysis.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 7,
      name: "Wave 7",
      description: "International equity exposure capturing diversification benefits and growth opportunities in emerging and frontier markets.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 8,
      name: "Wave 8",
      description: "Small and mid-cap focus exploiting market inefficiencies and structural advantages in less-researched segments of public equities.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 9,
      name: "Wave 9",
      description: "Technology and innovation portfolio accessing secular growth themes in digital transformation, automation, and next-generation infrastructure.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 10,
      name: "Wave 10",
      description: "Defensive positioning strategy utilizing low-beta securities, consumer staples, and utilities to preserve capital during market volatility.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 11,
      name: "Wave 11",
      description: "ESG-integrated approach aligning investment decisions with environmental, social, and governance criteria while maintaining competitive returns.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 12,
      name: "Wave 12",
      description: "Volatility-harvesting strategy exploiting options markets and derivative instruments to generate alpha from regime changes and dispersion.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 13,
      name: "Wave 13",
      description: "Macro-driven allocation adjusting equity beta, geographic exposure, and currency positioning based on global economic indicators.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 14,
      name: "Wave 14",
      description: "Long-short equity mandate generating market-neutral returns through paired positions in overvalued and undervalued securities.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 15,
      name: "Wave 15",
      description: "Opportunistic special situations strategy targeting corporate events, restructurings, and catalyst-driven mispricings for asymmetric returns.",
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
