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

// Professional wave descriptions with diverse investment strategies
const WAVE_DESCRIPTIONS = [
  "Large-cap growth equities with emphasis on quality fundamentals, sustainable competitive advantages, and strong cash flow generation.",
  "Technology sector allocation targeting secular growth trends in cloud computing, artificial intelligence, and digital transformation.",
  "Global diversification strategy spanning developed and emerging markets with dynamic regional allocation based on regime analysis.",
  "Value-oriented portfolio focusing on undervalued securities with strong balance sheets and margin of safety principles.",
  "Dividend growth strategy emphasizing companies with consistent payout increases and sustainable cash flow dynamics.",
  "Small and mid-cap opportunities leveraging inefficiencies in less-covered market segments with rigorous fundamental analysis.",
  "Defensive positioning in consumer staples, utilities, and healthcare sectors designed for capital preservation during volatility.",
  "Fixed income allocation across government and investment-grade corporate bonds with duration management and credit analysis.",
  "Alternative strategies including real assets, commodities, and absolute return approaches for portfolio diversification.",
  "Emerging market equities capturing growth in developing economies with focus on demographic trends and structural reforms.",
  "Sector rotation framework dynamically allocating capital based on economic cycle positioning and relative strength indicators.",
  "ESG-integrated investment approach combining financial analysis with environmental, social, and governance criteria.",
  "Quantitative strategies employing systematic factor models, momentum signals, and risk parity frameworks.",
  "Volatility management through options strategies, tail risk hedging, and tactical de-risking during elevated VIX regimes.",
  "Opportunistic allocation for special situations including mergers, restructurings, and event-driven catalysts with asymmetric return profiles.",
];

export default function WaveCards({ waves, count = 15 }: WaveCardsProps) {
  const defaultWaves: WaveCard[] = Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: `Wave ${i + 1}`,
    description: WAVE_DESCRIPTIONS[i % WAVE_DESCRIPTIONS.length],
    performance: "+0.00%",
    status: "Active",
  }));

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
