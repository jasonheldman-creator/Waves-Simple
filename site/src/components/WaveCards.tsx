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
  // Generate placeholder waves if not provided
  const defaultWaves: WaveCard[] = Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: `Wave ${i + 1}`,
    description: `Strategic investment wave focused on specific market opportunities and alpha generation.`,
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
