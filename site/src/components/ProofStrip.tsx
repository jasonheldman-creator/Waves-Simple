interface ProofTile {
  icon: string;
  title: string;
  description: string;
}

interface ProofStripProps {
  tiles: ProofTile[];
}

export default function ProofStrip({ tiles }: ProofStripProps) {
  return (
    <section className="bg-black py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {tiles.map((tile, index) => (
            <div
              key={index}
              className="group rounded-lg border border-gray-800 bg-gradient-to-br from-gray-900 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
            >
              <div className="mb-4 text-4xl">{tile.icon}</div>
              <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400 mb-2">
                {tile.title}
              </h3>
              <p className="text-sm text-gray-400 leading-relaxed">{tile.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
