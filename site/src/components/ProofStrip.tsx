interface ProofPoint {
  metric: string;
  value: string;
  description: string;
}

interface ProofStripProps {
  title: string;
  subtitle: string;
  points: ProofPoint[];
}

export default function ProofStrip({ title, subtitle, points }: ProofStripProps) {
  return (
    <section className="bg-black py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">{title}</h2>
          <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">{subtitle}</p>
        </div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {points.map((point, index) => (
            <div
              key={index}
              className="group rounded-lg border border-gray-800 bg-gradient-to-br from-gray-900 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
            >
              <div className="mb-3">
                <div className="text-sm font-medium text-cyan-400 mb-1">{point.metric}</div>
                <div className="text-2xl font-bold text-white group-hover:text-cyan-300">
                  {point.value}
                </div>
              </div>
              <p className="text-sm text-gray-400 leading-relaxed">{point.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
