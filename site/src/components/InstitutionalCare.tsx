interface OppositionPoint {
  oppose: string;
  position: string;
  description: string;
}

interface InstitutionalCareProps {
  title: string;
  subtitle: string;
  points: OppositionPoint[];
}

export default function InstitutionalCare({ title, subtitle, points }: InstitutionalCareProps) {
  return (
    <section className="bg-black py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
            {title}
          </h2>
          <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
            {subtitle}
          </p>
        </div>
        
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
          {points.map((point, index) => (
            <div
              key={index}
              className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
            >
              <div className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm text-red-400 font-medium">Oppose:</span>
                  <span className="text-xs text-gray-500">vs</span>
                  <span className="text-sm text-cyan-400 font-medium">Position:</span>
                </div>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-base text-red-300/80 line-through">{point.oppose}</span>
                  <span className="text-2xl text-gray-600">→</span>
                  <span className="text-base text-cyan-300 font-semibold">{point.position}</span>
                </div>
              </div>
              <p className="text-gray-400 leading-relaxed text-sm">{point.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
