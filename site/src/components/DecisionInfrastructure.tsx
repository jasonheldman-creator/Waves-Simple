interface DecisionInfrastructureProps {
  title: string;
  subtitle: string;
  processes: {
    title: string;
    description: string;
    icon: string;
  }[];
  concepts?: {
    title: string;
    description: string;
  }[];
}

export default function DecisionInfrastructure({
  title,
  subtitle,
  processes,
  concepts,
}: DecisionInfrastructureProps) {
  return (
    <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">{title}</h2>
          <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">{subtitle}</p>
        </div>

        {/* Decision Processes */}
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4 mb-16">
          {processes.map((process, index) => (
            <div
              key={index}
              className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
            >
              <div className="mb-4 text-5xl">{process.icon}</div>
              <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3">
                {process.title}
              </h3>
              <p className="text-sm text-gray-400 leading-relaxed">{process.description}</p>
            </div>
          ))}
        </div>

        {/* Core Concepts */}
        {concepts && concepts.length > 0 && (
          <div className="mt-20">
            <div className="text-center mb-12">
              <h3 className="text-2xl font-bold text-white sm:text-3xl">
                Core Infrastructure Principles
              </h3>
              <p className="mt-4 text-base text-gray-400 max-w-2xl mx-auto">
                Aladdin-class infrastructure built for institutional decision-making at scale
              </p>
            </div>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
              {concepts.map((concept, index) => (
                <div
                  key={index}
                  className="rounded-lg border border-gray-800 bg-gray-900/50 p-6 transition-all hover:border-cyan-500/30"
                >
                  <h4 className="text-lg font-semibold text-cyan-400 mb-3">{concept.title}</h4>
                  <p className="text-sm text-gray-400 leading-relaxed">{concept.description}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
