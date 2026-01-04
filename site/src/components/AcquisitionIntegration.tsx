interface Capability {
  title: string;
  description: string;
  icon: string;
}

interface AcquisitionIntegrationProps {
  title: string;
  subtitle: string;
  intro: string;
  capabilities: Capability[];
}

export default function AcquisitionIntegration({
  title,
  subtitle,
  intro,
  capabilities,
}: AcquisitionIntegrationProps) {
  return (
    <section className="bg-black py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">{title}</h2>
          <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">{subtitle}</p>
          <div className="mt-8 max-w-4xl mx-auto">
            <p className="text-base text-gray-300 leading-relaxed border-l-4 border-cyan-500 pl-6 py-2 bg-gray-900/30 rounded-r">
              {intro}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
          {capabilities.map((capability, index) => (
            <div
              key={index}
              className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
            >
              <div className="mb-4 text-5xl">{capability.icon}</div>
              <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3 transition-colors">
                {capability.title}
              </h3>
              <p className="text-gray-400 leading-relaxed">{capability.description}</p>
            </div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="inline-flex items-center px-6 py-3 border border-cyan-500/30 rounded-lg bg-cyan-500/5">
            <span className="text-cyan-400 text-sm font-medium">
              Interested in acquisition, partnership, or licensing discussions?
            </span>
            <a
              href="/contact"
              className="ml-4 text-cyan-400 hover:text-cyan-300 font-semibold transition-colors"
            >
              Contact Us →
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
