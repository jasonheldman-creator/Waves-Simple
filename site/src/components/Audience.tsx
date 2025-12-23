interface AudienceColumn {
  title: string;
  description: string;
  benefits: string[];
  icon: string;
}

interface AudienceProps {
  columns: AudienceColumn[];
  title?: string;
  subtitle?: string;
}

export default function Audience({ columns, title, subtitle }: AudienceProps) {
  return (
    <section className="bg-black py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {(title || subtitle) && (
          <div className="text-center mb-16">
            {title && (
              <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
                {subtitle}
              </p>
            )}
          </div>
        )}
        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          {columns.map((column, index) => (
            <div
              key={index}
              className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
            >
              {/* Icon */}
              <div className="mb-4 text-5xl">{column.icon}</div>
              
              {/* Title */}
              <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3">
                {column.title}
              </h3>
              
              {/* Description */}
              <p className="text-gray-400 leading-relaxed mb-6">{column.description}</p>
              
              {/* Benefits */}
              <div className="space-y-2">
                {column.benefits.map((benefit, benefitIndex) => (
                  <div key={benefitIndex} className="flex items-start text-sm text-gray-300">
                    <span className="text-cyan-400 mr-2 mt-1">✓</span>
                    <span>{benefit}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
