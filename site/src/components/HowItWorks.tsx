interface Step {
  number: number;
  title: string;
  description: string;
  icon: string;
}

interface HowItWorksProps {
  steps: Step[];
  title?: string;
  subtitle?: string;
}

export default function HowItWorks({ steps, title, subtitle }: HowItWorksProps) {
  return (
    <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {(title || subtitle) && (
          <div className="text-center mb-16">
            {title && (
              <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">{title}</h2>
            )}
            {subtitle && <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">{subtitle}</p>}
          </div>
        )}
        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          {steps.map((step, index) => (
            <div
              key={index}
              className="group relative rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
            >
              {/* Step number */}
              <div className="absolute -top-4 -left-4 flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-r from-cyan-500 to-green-500 text-xl font-bold text-black">
                {step.number}
              </div>

              {/* Icon */}
              <div className="mb-4 text-5xl">{step.icon}</div>

              {/* Title */}
              <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3">
                {step.title}
              </h3>

              {/* Description */}
              <p className="text-gray-400 leading-relaxed">{step.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
