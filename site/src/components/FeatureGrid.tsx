interface Feature {
  title: string;
  description: string;
  icon?: string;
}

interface FeatureGridProps {
  features: Feature[];
  columns?: 2 | 3 | 4;
}

export default function FeatureGrid({ features, columns = 3 }: FeatureGridProps) {
  const gridCols = {
    2: "md:grid-cols-2",
    3: "md:grid-cols-3",
    4: "md:grid-cols-4",
  };

  return (
    <section className="bg-black py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className={`grid grid-cols-1 gap-8 ${gridCols[columns]}`}>
          {features.map((feature, index) => (
            <div
              key={index}
              className="group rounded-lg border border-gray-800 bg-gray-900/50 p-6 transition-all hover:border-cyan-500/50 hover:bg-gray-900"
            >
              {feature.icon && <div className="mb-4 text-4xl">{feature.icon}</div>}
              <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400">
                {feature.title}
              </h3>
              <p className="mt-2 text-gray-400">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
