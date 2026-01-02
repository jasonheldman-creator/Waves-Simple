interface Distinction {
  title: string;
  description: string;
}

interface WavesDistinctionProps {
  title: string;
  subtitle: string;
  intro: string;
  distinctions: Distinction[];
}

export default function WavesDistinction({
  title,
  subtitle,
  intro,
  distinctions,
}: WavesDistinctionProps) {
  return (
    <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
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

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
          {distinctions.map((distinction, index) => (
            <div
              key={index}
              className="rounded-lg border border-gray-800 bg-gradient-to-br from-gray-900 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
            >
              <div className="flex items-start">
                <div className="flex-shrink-0 flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/10 text-cyan-400 mr-4">
                  <span className="text-xl font-bold">{index + 1}</span>
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-white mb-3">{distinction.title}</h3>
                  <p className="text-gray-400 leading-relaxed">{distinction.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
