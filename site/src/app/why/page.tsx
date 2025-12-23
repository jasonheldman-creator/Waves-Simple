import Hero from "@/components/Hero";
import { siteContent } from "@/content/newSiteContent";

export default function WhyPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.why.hero.title}
        subtitle={siteContent.why.hero.subtitle}
      />
      
      {/* Main Reasons Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
            {siteContent.why.reasons.map((reason, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="mb-4 text-5xl">{reason.icon}</div>
                <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3">
                  {reason.title}
                </h3>
                <p className="text-gray-400 leading-relaxed">{reason.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Differentiators Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.why.differentiators.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.why.differentiators.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            {siteContent.why.differentiators.points.map((point, index) => (
              <div
                key={index}
                className="rounded-lg border border-gray-800 bg-gradient-to-br from-gray-900 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <h3 className="text-xl font-semibold text-white mb-3">
                  {point.title}
                </h3>
                <p className="text-gray-400 leading-relaxed">{point.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
