import Hero from "@/components/Hero";
import FeatureGrid from "@/components/FeatureGrid";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export default function Home() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.home.hero.title}
        subtitle={siteContent.home.hero.subtitle}
        ctaText={siteContent.home.hero.ctaText}
        ctaLink={siteContent.home.hero.ctaLink}
        secondaryCtaText={siteContent.home.hero.secondaryCtaText}
        secondaryCtaLink={siteContent.home.hero.secondaryCtaLink}
      />
      
      {/* Operational Proof Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.home.operationalProof.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.home.operationalProof.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
            {siteContent.home.operationalProof.examples.map((example, index) => (
              <div
                key={index}
                className="group rounded-lg border border-gray-800 bg-gradient-to-br from-gray-900 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="text-5xl">{example.icon}</div>
                  <div className="text-right">
                    <div className="text-xs text-gray-500 uppercase tracking-wide">{example.metric}</div>
                    <div className="text-sm font-semibold text-cyan-400">{example.value}</div>
                  </div>
                </div>
                <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3">
                  {example.title}
                </h3>
                <p className="text-gray-400 leading-relaxed">{example.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Buyer Personas Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.home.buyerPersonas.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.home.buyerPersonas.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
            {siteContent.home.buyerPersonas.roles.map((role, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="mb-4 text-5xl">{role.icon}</div>
                <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400 mb-3">
                  {role.title}
                </h3>
                <p className="text-sm text-gray-400 leading-relaxed mb-4">{role.description}</p>
                <div className="space-y-1">
                  {role.needs.map((need, needIndex) => (
                    <div key={needIndex} className="flex items-start text-xs text-gray-500">
                      <span className="text-cyan-400 mr-2">•</span>
                      <span>{need}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Institutional Trust Signals Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.home.institutionalTrust.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.home.institutionalTrust.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            {siteContent.home.institutionalTrust.features.map((feature, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="mb-4 text-5xl">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <FeatureGrid features={siteContent.home.features} />
      <CallToAction
        title={siteContent.cta.default.title}
        description={siteContent.cta.default.description}
        primaryButtonText={siteContent.cta.default.primaryButtonText}
        primaryButtonLink={siteContent.cta.default.primaryButtonLink}
        secondaryButtonText={siteContent.cta.default.secondaryButtonText}
        secondaryButtonLink={siteContent.cta.default.secondaryButtonLink}
      />
    </main>
  );
}
