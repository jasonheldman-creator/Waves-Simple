import type { Metadata } from "next";
import Hero from "@/components/Hero";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Governance & Oversight - WAVES Intelligence",
  description: "Institutional-grade governance frameworks for fiduciary oversight, regulatory compliance, and operational transparency.",
};

export default function GovernancePage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.governance.hero.title}
        subtitle={siteContent.governance.hero.subtitle}
        ctaText="Request Institutional Demo"
        ctaLink="/demo"
        secondaryCtaText="View Governance Architecture"
        secondaryCtaLink="/architecture"
      />
      
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-lg text-gray-300">
            {siteContent.governance.description}
          </p>
        </div>
      </section>

      {/* Governance Pillars Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.governance.pillars.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.governance.pillars.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            {siteContent.governance.pillars.items.map((pillar, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="mb-4 text-5xl">{pillar.icon}</div>
                <h3 className="text-2xl font-semibold text-white group-hover:text-cyan-400 mb-4">
                  {pillar.title}
                </h3>
                <p className="text-gray-400 leading-relaxed mb-6">{pillar.description}</p>
                <div className="space-y-3 pt-4 border-t border-gray-700">
                  {pillar.capabilities.map((capability, capIndex) => (
                    <div key={capIndex} className="flex items-start text-sm text-gray-500">
                      <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                      <span>{capability}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Regulatory Compliance Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.governance.compliance.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.governance.compliance.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
            {siteContent.governance.compliance.frameworks.map((framework, index) => (
              <div
                key={index}
                className="group rounded-lg border border-gray-800 bg-gray-900/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="text-4xl mb-4">{framework.icon}</div>
                <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400 mb-3">
                  {framework.title}
                </h3>
                <p className="text-sm text-gray-400 leading-relaxed">{framework.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Governance Controls Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.governance.controls.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.governance.controls.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            {siteContent.governance.controls.features.map((feature, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="mb-4 text-4xl">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400 mb-3">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-400 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

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
