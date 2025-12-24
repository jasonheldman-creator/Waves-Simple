import type { Metadata } from "next";
import Hero from "@/components/Hero";
import FeatureGrid from "@/components/FeatureGrid";
import ScreenshotGallery from "@/components/ScreenshotGallery";
import CallToAction from "@/components/CallToAction";
import RegimeCard from "@/components/RegimeCard";
import ExecutiveBriefing from "@/components/ExecutiveBriefing";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Platform - WAVES Intelligence",
  description: "A comprehensive suite of tools designed for institutional portfolio management and analytics.",
};

export default function PlatformPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.platform.hero.title}
        subtitle={siteContent.platform.hero.subtitle}
        ctaText="Request Institutional Demo"
        ctaLink="/demo"
        secondaryCtaText="Discuss Platform Licensing"
        secondaryCtaLink="/contact"
      />
      
      {/* Institutional Buyer Validation Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          {/* Executive Briefing Reference */}
          <div className="mb-16">
            <ExecutiveBriefing variant="secondary" showDownload={true} />
          </div>

          {/* Today's Regime Card */}
          <div className="mb-16">
            <RegimeCard />
          </div>

          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Institutional Validation Frameworks
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Purpose-built capabilities addressing institutional buyer needs across the decision-making organization
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">👔</div>
              <h3 className="text-2xl font-semibold text-white mb-4">For CIOs & Investment Committees</h3>
              <p className="text-gray-400 mb-6">
                Comprehensive oversight infrastructure supporting strategic allocation decisions, risk governance, and fiduciary responsibilities.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Board-ready reporting with complete attribution transparency</span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Risk budget monitoring and limit breach alerting</span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Governance workflow approval and delegation controls</span>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">🛡️</div>
              <h3 className="text-2xl font-semibold text-white mb-4">For Risk & Compliance Officers</h3>
              <p className="text-gray-400 mb-6">
                Continuous monitoring, regulatory reporting, and comprehensive audit documentation meeting institutional compliance standards.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Real-time exposure monitoring with multi-layer risk overlays</span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Immutable audit trails for regulatory examination</span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Policy enforcement and compliance validation gates</span>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">📊</div>
              <h3 className="text-2xl font-semibold text-white mb-4">For Portfolio Managers</h3>
              <p className="text-gray-400 mb-6">
                Tactical execution support with real-time analytics, benchmark validation, and position-level decision intelligence.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Multi-factor attribution with benchmark integrity controls</span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Position-level exposure analysis and risk decomposition</span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Decision support with complete audit trail documentation</span>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">🏢</div>
              <h3 className="text-2xl font-semibold text-white mb-4">For Corporate Development & Acquirers</h3>
              <p className="text-gray-400 mb-6">
                Integration-ready infrastructure with technical transparency supporting M&A evaluation and enterprise deployment.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Complete architecture documentation and data lineage clarity</span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">API access and integration pathway specifications</span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">Scalability validation and performance benchmarks</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* How WAVES Is Different Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.platform.differentiation.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.platform.differentiation.subtitle}
            </p>
            <p className="mt-6 text-base text-gray-300 max-w-4xl mx-auto leading-relaxed">
              {siteContent.platform.differentiation.content}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-12">
            {siteContent.platform.differentiation.features.map((feature, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="mb-3 text-4xl">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400 mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-400 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Platform Preview Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.platform.platformPreview.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.platform.platformPreview.subtitle}
            </p>
            <p className="mt-6 text-base text-gray-300 max-w-4xl mx-auto leading-relaxed">
              {siteContent.platform.platformPreview.description}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            {siteContent.platform.platformPreview.capabilities.map((capability, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="text-5xl">{capability.icon}</div>
                </div>
                <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3">
                  {capability.title}
                </h3>
                <p className="text-gray-400 leading-relaxed mb-6">{capability.description}</p>
                <div className="space-y-2 pt-4 border-t border-gray-700">
                  {capability.features.map((feature, featureIndex) => (
                    <div key={featureIndex} className="flex items-start text-sm text-gray-500">
                      <span className="text-cyan-400 mr-2">✓</span>
                      <span>{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <FeatureGrid features={siteContent.platform.features} columns={2} />
      <ScreenshotGallery />
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
