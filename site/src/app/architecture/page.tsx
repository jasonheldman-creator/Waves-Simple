import type { Metadata } from "next";
import Hero from "@/components/Hero";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Architecture - WAVES Intelligence",
  description: "Enterprise-grade infrastructure designed for institutional performance, security, and scalability.",
};

export default function ArchitecturePage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.architecture.hero.title}
        subtitle={siteContent.architecture.hero.subtitle}
        ctaText="View Security"
        ctaLink="/security"
        secondaryCtaText="Contact Us"
        secondaryCtaLink="/contact"
      />
      
      {/* Architecture as Trust Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.architecture.architectureAsTrust.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.architecture.architectureAsTrust.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
            {siteContent.architecture.architectureAsTrust.principles.map((principle, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="mb-4 text-5xl">{principle.icon}</div>
                <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 mb-3">
                  {principle.title}
                </h3>
                <p className="text-gray-400 leading-relaxed mb-4">{principle.description}</p>
                <div className="space-y-2 pt-4 border-t border-gray-700">
                  {principle.details.map((detail, detailIndex) => (
                    <div key={detailIndex} className="flex items-start text-sm text-gray-500">
                      <span className="text-cyan-400 mr-2">✓</span>
                      <span>{detail}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-lg text-gray-300">
            {siteContent.architecture.description}
          </p>
        </div>
      </section>
      
      {/* How WAVES Is Different Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.architecture.differentiation.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.architecture.differentiation.subtitle}
            </p>
            <p className="mt-6 text-base text-gray-300 max-w-4xl mx-auto leading-relaxed">
              {siteContent.architecture.differentiation.content}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-12">
            {siteContent.architecture.differentiation.features.map((feature, index) => (
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

      <ArchitectureDiagram />
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="text-3xl mb-3">☁️</div>
              <h3 className="text-xl font-semibold text-white">Cloud-Native</h3>
              <p className="mt-2 text-gray-400">
                Built on modern cloud infrastructure with auto-scaling and global distribution.
              </p>
            </div>
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="text-3xl mb-3">⚡</div>
              <h3 className="text-xl font-semibold text-white">High Performance</h3>
              <p className="mt-2 text-gray-400">
                Optimized data processing pipelines delivering real-time analytics at scale.
              </p>
            </div>
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="text-3xl mb-3">🔄</div>
              <h3 className="text-xl font-semibold text-white">Resilient</h3>
              <p className="mt-2 text-gray-400">
                Multiple layers of redundancy ensuring 99.9% uptime and data integrity.
              </p>
            </div>
          </div>
        </div>
      </section>
      <CallToAction
        title={siteContent.cta.architecture.title}
        description={siteContent.cta.architecture.description}
        primaryButtonText={siteContent.cta.architecture.primaryButtonText}
        primaryButtonLink={siteContent.cta.architecture.primaryButtonLink}
        secondaryButtonText={siteContent.cta.architecture.secondaryButtonText}
        secondaryButtonLink={siteContent.cta.architecture.secondaryButtonLink}
      />
    </main>
  );
}
