import Link from "next/link";
import Hero from "@/components/Hero";

export default function Home() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title="Decision Infrastructure for Modern Asset Management"
        subtitle="WAVES Intelligence™ is decision infrastructure for institutional portfolio management—delivering transparency, attribution integrity, and governance-first operations. WAVES™ portfolios demonstrate the platform through disciplined execution under real market conditions."
        ctaText="Request Institutional Demo"
        ctaLink="/demo"
        secondaryCtaText="View Platform"
        secondaryCtaLink="/platform"
      />
      
      {/* Value Proposition Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Executive Summary
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Decision infrastructure and portfolio proof unified for institutional governance
            </p>
          </div>
          
          {/* Platform Overview */}
          <div className="mb-16 rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-10">
            <div className="flex items-start mb-4">
              <div className="text-5xl mr-6">🏗️</div>
              <div>
                <h3 className="text-2xl font-semibold text-white mb-4">
                  WAVES Intelligence™ Platform
                </h3>
                <p className="text-lg text-gray-300 leading-relaxed">
                  Decision infrastructure for institutional portfolio management—not a product, but foundational 
                  architecture. The platform delivers multi-dimensional attribution, risk exposure control, and 
                  governance-native workflows. From data ingestion through decision execution to board-ready 
                  documentation, WAVES Intelligence™ provides the canonical source of truth for institutional 
                  asset management. Built for CIOs, risk officers, and fiduciaries requiring Aladdin-class 
                  transparency and control.
                </p>
              </div>
            </div>
          </div>

          {/* WAVES™ Portfolios Overview */}
          <div className="mb-16 rounded-lg border border-green-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-10">
            <div className="flex items-start mb-4">
              <div className="text-5xl mr-6">📈</div>
              <div>
                <h3 className="text-2xl font-semibold text-white mb-4">
                  WAVES™ Portfolios
                </h3>
                <p className="text-lg text-gray-300 leading-relaxed">
                  Live portfolios demonstrating platform effectiveness under real market conditions. Each WAVES™ 
                  portfolio operates as institutional proof—showcasing risk-aware construction, regime-adaptive 
                  positioning, and complete attribution transparency. Not speculative products, but disciplined 
                  validation of the underlying infrastructure. Alpha capture through systematic selection, overlay 
                  management, and regime intelligence with full governance documentation.
                </p>
              </div>
            </div>
          </div>

          {/* Governance Signal */}
          <div className="rounded-lg border border-purple-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-10">
            <div className="flex items-start mb-4">
              <div className="text-5xl mr-6">⚖️</div>
              <div>
                <h3 className="text-2xl font-semibold text-white mb-4">
                  Governance & Compliance
                </h3>
                <p className="text-lg text-gray-300 leading-relaxed">
                  Institutional-grade governance frameworks embedded throughout. Immutable audit trails capture 
                  every decision, attribution explainability ensures fiduciary accountability, and compliance-ready 
                  monitoring supports regulatory examination. Board-ready documentation providing investment 
                  committees with complete transparency into methodology, risk controls, and performance attribution. 
                  Built for fiduciary oversight, not retail convenience.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Enterprise Call to Action */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-12 text-center">
            <div className="text-5xl mb-6">🏢</div>
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl mb-6">
              Enterprise Solutions
            </h2>
            <p className="text-lg text-gray-300 mb-8 max-w-3xl mx-auto">
              Flexible deployment models for strategic enterprise buyers and partners. Platform licensing for 
              internal operations, WAVES™ portfolio mandates for externally-managed allocations, or integrated 
              solutions combining both. Acquisition-grade positioning with complete technical transparency, 
              integration readiness, and institutional use case validation.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="/enterprise"
                className="inline-block rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-4 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400 hover:shadow-lg hover:shadow-cyan-500/50"
              >
                Explore Enterprise Solutions
              </a>
              <a
                href="/demo"
                className="inline-block rounded-md border border-cyan-500 px-8 py-4 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10 hover:shadow-lg hover:shadow-cyan-500/30"
              >
                Request Strategic Demo
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl mb-6">
            Ready to Transform Your Decision Infrastructure?
          </h2>
          <p className="text-lg text-gray-400 mb-12 max-w-2xl mx-auto">
            Connect with our institutional team to explore platform deployment, portfolio access, or integrated solutions tailored to your organization.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/demo"
              className="inline-block rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-4 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400 hover:shadow-lg hover:shadow-cyan-500/50"
            >
              Request Institutional Demo
            </Link>
            <Link
              href="/contact"
              className="inline-block rounded-md border border-cyan-500 px-8 py-4 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10 hover:shadow-lg hover:shadow-cyan-500/30"
            >
              Contact Our Team
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
