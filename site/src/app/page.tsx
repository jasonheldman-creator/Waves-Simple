import Link from "next/link";
import Hero from "@/components/Hero";
import RegimeCard from "@/components/RegimeCard";
import ExecutiveBriefing from "@/components/ExecutiveBriefing";

export default function Home() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title="Decision Infrastructure for Modern Asset Management"
        subtitle="WAVES Intelligence™ provides a unified decision framework that connects portfolio construction, risk control, and performance attribution — demonstrated through live portfolios."
        ctaText="Request Institutional Demo"
        ctaLink="/demo"
        secondaryCtaText="View Platform"
        secondaryCtaLink="/platform"
      />

      {/* Executive Briefing Section */}
      <section className="bg-gradient-to-b from-black to-gray-900 py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Executive Briefing
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              A narrated overview of the WAVES Intelligence™ decision framework
              (approximately 8 minutes).
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <ExecutiveBriefing compact />
            
            <div className="mt-8 text-center">
              <Link
                href="/briefing"
                className="inline-block rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-4 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400 hover:shadow-lg hover:shadow-cyan-500/50"
              >
                Watch Full Briefing
              </Link>
            </div>
          </div>
        </div>
      </section>
      
      {/* Value Proposition Section */}
      <section className="bg-gradient-to-b from-gray-900 via-black to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Executive Overview
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Unified decision infrastructure and live portfolio validation
            </p>
          </div>

          {/* Today's Regime Card */}
          <div className="mb-16">
            <RegimeCard />
          </div>
          
          {/* Two-part explanation: Platform vs Portfolios */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
            {/* Platform Overview */}
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-4xl mb-4">🏗️</div>
              <h3 className="text-xl font-semibold text-white mb-3">
                WAVES Intelligence™ Platform
              </h3>
              <p className="text-base text-gray-300 leading-relaxed mb-4">
                Decision infrastructure for institutional portfolio management. The platform unifies portfolio construction, 
                risk control, and performance attribution into one canonical framework.
              </p>
              <Link
                href="/platform"
                className="inline-flex items-center text-sm font-medium text-cyan-400 hover:text-cyan-300"
              >
                Explore Platform →
              </Link>
            </div>

            {/* WAVES™ Portfolios Overview */}
            <div className="rounded-lg border border-green-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-4xl mb-4">📈</div>
              <h3 className="text-xl font-semibold text-white mb-3">
                WAVES™ Portfolios
              </h3>
              <p className="text-base text-gray-300 leading-relaxed mb-4">
                Live portfolios demonstrating the platform under real market conditions. Each portfolio validates 
                the infrastructure through disciplined, risk-aware execution with full transparency.
              </p>
              <Link
                href="/waves"
                className="inline-flex items-center text-sm font-medium text-green-400 hover:text-green-300"
              >
                View WAVES™ Portfolios →
              </Link>
            </div>
          </div>

          {/* Key Differentiators - Brief */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-6 text-center">
              <div className="text-3xl mb-3">⚖️</div>
              <h4 className="text-base font-semibold text-white mb-2">Governance-First</h4>
              <p className="text-sm text-gray-400">
                Audit trails, compliance monitoring, and board-ready reporting built into every workflow.
              </p>
            </div>
            <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-6 text-center">
              <div className="text-3xl mb-3">📊</div>
              <h4 className="text-base font-semibold text-white mb-2">Attribution Integrity</h4>
              <p className="text-sm text-gray-400">
                Multi-dimensional performance attribution with complete transparency and benchmark controls.
              </p>
            </div>
            <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-6 text-center">
              <div className="text-3xl mb-3">🎯</div>
              <h4 className="text-base font-semibold text-white mb-2">Live Validation</h4>
              <p className="text-sm text-gray-400">
                Platform proven through WAVES™ portfolios operating under real market conditions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Navigation to Key Sections */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-2xl font-bold text-white sm:text-3xl mb-4">
              Explore by Area
            </h2>
            <p className="text-base text-gray-400 max-w-2xl mx-auto">
              Learn more about specific aspects of WAVES Intelligence™
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Link
              href="/platform"
              className="group rounded-lg border border-gray-800 bg-gray-900/50 p-6 transition-all hover:border-cyan-500/50 hover:bg-gray-900"
            >
              <div className="text-3xl mb-3">🏗️</div>
              <h3 className="text-base font-semibold text-white mb-2 group-hover:text-cyan-400">Platform</h3>
              <p className="text-sm text-gray-400">Decision infrastructure capabilities and features</p>
            </Link>
            
            <Link
              href="/waves"
              className="group rounded-lg border border-gray-800 bg-gray-900/50 p-6 transition-all hover:border-green-500/50 hover:bg-gray-900"
            >
              <div className="text-3xl mb-3">📈</div>
              <h3 className="text-base font-semibold text-white mb-2 group-hover:text-green-400">WAVES™</h3>
              <p className="text-sm text-gray-400">Live portfolios and performance tracking</p>
            </Link>
            
            <Link
              href="/governance"
              className="group rounded-lg border border-gray-800 bg-gray-900/50 p-6 transition-all hover:border-purple-500/50 hover:bg-gray-900"
            >
              <div className="text-3xl mb-3">⚖️</div>
              <h3 className="text-base font-semibold text-white mb-2 group-hover:text-purple-400">Governance</h3>
              <p className="text-sm text-gray-400">Compliance, audit, and oversight frameworks</p>
            </Link>
            
            <Link
              href="/enterprise"
              className="group rounded-lg border border-gray-800 bg-gray-900/50 p-6 transition-all hover:border-cyan-500/50 hover:bg-gray-900"
            >
              <div className="text-3xl mb-3">🏢</div>
              <h3 className="text-base font-semibold text-white mb-2 group-hover:text-cyan-400">Enterprise</h3>
              <p className="text-sm text-gray-400">Deployment options and integration</p>
            </Link>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-6">
            Connect With Our Team
          </h2>
          <p className="text-lg text-gray-400 mb-12 max-w-2xl mx-auto">
            Discuss platform deployment, portfolio access, or integrated solutions for your organization.
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
              Contact
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
