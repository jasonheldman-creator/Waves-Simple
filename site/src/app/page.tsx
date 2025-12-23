import Link from "next/link";
import Hero from "@/components/Hero";

export default function Home() {
  const navigationCards = [
    {
      href: "/platform",
      icon: "🏗️",
      title: "Platform",
      description: "Decision infrastructure for institutional portfolio management—complete transparency, attribution integrity, and governance-first operations.",
      highlights: [
        "Multi-dimensional exposure analytics",
        "Real-time compliance monitoring",
        "Board-ready reporting frameworks"
      ]
    },
    {
      href: "/waves",
      icon: "📈",
      title: "WAVES™ Portfolios",
      description: "Live validation of platform infrastructure through disciplined, benchmark-aware execution under real market conditions.",
      highlights: [
        "Systematic decision-making proof",
        "Complete construction transparency",
        "Institutional governance controls"
      ]
    },
    {
      href: "/governance",
      icon: "⚖️",
      title: "Governance",
      description: "Institutional-grade frameworks for fiduciary oversight, regulatory compliance, and operational transparency.",
      highlights: [
        "Immutable audit trails",
        "Regulatory compliance frameworks",
        "Risk limit enforcement"
      ]
    },
    {
      href: "/enterprise",
      icon: "🏢",
      title: "Enterprise Solutions",
      description: "Flexible deployment models supporting platform licensing, managed mandates, and acquisition pathways.",
      highlights: [
        "Integration-ready infrastructure",
        "M&A evaluation support",
        "Scalable deployment options"
      ]
    },
    {
      href: "/company",
      icon: "🎯",
      title: "Company",
      description: "Building the future of institutional decision infrastructure with transparency, governance, and operational excellence.",
      highlights: [
        "Institutional expertise",
        "Technical transparency",
        "Commitment to compliance"
      ]
    },
  ];

  return (
    <main className="min-h-screen bg-black">
      <Hero
        title="Institutional Decision Infrastructure"
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
              One Platform. Complete Transparency.
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Decision infrastructure and portfolio proof unified for institutional governance
            </p>
          </div>
          
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mb-12">
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 text-center">
              <div className="text-5xl mb-4">🔍</div>
              <h3 className="text-xl font-semibold text-white mb-3">
                Platform vs WAVES™
              </h3>
              <p className="text-gray-400 text-sm">
                <span className="text-cyan-400 font-semibold">Platform:</span> Decision infrastructure for institutional operations<br/>
                <span className="text-green-400 font-semibold">WAVES™:</span> Live portfolios demonstrating platform effectiveness
              </p>
            </div>
            
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 text-center">
              <div className="text-5xl mb-4">📊</div>
              <h3 className="text-xl font-semibold text-white mb-3">
                Institutional Grade
              </h3>
              <p className="text-gray-400 text-sm">
                Built for CIOs, risk officers, portfolio managers, and compliance teams—governance-native architecture at every layer
              </p>
            </div>
            
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 text-center">
              <div className="text-5xl mb-4">🔗</div>
              <h3 className="text-xl font-semibold text-white mb-3">
                Flexible Engagement
              </h3>
              <p className="text-gray-400 text-sm">
                License infrastructure, access portfolios as mandates, or integrate both—modular deployment pathways
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Navigation Cards Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Explore Our Solutions
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Comprehensive infrastructure addressing every aspect of institutional decision-making
            </p>
          </div>
          
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-2">
            {navigationCards.map((card, index) => (
              <Link
                key={index}
                href={card.href}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/20 hover:scale-[1.02]"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="text-5xl">{card.icon}</div>
                  <svg
                    className="h-6 w-6 text-gray-600 transition-all group-hover:text-cyan-400 group-hover:translate-x-1"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                <h3 className="text-2xl font-semibold text-white group-hover:text-cyan-400 mb-3 transition-colors">
                  {card.title}
                </h3>
                <p className="text-gray-400 leading-relaxed mb-6">
                  {card.description}
                </p>
                <div className="space-y-2 pt-4 border-t border-gray-700">
                  {card.highlights.map((highlight, hIndex) => (
                    <div key={hIndex} className="flex items-start text-sm text-gray-500">
                      <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                      <span>{highlight}</span>
                    </div>
                  ))}
                </div>
              </Link>
            ))}
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
