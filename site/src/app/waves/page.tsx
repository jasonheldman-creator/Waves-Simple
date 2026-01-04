import type { Metadata } from "next";
import Hero from "@/components/Hero";
import WaveCards from "@/components/WaveCards";
import CallToAction from "@/components/CallToAction";
import ScreenshotGallery from "@/components/ScreenshotGallery";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Investment Waves - WAVES Intelligence",
  description:
    "Strategic investment themes designed to capture alpha across different market conditions.",
};

export default function WavesPage() {
  const platformScreenshots = [
    {
      src: "/screens/portfolio-dashboard.svg",
      alt: "Portfolio Dashboard",
      caption: "Real-time portfolio monitoring with position-level exposure analytics",
    },
    {
      src: "/screens/attribution-analysis.svg",
      alt: "Attribution Analysis",
      caption: "Multi-factor performance attribution with benchmark integrity controls",
    },
    {
      src: "/screens/governance-console.svg",
      alt: "Governance Console",
      caption: "Comprehensive audit trails and compliance monitoring infrastructure",
    },
  ];

  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.waves.hero.title}
        subtitle={siteContent.waves.hero.subtitle}
        ctaText="Request Institutional Demo"
        ctaLink="/demo"
        secondaryCtaText="View Platform"
        secondaryCtaLink="/platform"
      />

      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-lg text-gray-300">{siteContent.waves.description}</p>
        </div>
      </section>

      {/* Risk Modes Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Risk Modes & Portfolio Construction
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Disciplined execution frameworks designed for institutional risk management
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">📉</div>
              <h3 className="text-2xl font-semibold text-white mb-4">Alpha-Minus-Beta Mode</h3>
              <p className="text-gray-400 mb-6">
                Isolating strategy alpha by systematically removing broad market exposure. WAVES™
                portfolios targeting returns independent of benchmark performance through
                disciplined beta-hedging frameworks.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Market-neutral construction reducing systematic beta exposure
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Continuous factor monitoring and rebalancing discipline
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Attribution clarity separating alpha from beta contributions
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">🔒</div>
              <h3 className="text-2xl font-semibold text-white mb-4">Private Logic Mode</h3>
              <p className="text-gray-400 mb-6">
                Proprietary signal generation and portfolio construction methodologies developed
                through institutional research. Transparent governance with protected intellectual
                property.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Institutional-grade research methodology with IP protection
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Complete audit trails without revealing proprietary signals
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Board-ready documentation balancing transparency and confidentiality
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Alpha Capture Mechanisms Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Alpha Capture Mechanisms
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Multi-layered approach to systematic alpha generation across market conditions
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">🎯</div>
              <h3 className="text-xl font-semibold text-white mb-4">Selection Alpha</h3>
              <p className="text-gray-400 mb-4">
                Systematic security selection through quantitative screening, fundamental
                validation, and institutional research frameworks.
              </p>
              <div className="space-y-2 pt-4 border-t border-gray-700">
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Multi-factor screening models</span>
                </div>
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Benchmark-aware positioning</span>
                </div>
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Risk-adjusted security weights</span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">🎛️</div>
              <h3 className="text-xl font-semibold text-white mb-4">Overlay Alpha</h3>
              <p className="text-gray-400 mb-4">
                Dynamic portfolio adjustments responding to market regime shifts, volatility
                conditions, and risk environment changes.
              </p>
              <div className="space-y-2 pt-4 border-t border-gray-700">
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Regime-aware positioning</span>
                </div>
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Volatility overlay management</span>
                </div>
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Tactical exposure adjustments</span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">📊</div>
              <h3 className="text-xl font-semibold text-white mb-4">Regime Alpha</h3>
              <p className="text-gray-400 mb-4">
                Strategic positioning adapting to macroeconomic regimes, market cycles, and
                institutional risk-return environments.
              </p>
              <div className="space-y-2 pt-4 border-t border-gray-700">
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Macro regime classification</span>
                </div>
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Cycle-aware allocations</span>
                </div>
                <div className="flex items-start text-sm text-gray-500">
                  <span className="text-cyan-400 mr-2">•</span>
                  <span>Risk budget optimization</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <WaveCards />

      {/* Platform Console Screenshots */}
      <ScreenshotGallery screenshots={platformScreenshots} />

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
