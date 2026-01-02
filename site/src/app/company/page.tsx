import type { Metadata } from "next";
import Hero from "@/components/Hero";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Company - WAVES Intelligence",
  description:
    "Building the future of institutional portfolio analytics and decision intelligence.",
};

export default function CompanyPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.company.hero.title}
        subtitle={siteContent.company.hero.subtitle}
        ctaText="Get in Touch"
        ctaLink="/contact"
        secondaryCtaText="View Platform"
        secondaryCtaLink="/platform"
      />
      <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl">Our Mission</h2>
          </div>
          <p className="text-lg text-gray-300 text-center mb-12">
            {siteContent.company.description}
          </p>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            {siteContent.company.values.map((value, index) => (
              <div
                key={index}
                className="rounded-lg border border-gray-800 bg-gray-900/50 p-6 text-center"
              >
                <h3 className="text-xl font-semibold text-cyan-400 mb-3">{value.title}</h3>
                <p className="text-gray-400">{value.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Leadership Team Section */}
      <section className="bg-gradient-to-b from-gray-900 to-black py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">Leadership Team</h2>
            <p className="text-lg text-gray-400 max-w-3xl mx-auto">
              Deep cross-cycle market experience, real-world portfolio and risk stewardship,
              platform and AI system design expertise, and strategic capital perspective.
            </p>
          </div>

          <div className="space-y-12">
            {/* Jason Heldman */}
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-8">
              <div className="mb-4">
                <h3 className="text-2xl font-bold text-white mb-1">Jason Heldman</h3>
                <p className="text-cyan-400 font-medium">Founder & Architect</p>
              </div>
              <div className="text-gray-300 space-y-4">
                <p>
                  Jason Heldman is the founder and chief architect of WAVES Intelligence™. With over
                  three decades in the financial industry, Jason&apos;s background spans investment
                  banking, portfolio construction, risk management, and platform design across both
                  retail and institutional environments.
                </p>
                <p>
                  His career has been defined by a consistent focus on discipline — how portfolios
                  are constructed, how risk is controlled, and how performance is explained across
                  market regimes. WAVES Intelligence™ reflects Jason&apos;s long-held view that
                  investment outcomes cannot be separated from decision architecture. The platform
                  was designed to close the gap between theory, execution, and governance by making
                  decisions observable, explainable, and auditable in real time.
                </p>
                <p>
                  Today, Jason&apos;s role centers on system design, strategic direction, and
                  ensuring that WAVES Intelligence™ remains grounded in institutional reality rather
                  than academic abstraction or performance marketing.
                </p>
              </div>
            </div>

            {/* Steve Demas */}
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-8">
              <div className="mb-4">
                <h3 className="text-2xl font-bold text-white mb-1">Steve Demas</h3>
                <p className="text-cyan-400 font-medium">Senior Advisor – Portfolio Stewardship</p>
              </div>
              <div className="text-gray-300 space-y-4">
                <p>
                  Steve Demas has worked alongside Jason in the trenches throughout the development
                  of WAVES Intelligence™, bringing decades of hands-on experience in portfolio
                  management, investment operations, and risk-aware decision-making. His background
                  spans both retail and institutional investment contexts, with deep familiarity in
                  how portfolios behave across different market environments, stress periods, and
                  structural shifts.
                </p>
                <p>
                  As Senior Advisor for Portfolio Stewardship, Steve provides practical oversight
                  rooted in lived experience. His role focuses on reinforcing discipline, validating
                  risk controls, and ensuring that the platform&apos;s decision logic reflects how
                  capital is actually managed in practice.
                </p>
                <p>
                  Steve&apos;s long-standing partnership with Jason is a cornerstone of WAVES
                  Intelligence™. The platform reflects not only design intent, but years of shared
                  operational insight forged through real market exposure.
                </p>
              </div>
            </div>

            {/* Dev Patel */}
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-8">
              <div className="mb-4">
                <h3 className="text-2xl font-bold text-white mb-1">Dev Patel</h3>
                <p className="text-cyan-400 font-medium">Technology & AI Advisor</p>
              </div>
              <div className="text-gray-300 space-y-4">
                <p>
                  Dev Patel brings extensive experience in technology systems, artificial
                  intelligence, and scalable platform architecture. His career has focused on
                  designing and advising on complex systems where reliability, transparency, and
                  performance are essential — qualities that underpin WAVES Intelligence™.
                </p>
                <p>
                  At WAVES, Dev advises on the technical and AI foundations of the platform, helping
                  ensure that advanced analytics and adaptive intelligence are implemented
                  responsibly and explainably. His perspective bridges cutting-edge technology with
                  institutional constraints, reinforcing WAVES Intelligence™&apos;s commitment to
                  clarity, governance, and trust over black-box complexity.
                </p>
                <p>
                  Dev plays a key role in aligning the platform with the future of intelligent
                  financial infrastructure while preserving the rigor required by institutional
                  stakeholders.
                </p>
              </div>
            </div>

            {/* David Tsutsumi */}
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-8">
              <div className="mb-4">
                <h3 className="text-2xl font-bold text-white mb-1">David Tsutsumi</h3>
                <p className="text-cyan-400 font-medium">Strategic & Capital Advisor</p>
              </div>
              <div className="text-gray-300 space-y-4">
                <p>
                  David Tsutsumi advises WAVES Intelligence™ on strategic positioning, capital
                  considerations, and long-term structural planning as the platform scales toward
                  institutional adoption. His background includes strategic advisory work across
                  growth initiatives, capital structuring, and organizational development, with a
                  focus on aligning technology platforms with durable market opportunity.
                </p>
                <p>
                  In his advisory role, David helps ensure that WAVES Intelligence™ is built not
                  just as a product, but as a sustainable platform — capable of supporting
                  enterprise partnerships, institutional integration, and long-term strategic
                  outcomes.
                </p>
              </div>
            </div>
          </div>

          {/* Why This Team Matters */}
          <div className="mt-16 rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
            <h3 className="text-2xl font-bold text-white mb-4">Why This Leadership Team Matters</h3>
            <p className="text-gray-300 mb-6">
              Together, the WAVES Intelligence™ leadership team brings:
            </p>
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-start">
                <span className="text-cyan-400 mr-3">•</span>
                <span>Deep cross-cycle market experience</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-3">•</span>
                <span>Real-world portfolio and risk stewardship</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-3">•</span>
                <span>Platform and AI system design expertise</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-3">•</span>
                <span>Strategic and capital perspective</span>
              </li>
            </ul>
            <p className="text-gray-300 mt-6 italic">
              This combination underpins WAVES Intelligence™&apos;s core philosophy: investment
              decisions deserve the same level of rigor, transparency, and governance as the capital
              they control.
            </p>
          </div>
        </div>
      </section>

      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-8">
            Join Our <span className="text-cyan-400">Team</span>
          </h2>
          <p className="text-lg text-gray-300 mb-8">
            We&apos;re looking for talented individuals passionate about fintech, analytics, and
            institutional investing.
          </p>
          <a
            href="/contact"
            className="inline-block rounded-md border border-cyan-500 px-8 py-3 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10"
          >
            View Opportunities
          </a>
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
