import type { Metadata } from "next";
import Hero from "@/components/Hero";
import CallToAction from "@/components/CallToAction";

export const metadata: Metadata = {
  title: "Enterprise Solutions - WAVES Intelligence",
  description:
    "Enterprise deployment, integration pathways, and institutional acquisition frameworks for WAVES Intelligence™.",
};

export default function EnterprisePage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title="Enterprise Solutions"
        subtitle="Integration-ready infrastructure designed for institutional deployment, M&A evaluation, and enterprise-scale operations. Complete technical transparency supporting acquisition pathways and strategic partnerships."
        ctaText="Request Institutional Demo"
        ctaLink="/demo"
        secondaryCtaText="View Architecture"
        secondaryCtaLink="/architecture"
      />

      {/* Deployment Models Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Flexible Deployment Models
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Modular architecture supporting diverse institutional engagement pathways
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">🏗️</div>
              <h3 className="text-2xl font-semibold text-white mb-4">Platform Licensing</h3>
              <p className="text-gray-400 mb-6">
                License WAVES Intelligence™ decision infrastructure for internal deployment across
                your organization.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    On-premise or private cloud deployment options
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Complete source code access and customization rights
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    White-label capabilities for institutional branding
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Ongoing technical support and platform updates
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">📈</div>
              <h3 className="text-2xl font-semibold text-white mb-4">Managed Portfolios</h3>
              <p className="text-gray-400 mb-6">
                Access WAVES™ portfolios as externally-managed mandates under your institutional
                oversight.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Live validation of platform decision infrastructure
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Complete transparency into construction methodology
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Real-time exposure analytics and attribution reporting
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Institutional governance and compliance frameworks
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">🔗</div>
              <h3 className="text-2xl font-semibold text-white mb-4">Integrated Solutions</h3>
              <p className="text-gray-400 mb-6">
                Combine platform infrastructure with portfolio mandates for comprehensive
                institutional deployment.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Platform for internal operations and oversight
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    WAVES™ portfolios as externally-managed allocations
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Unified governance across infrastructure and mandates
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Flexible commercial structures for hybrid engagement
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Integration & Acquisition Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Acquisition & Integration Pathways
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Complete technical transparency supporting M&A evaluation and strategic acquisition
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">📋</div>
              <h3 className="text-2xl font-semibold text-white mb-4">Due Diligence Ready</h3>
              <p className="text-gray-400 mb-6">
                Complete documentation, architecture transparency, and audit trails supporting
                institutional acquisition evaluation.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Full system architecture and data flow documentation
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Complete source code access with IP clarity
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Operational performance benchmarks and metrics
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Regulatory compliance and audit trail documentation
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Technology stack validation and dependency mapping
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8">
              <div className="text-5xl mb-4">🔧</div>
              <h3 className="text-2xl font-semibold text-white mb-4">Integration Infrastructure</h3>
              <p className="text-gray-400 mb-6">
                API-first architecture with comprehensive integration specifications supporting
                seamless deployment.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    RESTful API with complete endpoint documentation
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Data migration pathways and ETL specifications
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Single sign-on (SSO) and enterprise authentication
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Existing system interoperability frameworks
                  </span>
                </div>
                <div className="flex items-start">
                  <span className="text-cyan-400 mr-3">✓</span>
                  <span className="text-sm text-gray-300">
                    Scalability validation and performance optimization
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Enterprise Support Section */}
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Enterprise-Grade Support
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Dedicated institutional support across deployment, operations, and ongoing
              optimization
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="text-4xl mb-4">👨‍💼</div>
              <h3 className="text-lg font-semibold text-white mb-3">
                Dedicated Account Management
              </h3>
              <p className="text-sm text-gray-400">
                Senior relationship managers supporting strategic deployment and institutional
                requirements
              </p>
            </div>

            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="text-4xl mb-4">🛠️</div>
              <h3 className="text-lg font-semibold text-white mb-3">Technical Implementation</h3>
              <p className="text-sm text-gray-400">
                Expert engineering support for deployment, integration, and system optimization
              </p>
            </div>

            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="text-4xl mb-4">📚</div>
              <h3 className="text-lg font-semibold text-white mb-3">Training & Onboarding</h3>
              <p className="text-sm text-gray-400">
                Comprehensive user training and institutional workflow customization
              </p>
            </div>

            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="text-4xl mb-4">🔄</div>
              <h3 className="text-lg font-semibold text-white mb-3">Continuous Updates</h3>
              <p className="text-sm text-gray-400">
                Ongoing platform enhancements, feature releases, and institutional feedback
                integration
              </p>
            </div>
          </div>
        </div>
      </section>

      <CallToAction
        title="Discuss Enterprise Deployment"
        description="Connect with our institutional team to explore deployment models, integration pathways, and acquisition frameworks tailored to your organization's needs."
        primaryButtonText="Request Enterprise Consultation"
        primaryButtonLink="/demo"
        secondaryButtonText="View Technical Documentation"
        secondaryButtonLink="/architecture"
      />
    </main>
  );
}
