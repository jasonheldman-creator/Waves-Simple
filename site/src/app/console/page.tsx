"use client";

import { useEffect } from "react";
import Hero from "@/components/Hero";
import { siteContent } from "@/content/siteContent";
import Link from "next/link";

export default function ConsolePage() {
  useEffect(() => {
    // Check if NEXT_PUBLIC_CONSOLE_URL is defined and redirect
    const consoleUrl = process.env.NEXT_PUBLIC_CONSOLE_URL;
    if (consoleUrl) {
      window.location.href = consoleUrl;
    }
  }, []);

  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.console.hero.title}
        subtitle={siteContent.console.hero.subtitle}
      />
      <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-8 sm:p-12">
            <div className="text-center">
              <div className="mb-6 text-6xl">🎛️</div>
              <h2 className="text-2xl font-bold text-white sm:text-3xl">
                Institutional Console Access
              </h2>
              <p className="mt-4 text-lg text-gray-300">
                {siteContent.console.description}
              </p>

              {/* Institutional Readiness Features */}
              <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="rounded-lg bg-gray-800/50 p-4 border border-gray-700">
                  <div className="text-2xl mb-2">🔍</div>
                  <h3 className="text-sm font-semibold text-cyan-400 mb-1">Risk Overlays</h3>
                  <p className="text-xs text-gray-400">
                    Real-time risk monitoring with multi-layer overlay analysis for comprehensive exposure management
                  </p>
                </div>
                <div className="rounded-lg bg-gray-800/50 p-4 border border-gray-700">
                  <div className="text-2xl mb-2">📊</div>
                  <h3 className="text-sm font-semibold text-cyan-400 mb-1">Benchmark Integrity</h3>
                  <p className="text-xs text-gray-400">
                    Continuous validation of performance benchmarks ensuring accuracy and consistency across analytics
                  </p>
                </div>
                <div className="rounded-lg bg-gray-800/50 p-4 border border-gray-700">
                  <div className="text-2xl mb-2">📝</div>
                  <h3 className="text-sm font-semibold text-cyan-400 mb-1">Complete Audit Trails</h3>
                  <p className="text-xs text-gray-400">
                    Immutable logging of all decisions, allocations, and system activities for regulatory compliance
                  </p>
                </div>
                <div className="rounded-lg bg-gray-800/50 p-4 border border-gray-700">
                  <div className="text-2xl mb-2">✅</div>
                  <h3 className="text-sm font-semibold text-cyan-400 mb-1">Governance Ready</h3>
                  <p className="text-xs text-gray-400">
                    Built-in compliance frameworks and fiduciary controls meeting institutional standards
                  </p>
                </div>
              </div>

              {/* Compliance Statement */}
              <div className="mt-8 p-6 rounded-lg bg-gradient-to-r from-cyan-900/20 to-green-900/20 border border-cyan-500/30">
                <h3 className="text-lg font-semibold text-white mb-3">Institutional Use Readiness</h3>
                <ul className="text-sm text-gray-300 text-left space-y-2 max-w-2xl mx-auto">
                  <li className="flex items-start">
                    <span className="text-cyan-400 mr-2">•</span>
                    <span>SOC 2 Type II certified infrastructure with comprehensive security controls</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-cyan-400 mr-2">•</span>
                    <span>End-to-end encryption (TLS 1.3 in transit, AES-256 at rest) with HSM key management</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-cyan-400 mr-2">•</span>
                    <span>Role-based access control (RBAC) with mandatory multi-factor authentication</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-cyan-400 mr-2">•</span>
                    <span>Immutable audit trails for all portfolio activities and decision workflows</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-cyan-400 mr-2">•</span>
                    <span>GDPR compliant with support for SEC, FINRA, and fiduciary regulatory frameworks</span>
                  </li>
                </ul>
              </div>

              <div className="mt-8 p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <p className="text-sm text-gray-400">
                  For institutional access and onboarding, contact: <span className="text-cyan-400 font-semibold">jasonheldman-creator</span>
                </p>
              </div>
              <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:justify-center">
                <Link
                  href="/"
                  className="rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-3 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400"
                >
                  Return Home
                </Link>
                <Link
                  href="/contact"
                  className="rounded-md border border-cyan-500 px-8 py-3 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10"
                >
                  Contact Team
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Intended Users Section */}
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl">
              {siteContent.console.intendedUsers.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400">
              {siteContent.console.intendedUsers.subtitle}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
            {siteContent.console.intendedUsers.roles.map((role, index) => (
              <div
                key={index}
                className="rounded-lg border border-gray-800 bg-gray-900/50 p-6 transition-all hover:border-cyan-500/50"
              >
                <div className="text-4xl mb-4">{role.icon}</div>
                <h3 className="text-lg font-semibold text-white mb-2">{role.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed">{role.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Governance & Oversight Readiness Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl">
              {siteContent.console.governance.title}
            </h2>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
            {siteContent.console.governance.features.map((feature, index) => (
              <div
                key={index}
                className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
