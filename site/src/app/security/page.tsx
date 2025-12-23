import type { Metadata } from "next";
import Hero from "@/components/Hero";
import FeatureGrid from "@/components/FeatureGrid";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Security - WAVES Intelligence",
  description: "Institutional-grade security controls and compliance frameworks protecting your sensitive data.",
};

export default function SecurityPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.security.hero.title}
        subtitle={siteContent.security.hero.subtitle}
        ctaText="View Architecture"
        ctaLink="/architecture"
        secondaryCtaText="Contact Us"
        secondaryCtaLink="/contact"
      />
      <FeatureGrid features={siteContent.security.features} />
      <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl">
              Security <span className="text-cyan-400">Standards</span>
            </h2>
            <p className="mt-4 text-lg text-gray-400">
              We maintain the highest security standards and compliance certifications
            </p>
          </div>
          <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
            {["SOC 2 Type II", "GDPR Compliant", "ISO 27001", "PCI DSS"].map((cert) => (
              <div
                key={cert}
                className="rounded-lg border border-gray-800 bg-gray-900/50 p-6 text-center"
              >
                <div className="text-3xl mb-2">✓</div>
                <div className="text-sm font-semibold text-white">{cert}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
      <CallToAction
        title="Security Questions?"
        description="Our security team is available to answer your questions and provide detailed documentation."
        primaryButtonText="Contact Security Team"
        primaryButtonLink="/contact"
        secondaryButtonText="View Architecture"
        secondaryButtonLink="/architecture"
      />
    </main>
  );
}
