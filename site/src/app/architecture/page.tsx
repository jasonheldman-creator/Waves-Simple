import type { Metadata } from "next";
import Hero from "@/components/Hero";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Architecture - WAVES Intelligence",
  description:
    "Enterprise-grade infrastructure designed for institutional performance, security, and scalability.",
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
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-lg text-gray-300">{siteContent.architecture.description}</p>
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
        title="Ready to Learn More?"
        description="Speak with our engineering team to understand how our architecture can support your requirements."
        primaryButtonText="Contact Sales"
        primaryButtonLink="/contact"
        secondaryButtonText="View Platform"
        secondaryButtonLink="/platform"
      />
    </main>
  );
}
