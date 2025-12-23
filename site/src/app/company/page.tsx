import type { Metadata } from "next";
import Hero from "@/components/Hero";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Company - WAVES Intelligence",
  description: "Building the future of institutional portfolio analytics and decision intelligence.",
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
