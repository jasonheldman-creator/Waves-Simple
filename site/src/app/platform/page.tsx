import type { Metadata } from "next";
import Hero from "@/components/Hero";
import FeatureGrid from "@/components/FeatureGrid";
import ScreenshotGallery from "@/components/ScreenshotGallery";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Platform - WAVES Intelligence",
  description: "A comprehensive suite of tools designed for institutional portfolio management and analytics.",
};

export default function PlatformPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.platform.hero.title}
        subtitle={siteContent.platform.hero.subtitle}
        ctaText="Get Started"
        ctaLink="/console"
        secondaryCtaText="Contact Sales"
        secondaryCtaLink="/contact"
      />
      
      {/* How WAVES Is Different Section */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              {siteContent.platform.differentiation.title}
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              {siteContent.platform.differentiation.subtitle}
            </p>
            <p className="mt-6 text-base text-gray-300 max-w-4xl mx-auto leading-relaxed">
              {siteContent.platform.differentiation.content}
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-12">
            {siteContent.platform.differentiation.features.map((feature, index) => (
              <div
                key={index}
                className="group rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10"
              >
                <div className="mb-3 text-4xl">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400 mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-400 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <FeatureGrid features={siteContent.platform.features} columns={2} />
      <ScreenshotGallery />
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
