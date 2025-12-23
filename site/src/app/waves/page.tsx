import type { Metadata } from "next";
import Hero from "@/components/Hero";
import WaveCards from "@/components/WaveCards";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Investment Waves - WAVES Intelligence",
  description:
    "Strategic investment themes designed to capture alpha across different market conditions.",
};

export default function WavesPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.waves.hero.title}
        subtitle={siteContent.waves.hero.subtitle}
        ctaText="Explore Platform"
        ctaLink="/platform"
        secondaryCtaText="Learn More"
        secondaryCtaLink="/contact"
      />
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-lg text-gray-300">{siteContent.waves.description}</p>
        </div>
      </section>
      <WaveCards count={15} />
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
