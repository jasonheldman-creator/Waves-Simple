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
