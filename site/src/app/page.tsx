import Hero from "@/components/Hero";
import FeatureGrid from "@/components/FeatureGrid";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export default function Home() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.home.hero.title}
        subtitle={siteContent.home.hero.subtitle}
        ctaText={siteContent.home.hero.ctaText}
        ctaLink={siteContent.home.hero.ctaLink}
        secondaryCtaText={siteContent.home.hero.secondaryCtaText}
        secondaryCtaLink={siteContent.home.hero.secondaryCtaLink}
      />
      <FeatureGrid features={siteContent.home.features} />
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
