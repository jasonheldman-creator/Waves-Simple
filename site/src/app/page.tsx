import Hero from "@/components/Hero";
import ProofStrip from "@/components/ProofStrip";
import HowItWorks from "@/components/HowItWorks";
import Audience from "@/components/Audience";
import { siteContent } from "@/content/newSiteContent";

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
      
      <ProofStrip tiles={siteContent.home.proofStrip} />
      
      <HowItWorks
        title={siteContent.home.howItWorks.title}
        subtitle={siteContent.home.howItWorks.subtitle}
        steps={siteContent.home.howItWorks.steps}
      />
      
      <Audience
        title={siteContent.home.audience.title}
        subtitle={siteContent.home.audience.subtitle}
        columns={siteContent.home.audience.columns}
      />
    </main>
  );
}
