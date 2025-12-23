import Hero from "@/components/Hero";
import ProofStrip from "@/components/ProofStrip";
import HowItWorks from "@/components/HowItWorks";
import Audience from "@/components/Audience";
import InstitutionalCare from "@/components/InstitutionalCare";
import WavesDistinction from "@/components/WavesDistinction";
import AcquisitionIntegration from "@/components/AcquisitionIntegration";
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
      
      <InstitutionalCare
        title={siteContent.institutionalCare.title}
        subtitle={siteContent.institutionalCare.subtitle}
        points={siteContent.institutionalCare.points}
      />
      
      <WavesDistinction
        title={siteContent.wavesDistinction.title}
        subtitle={siteContent.wavesDistinction.subtitle}
        intro={siteContent.wavesDistinction.intro}
        distinctions={siteContent.wavesDistinction.distinctions}
      />
      
      <AcquisitionIntegration
        title={siteContent.acquisitionIntegration.title}
        subtitle={siteContent.acquisitionIntegration.subtitle}
        intro={siteContent.acquisitionIntegration.intro}
        capabilities={siteContent.acquisitionIntegration.capabilities}
      />
    </main>
  );
}
