import Hero from "@/components/Hero";
import FeatureGrid from "@/components/FeatureGrid";
import Audience from "@/components/Audience";
import InstitutionalCare from "@/components/InstitutionalCare";
import ProofStrip from "@/components/ProofStrip";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export default function Home() {
  const { home } = siteContent;

  // Convert operationalProof examples to ProofStrip format
  const proofPoints = home.operationalProof.examples.map((example) => ({
    metric: example.metric,
    value: example.value,
    description: example.description,
  }));

  // Convert buyerPersonas to Audience format
  const audienceColumns = home.buyerPersonas.roles.map((role) => ({
    title: role.title,
    description: role.description,
    benefits: role.needs,
    icon: role.icon,
  }));

  // Convert institutionalTrust features to InstitutionalCare format
  const institutionalPoints = home.institutionalTrust.features.map((feature) => ({
    oppose: "Generic Tools",
    position: feature.title,
    description: feature.description,
  }));

  return (
    <main>
      {/* Hero Section */}
      <Hero
        title={home.hero.title}
        subtitle={home.hero.subtitle}
        ctaText={home.hero.ctaText}
        ctaLink={home.hero.ctaLink}
        secondaryCtaText={home.hero.secondaryCtaText}
        secondaryCtaLink={home.hero.secondaryCtaLink}
      />

      {/* Operational Proof Strip */}
      <ProofStrip
        title={home.operationalProof.title}
        subtitle={home.operationalProof.subtitle}
        points={proofPoints}
      />

      {/* Features Grid */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Platform Capabilities
            </h2>
            <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
              Comprehensive infrastructure addressing institutional decision-making needs
            </p>
          </div>
          <FeatureGrid features={home.features} />
        </div>
      </section>

      {/* Buyer Personas / Audience */}
      <Audience
        columns={audienceColumns}
        title={home.buyerPersonas.title}
        subtitle={home.buyerPersonas.subtitle}
      />

      {/* Institutional Trust Signals */}
      <InstitutionalCare
        title={home.institutionalTrust.title}
        subtitle={home.institutionalTrust.subtitle}
        points={institutionalPoints}
      />

      {/* Call to Action */}
      <CallToAction
        title="Ready to Experience Institutional-Grade Decision Infrastructure?"
        description="Request a private demonstration to explore how WAVES Intelligence™ delivers transparency, governance, and explainability across your portfolio operations."
        primaryButtonText="Request Institutional Demo"
        primaryButtonLink="/demo"
        secondaryButtonText="Discuss Platform Licensing"
        secondaryButtonLink="/contact"
      />
    </main>
  );
}
