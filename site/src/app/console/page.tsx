import type { Metadata } from "next";
import Hero from "@/components/Hero";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Console - WAVES Intelligence",
  description: "Access the institutional-grade analytics platform powering sophisticated investment decisions.",
};

export default function ConsolePage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.console.hero.title}
        subtitle={siteContent.console.hero.subtitle}
      />
      <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-8 sm:p-12">
            <div className="text-center">
              <div className="mb-6 text-6xl">🎛️</div>
              <h2 className="text-2xl font-bold text-white sm:text-3xl">
                Console Access
              </h2>
              <p className="mt-4 text-lg text-gray-400">
                {siteContent.console.description}
              </p>
              <div className="mt-8">
                <p className="text-sm text-gray-500 mb-4">
                  The Streamlit-based console is available at:
                </p>
                <div className="inline-block rounded-md bg-gray-800 px-4 py-2 font-mono text-cyan-400">
                  console.wavesintelligence.com
                </div>
              </div>
              <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:justify-center">
                <a
                  href="https://console.wavesintelligence.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-3 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400"
                >
                  Access Console
                </a>
                <a
                  href="/contact"
                  className="rounded-md border border-cyan-500 px-8 py-3 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10"
                >
                  Request Demo
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>
      <CallToAction
        title="Need Help?"
        description="Our team is available to assist with onboarding, training, and technical support."
        primaryButtonText="Contact Support"
        primaryButtonLink="/contact"
        secondaryButtonText="View Documentation"
        secondaryButtonLink="/platform"
      />
    </main>
  );
}
