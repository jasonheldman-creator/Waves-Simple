import type { Metadata } from "next";
import Hero from "@/components/Hero";
import CallToAction from "@/components/CallToAction";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Press - WAVES Intelligence",
  description: "Latest news, announcements, and media resources from WAVES Intelligence.",
};

export default function PressPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.press.hero.title}
        subtitle={siteContent.press.hero.subtitle}
        ctaText="Contact Press Team"
        ctaLink="/contact"
      />
      <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <p className="text-lg text-gray-300">{siteContent.press.description}</p>
          </div>
          <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-8 sm:p-12">
            <h2 className="text-2xl font-bold text-white mb-6">Media Resources</h2>
            <div className="space-y-4">
              <div className="flex items-start">
                <div className="text-cyan-400 mr-3">📰</div>
                <div>
                  <h3 className="font-semibold text-white">Press Kit</h3>
                  <p className="text-gray-400 text-sm">
                    Download logos, brand guidelines, and company information
                  </p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="text-cyan-400 mr-3">📧</div>
                <div>
                  <h3 className="font-semibold text-white">Media Inquiries</h3>
                  <p className="text-gray-400 text-sm">Contact: press@wavesintelligence.com</p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="text-cyan-400 mr-3">🎤</div>
                <div>
                  <h3 className="font-semibold text-white">Interview Requests</h3>
                  <p className="text-gray-400 text-sm">
                    Schedule interviews with our leadership team
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-white text-center mb-12">
            Recent <span className="text-cyan-400">News</span>
          </h2>
          <div className="space-y-6">
            {[
              {
                date: "2024",
                title: "WAVES Intelligence Platform Launch",
                description:
                  "Introducing institutional-grade portfolio analytics and decision intelligence.",
              },
            ].map((item, index) => (
              <div
                key={index}
                className="rounded-lg border border-gray-800 bg-gray-900/50 p-6 hover:border-cyan-500/50 transition-colors"
              >
                <div className="text-sm text-gray-500 mb-2">{item.date}</div>
                <h3 className="text-xl font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-gray-400">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
      <CallToAction
        title="Get in Touch"
        description="Contact our press team for media inquiries, interviews, or partnership opportunities."
        primaryButtonText="Contact Press Team"
        primaryButtonLink="/contact"
        secondaryButtonText="View Company Info"
        secondaryButtonLink="/company"
      />
    </main>
  );
}
