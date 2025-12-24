import type { Metadata } from "next";
import Link from "next/link";
import ExecutiveBriefing from "@/components/ExecutiveBriefing";

export const metadata: Metadata = {
  title: "Executive Briefing - WAVES Intelligence",
  description:
    "A narrated overview of the WAVES Intelligence™ decision framework (approximately 8 minutes).",
};

export default function BriefingPage() {
  return (
    <main className="min-h-screen bg-black">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-gray-900 via-black to-black py-20 sm:py-28">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-white sm:text-5xl lg:text-6xl">
              Executive Briefing
            </h1>
            <p className="mt-6 text-lg text-gray-400 max-w-3xl mx-auto">
              A narrated overview of the WAVES Intelligence™ decision framework
              (approximately 8 minutes).
            </p>
            <p className="mt-4 text-base text-gray-300 max-w-3xl mx-auto">
              This briefing provides a comprehensive introduction to our institutional-grade
              decision infrastructure, covering portfolio construction, risk management, and
              performance attribution within a unified framework.
            </p>
          </div>
        </div>
      </section>

      {/* Video Player Section */}
      <section className="bg-black py-12 sm:py-16">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
          <ExecutiveBriefing />
        </div>
      </section>

      {/* Optional Resources Section */}
      <section className="bg-black py-12 sm:py-16 border-t border-gray-800">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-white sm:text-3xl mb-4">
              Additional Resources
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Optional materials to complement the briefing video.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Presentation Slides */}
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="flex items-start">
                <div className="text-3xl mr-4">📊</div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2">
                    Presentation Slides
                  </h3>
                  <p className="text-sm text-gray-400 mb-4">
                    Download the optional PowerPoint deck accompanying this briefing.
                  </p>
                  <a
                    href="/briefing/executive-briefing-slides.pptx"
                    className="inline-flex items-center text-sm font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
                    download
                  >
                    Download PPTX (Optional) →
                  </a>
                </div>
              </div>
            </div>

            {/* Platform Overview */}
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
              <div className="flex items-start">
                <div className="text-3xl mr-4">🏗️</div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2">
                    Platform Details
                  </h3>
                  <p className="text-sm text-gray-400 mb-4">
                    Explore the full capabilities of WAVES Intelligence™ platform.
                  </p>
                  <Link
                    href="/platform"
                    className="inline-flex items-center text-sm font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
                  >
                    View Platform →
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-6">
            Ready to Learn More?
          </h2>
          <p className="text-lg text-gray-400 mb-12 max-w-2xl mx-auto">
            Schedule a personalized demo to see how WAVES Intelligence™ can serve your
            organization's needs.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/demo"
              className="inline-block rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-4 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400 hover:shadow-lg hover:shadow-cyan-500/50"
            >
              Request Institutional Demo
            </Link>
            <Link
              href="/contact"
              className="inline-block rounded-md border border-cyan-500 px-8 py-4 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10 hover:shadow-lg hover:shadow-cyan-500/30"
            >
              Contact Us
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
