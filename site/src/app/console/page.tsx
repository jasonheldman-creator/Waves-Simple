"use client";

import { useEffect } from "react";
import Hero from "@/components/Hero";
import { siteContent } from "@/content/siteContent";
import Link from "next/link";

export default function ConsolePage() {
  useEffect(() => {
    // Check if NEXT_PUBLIC_CONSOLE_URL is defined and redirect
    const consoleUrl = process.env.NEXT_PUBLIC_CONSOLE_URL;
    if (consoleUrl) {
      window.location.href = consoleUrl;
    }
  }, []);

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
              <div className="mt-8 p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <p className="text-sm text-gray-400">
                  For early access, contact: <span className="text-cyan-400 font-semibold">jasonheldman-creator</span>
                </p>
              </div>
              <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:justify-center">
                <Link
                  href="/"
                  className="rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-3 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400"
                >
                  Return Home
                </Link>
                <Link
                  href="/contact"
                  className="rounded-md border border-cyan-500 px-8 py-3 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10"
                >
                  Contact
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
