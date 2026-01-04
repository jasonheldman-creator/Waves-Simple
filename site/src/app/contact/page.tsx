import type { Metadata } from "next";
import Hero from "@/components/Hero";
import ContactForm from "@/components/ContactForm";
import { siteContent } from "@/content/siteContent";

export const metadata: Metadata = {
  title: "Contact - WAVES Intelligence",
  description: "Get in touch with our team to learn more about WAVES Intelligence.",
};

export default function ContactPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero title={siteContent.contact.hero.title} subtitle={siteContent.contact.hero.subtitle} />
      <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <p className="text-lg text-gray-300">{siteContent.contact.description}</p>
          </div>

          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            {/* Contact Form */}
            <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-8">
              <h2 className="text-2xl font-bold text-white mb-6">Send us a message</h2>
              <ContactForm />
            </div>

            {/* Contact Information */}
            <div className="space-y-6">
              <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
                <div className="flex items-start">
                  <div className="text-2xl text-cyan-400 mr-4">📧</div>
                  <div>
                    <h3 className="font-semibold text-white mb-2">Email</h3>
                    <p className="text-gray-400">info@wavesintelligence.com</p>
                    <p className="text-gray-400">sales@wavesintelligence.com</p>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
                <div className="flex items-start">
                  <div className="text-2xl text-cyan-400 mr-4">💼</div>
                  <div>
                    <h3 className="font-semibold text-white mb-2">Sales</h3>
                    <p className="text-gray-400">
                      Interested in the platform? Our sales team is ready to provide a demo and
                      answer your questions.
                    </p>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
                <div className="flex items-start">
                  <div className="text-2xl text-cyan-400 mr-4">🔧</div>
                  <div>
                    <h3 className="font-semibold text-white mb-2">Support</h3>
                    <p className="text-gray-400">
                      For technical support and customer service, existing customers can reach our
                      support team 24/7.
                    </p>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
                <div className="flex items-start">
                  <div className="text-2xl text-cyan-400 mr-4">📰</div>
                  <div>
                    <h3 className="font-semibold text-white mb-2">Press</h3>
                    <p className="text-gray-400">
                      Media inquiries and press requests can be directed to
                      press@wavesintelligence.com
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
