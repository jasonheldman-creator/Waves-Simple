"use client";

import { useState } from "react";

interface DemoFormProps {
  title?: string;
  subtitle?: string;
}

export default function DemoForm({ title, subtitle }: DemoFormProps) {
  const [formData, setFormData] = useState({
    name: "",
    firm: "",
    role: "",
    email: "",
    currentSystem: "",
    primaryChallenge: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError("");

    // Basic validation
    if (
      !formData.name ||
      !formData.firm ||
      !formData.role ||
      !formData.email ||
      !formData.currentSystem ||
      !formData.primaryChallenge
    ) {
      setError("Please fill in all fields");
      setIsSubmitting(false);
      return;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setError("Please enter a valid email address");
      setIsSubmitting(false);
      return;
    }

    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSuccess(true);
      setFormData({
        name: "",
        firm: "",
        role: "",
        email: "",
        currentSystem: "",
        primaryChallenge: "",
      });
    }, 1000);
  };

  if (isSuccess) {
    return (
      <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
        <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
          <div className="rounded-lg border border-cyan-500/50 bg-gradient-to-br from-gray-900 to-gray-800/50 p-12 text-center shadow-lg shadow-cyan-500/20">
            <div className="mb-6 text-6xl">✓</div>
            <h2 className="text-3xl font-bold text-white mb-4">Thank You for Your Interest</h2>
            <p className="text-lg text-gray-300 mb-6">
              Your demo request has been received. Our team will reach out to you shortly to
              schedule an institutional demonstration of WAVES Intelligence™.
            </p>
            <button
              onClick={() => setIsSuccess(false)}
              className="rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-3 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400"
            >
              Submit Another Request
            </button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
      <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
        {(title || subtitle) && (
          <div className="text-center mb-12">
            {title && (
              <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">{title}</h2>
            )}
            {subtitle && <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto">{subtitle}</p>}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {/* Name */}
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
                Name *
              </label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
                className="w-full rounded-md border border-gray-700 bg-gray-900 px-4 py-3 text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                placeholder="John Doe"
              />
            </div>

            {/* Firm */}
            <div>
              <label htmlFor="firm" className="block text-sm font-medium text-gray-300 mb-2">
                Firm *
              </label>
              <input
                type="text"
                id="firm"
                name="firm"
                value={formData.firm}
                onChange={handleChange}
                required
                className="w-full rounded-md border border-gray-700 bg-gray-900 px-4 py-3 text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                placeholder="Investment Firm LLC"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {/* Role */}
            <div>
              <label htmlFor="role" className="block text-sm font-medium text-gray-300 mb-2">
                Role *
              </label>
              <input
                type="text"
                id="role"
                name="role"
                value={formData.role}
                onChange={handleChange}
                required
                className="w-full rounded-md border border-gray-700 bg-gray-900 px-4 py-3 text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                placeholder="Portfolio Manager"
              />
            </div>

            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                Email *
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                className="w-full rounded-md border border-gray-700 bg-gray-900 px-4 py-3 text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                placeholder="john@firm.com"
              />
            </div>
          </div>

          {/* Current System */}
          <div>
            <label htmlFor="currentSystem" className="block text-sm font-medium text-gray-300 mb-2">
              Current System *
            </label>
            <select
              id="currentSystem"
              name="currentSystem"
              value={formData.currentSystem}
              onChange={handleChange}
              required
              className="w-full rounded-md border border-gray-700 bg-gray-900 px-4 py-3 text-white focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
            >
              <option value="">Select your current system</option>
              <option value="bloomberg">Bloomberg Terminal</option>
              <option value="factset">FactSet</option>
              <option value="refinitiv">Refinitiv Eikon</option>
              <option value="excel">Excel-based</option>
              <option value="internal">Internal/Proprietary</option>
              <option value="other">Other</option>
              <option value="none">None</option>
            </select>
          </div>

          {/* Primary Challenge */}
          <div>
            <label
              htmlFor="primaryChallenge"
              className="block text-sm font-medium text-gray-300 mb-2"
            >
              Primary Challenge *
            </label>
            <textarea
              id="primaryChallenge"
              name="primaryChallenge"
              value={formData.primaryChallenge}
              onChange={handleChange}
              required
              rows={4}
              className="w-full rounded-md border border-gray-700 bg-gray-900 px-4 py-3 text-white placeholder-gray-500 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
              placeholder="Describe your primary challenge with portfolio oversight, attribution, or governance..."
            />
          </div>

          {/* Error Message */}
          {error && (
            <div className="rounded-md bg-red-900/20 border border-red-500/50 px-4 py-3 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-4 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? "Submitting..." : "Request Institutional Demo"}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
