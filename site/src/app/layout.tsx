import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "WAVES Intelligence - Institutional Portfolio Analytics",
  description:
    "Institutional-grade portfolio analytics, alpha attribution, and decision intelligence for sophisticated investors.",
  keywords: [
    "portfolio analytics",
    "alpha attribution",
    "institutional investing",
    "decision intelligence",
    "risk management",
  ],
  authors: [{ name: "WAVES Intelligence" }],
  openGraph: {
    title: "WAVES Intelligence - Institutional Portfolio Analytics",
    description:
      "Institutional-grade portfolio analytics, alpha attribution, and decision intelligence for sophisticated investors.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <Navbar />
        {children}
        <Footer />
      </body>
    </html>
  );
}
