import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import MarketTape from "@/components/MarketTape";
import DataStateBadge from "@/components/DataStateBadge";

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://www.wavesintelligence.app";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "WAVES Intelligence™ - Decision Infrastructure for Modern Asset Management",
  description:
    "WAVES Intelligence™ unifies portfolio construction, exposure, attribution, and governance into one canonical source of truth.",
  keywords: [
    "asset management",
    "portfolio construction",
    "attribution",
    "governance",
    "decision infrastructure",
  ],
  authors: [{ name: "WAVES Intelligence" }],
  openGraph: {
    title: "WAVES Intelligence™ - Decision Infrastructure for Modern Asset Management",
    description:
      "WAVES Intelligence™ unifies portfolio construction, exposure, attribution, and governance into one canonical source of truth.",
    type: "website",
    url: siteUrl,
    siteName: "WAVES Intelligence",
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
        <DataStateBadge />
        <MarketTape />
        {children}
        <Footer />
      </body>
    </html>
  );
}
