import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
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
