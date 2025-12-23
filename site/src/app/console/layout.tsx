import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Console - WAVES Intelligence",
  description:
    "Access the institutional-grade analytics platform powering sophisticated investment decisions.",
};

export default function ConsoleLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
