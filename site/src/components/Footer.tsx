import Link from "next/link";

export default function Footer() {
  const currentYear = new Date().getFullYear();
  const buildId = process.env.NEXT_PUBLIC_BUILD_ID || "dev";

  return (
    <footer className="border-t border-gray-800 bg-black">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          {/* Brand */}
          <div className="col-span-1">
            <div className="text-xl font-bold">
              <span className="text-cyan-400">WAVES</span>
              <span className="text-white"> Intelligence™</span>
            </div>
            <p className="mt-4 text-sm text-gray-400">
              Decision infrastructure for modern asset management.
            </p>
          </div>

          {/* Navigation */}
          <div>
            <h3 className="text-sm font-semibold text-white">Navigation</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <Link href="/" className="text-sm text-gray-400 hover:text-cyan-400">
                  Home
                </Link>
              </li>
              <li>
                <Link href="/platform" className="text-sm text-gray-400 hover:text-cyan-400">
                  Platform
                </Link>
              </li>
              <li>
                <Link href="/waves" className="text-sm text-gray-400 hover:text-cyan-400">
                  WAVES™
                </Link>
              </li>
              <li>
                <Link href="/governance" className="text-sm text-gray-400 hover:text-cyan-400">
                  Governance
                </Link>
              </li>
            </ul>
          </div>

          {/* Company & Resources */}
          <div>
            <h3 className="text-sm font-semibold text-white">Company & Resources</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <Link href="/enterprise" className="text-sm text-gray-400 hover:text-cyan-400">
                  Enterprise
                </Link>
              </li>
              <li>
                <Link href="/company" className="text-sm text-gray-400 hover:text-cyan-400">
                  Company
                </Link>
              </li>
              <li>
                <Link href="/market" className="text-sm text-gray-400 hover:text-cyan-400">
                  Market
                </Link>
              </li>
              <li>
                <Link href="/contact" className="text-sm text-gray-400 hover:text-cyan-400">
                  Contact
                </Link>
              </li>
              <li>
                <Link href="/demo" className="text-sm text-gray-400 hover:text-cyan-400">
                  Request Demo
                </Link>
              </li>
              <li>
                <Link href="/briefing" className="text-sm text-gray-400 hover:text-cyan-400">
                  Executive Briefing
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 border-t border-gray-800 pt-8">
          <p className="text-center text-sm text-gray-400 mb-2">
            Analytics platform. Not investment advice.
          </p>
          <p className="text-center text-sm text-gray-400">
            &copy; {currentYear} WAVES Intelligence. All rights reserved.
          </p>
          <p className="text-center text-xs text-gray-500 mt-2 font-mono">Build: {buildId}</p>
        </div>
      </div>
    </footer>
  );
}
