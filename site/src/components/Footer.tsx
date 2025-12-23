import Link from "next/link";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-gray-800 bg-black">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          {/* Brand */}
          <div className="col-span-1 md:col-span-1">
            <div className="text-xl font-bold">
              <span className="text-cyan-400">WAVES</span>
              <span className="text-white"> Intelligence</span>
            </div>
            <p className="mt-4 text-sm text-gray-400">
              Institutional-grade portfolio analytics and decision intelligence.
            </p>
          </div>

          {/* Product */}
          <div>
            <h3 className="text-sm font-semibold text-white">Product</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <Link href="/platform" className="text-sm text-gray-400 hover:text-cyan-400">
                  Platform
                </Link>
              </li>
              <li>
                <Link href="/waves" className="text-sm text-gray-400 hover:text-cyan-400">
                  Waves
                </Link>
              </li>
              <li>
                <Link href="/architecture" className="text-sm text-gray-400 hover:text-cyan-400">
                  Architecture
                </Link>
              </li>
              <li>
                <Link href="/security" className="text-sm text-gray-400 hover:text-cyan-400">
                  Security
                </Link>
              </li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h3 className="text-sm font-semibold text-white">Company</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <Link href="/company" className="text-sm text-gray-400 hover:text-cyan-400">
                  About
                </Link>
              </li>
              <li>
                <Link href="/press" className="text-sm text-gray-400 hover:text-cyan-400">
                  Press
                </Link>
              </li>
              <li>
                <Link href="/contact" className="text-sm text-gray-400 hover:text-cyan-400">
                  Contact
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className="text-sm font-semibold text-white">Legal</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="#" className="text-sm text-gray-400 hover:text-cyan-400">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-400 hover:text-cyan-400">
                  Terms of Service
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 border-t border-gray-800 pt-8">
          <p className="text-center text-sm text-gray-400">
            &copy; {currentYear} WAVES Intelligence. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
