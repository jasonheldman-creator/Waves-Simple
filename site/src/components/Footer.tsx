import Link from "next/link";

export default function Footer() {
  const currentYear = new Date().getFullYear();

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

          {/* Product */}
          <div>
            <h3 className="text-sm font-semibold text-white">Product</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <Link href="/product" className="text-sm text-gray-400 hover:text-cyan-400">
                  Platform Screens
                </Link>
              </li>
              <li>
                <Link href="/why" className="text-sm text-gray-400 hover:text-cyan-400">
                  Why WAVES
                </Link>
              </li>
              <li>
                <Link href="/demo" className="text-sm text-gray-400 hover:text-cyan-400">
                  Request Demo
                </Link>
              </li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h3 className="text-sm font-semibold text-white">Company</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <Link href="/" className="text-sm text-gray-400 hover:text-cyan-400">
                  Home
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
        </div>
      </div>
    </footer>
  );
}
