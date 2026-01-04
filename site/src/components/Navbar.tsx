"use client";

import Link from "next/link";
import { useState, useRef, useEffect } from "react";

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [desktopMenuOpen, setDesktopMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const navLinks = [
    { href: "/", label: "Home" },
    { href: "/platform", label: "Platform" },
    { href: "/waves", label: "WAVES™" },
    { href: "/governance", label: "Governance" },
    { href: "/enterprise", label: "Enterprise" },
    { href: "/company", label: "Company" },
    { href: "/market", label: "Market" },
    { href: "/contact", label: "Contact" },
  ];

  // Close desktop menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setDesktopMenuOpen(false);
      }
    }

    if (desktopMenuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [desktopMenuOpen]);

  return (
    <nav className="sticky top-0 z-50 border-b border-gray-800 bg-black/95 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <div className="text-2xl font-bold">
              <span className="text-cyan-400">WAVES</span>
              <span className="text-white"> Intelligence™</span>
            </div>
          </Link>

          {/* Desktop Navigation with Dropdown */}
          <div className="hidden md:flex md:items-center md:space-x-4">
            {/* Dropdown Menu Button */}
            <div className="relative" ref={menuRef}>
              <button
                onClick={() => setDesktopMenuOpen(!desktopMenuOpen)}
                className="flex items-center space-x-2 rounded-md border border-gray-700 px-4 py-2 text-sm font-medium text-gray-300 transition-colors hover:border-cyan-500 hover:text-cyan-400"
                aria-label="Toggle navigation menu"
              >
                <span>Navigation</span>
                <svg
                  className={`h-4 w-4 transition-transform ${desktopMenuOpen ? "rotate-180" : ""}`}
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* Dropdown Menu */}
              {desktopMenuOpen && (
                <div className="absolute right-0 mt-2 w-56 origin-top-right rounded-md border border-gray-800 bg-gray-900/95 backdrop-blur-sm shadow-xl">
                  <div className="py-2">
                    {navLinks.map((link) => (
                      <Link
                        key={link.href}
                        href={link.href}
                        className="block px-4 py-2 text-sm font-medium text-gray-300 transition-colors hover:bg-gray-800 hover:text-cyan-400"
                        onClick={() => setDesktopMenuOpen(false)}
                      >
                        {link.label}
                      </Link>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <Link
              href="/demo"
              className="rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-4 py-2 text-sm font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400"
            >
              Request Institutional Demo
            </Link>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden rounded-md p-2 text-gray-400 hover:bg-gray-800 hover:text-white"
            aria-label="Toggle menu"
          >
            <svg
              className="h-6 w-6"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              {mobileMenuOpen ? (
                <path d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-gray-800">
          <div className="space-y-1 px-4 pb-3 pt-2">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="block rounded-md px-3 py-2 text-base font-medium text-gray-300 hover:bg-gray-800 hover:text-cyan-400"
                onClick={() => setMobileMenuOpen(false)}
              >
                {link.label}
              </Link>
            ))}
            <Link
              href="/demo"
              className="block rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-3 py-2 text-base font-semibold text-black"
              onClick={() => setMobileMenuOpen(false)}
            >
              Request Institutional Demo
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}
