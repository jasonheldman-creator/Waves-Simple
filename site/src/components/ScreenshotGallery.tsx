import Image from "next/image";

interface Screenshot {
  src: string;
  alt: string;
  caption: string;
}

interface ScreenshotGalleryProps {
  screenshots?: Screenshot[];
}

export default function ScreenshotGallery({ screenshots }: ScreenshotGalleryProps) {
  // Default branded screenshots
  const defaultScreenshots: Screenshot[] = [
    {
      src: "/images/dashboard.svg",
      alt: "Dashboard Overview",
      caption: "Comprehensive portfolio analytics dashboard",
    },
    {
      src: "/images/analytics.svg",
      alt: "Analytics View",
      caption: "Real-time performance metrics and insights",
    },
    {
      src: "/images/reports.svg",
      alt: "Reports",
      caption: "Institutional-grade reporting and attribution",
    },
  ];

  const displayScreenshots = screenshots || defaultScreenshots;

  return (
    <section className="bg-black py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">
            Platform <span className="text-cyan-400">Screenshots</span>
          </h2>
          <p className="mt-4 text-lg text-gray-400">
            Explore the WAVES Intelligence platform interface
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
          {displayScreenshots.map((screenshot, index) => (
            <div
              key={index}
              className="group overflow-hidden rounded-lg border border-gray-800 bg-gray-900/50 transition-all hover:border-cyan-500/50"
            >
              <div className="relative aspect-video w-full overflow-hidden bg-gray-800">
                <Image
                  src={screenshot.src}
                  alt={screenshot.alt}
                  fill
                  className="object-cover transition-transform group-hover:scale-105"
                />
              </div>
              <div className="p-4">
                <h3 className="font-semibold text-white group-hover:text-cyan-400 transition-colors">
                  {screenshot.alt}
                </h3>
                <p className="mt-1 text-sm text-gray-400">{screenshot.caption}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
