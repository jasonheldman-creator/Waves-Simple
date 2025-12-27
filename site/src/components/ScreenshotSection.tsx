import Image from "next/image";

interface Screenshot {
  imagePath: string;
  alt: string;
  title: string;
  bullets: string[];
}

interface ScreenshotSectionProps {
  screenshot: Screenshot;
  reverse?: boolean;
}

export default function ScreenshotSection({ screenshot, reverse = false }: ScreenshotSectionProps) {
  // Map old paths to new SVG paths - using exact matching for robustness
  const getImagePath = (path: string): string => {
    const pathMap: Record<string, string> = {
      '/public/screens/product-screenshot1.png': '/screens/portfolio-dashboard.svg',
      '/public/screens/product-screenshot2.png': '/screens/attribution-analysis.svg',
      '/public/screens/product-screenshot3.png': '/screens/governance-console.svg',
    };
    
    return pathMap[path] || path;
  };

  const imagePath = getImagePath(screenshot.imagePath);

  return (
    <section className="bg-gradient-to-b from-gray-900 to-black py-16 sm:py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className={`grid grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-12 items-center ${reverse ? 'lg:flex-row-reverse' : ''}`}>
          {/* Screenshot */}
          <div className={`${reverse ? 'lg:order-2' : ''}`}>
            <div className="relative rounded-lg border border-cyan-500/20 bg-gray-900/50 p-2 shadow-lg shadow-cyan-500/10">
              <div className="relative aspect-video w-full bg-gray-800 rounded overflow-hidden">
                <Image 
                  src={imagePath} 
                  alt={screenshot.alt} 
                  fill
                  className="object-contain"
                />
              </div>
            </div>
          </div>

          {/* Content */}
          <div className={`${reverse ? 'lg:order-1' : ''}`}>
            <h3 className="text-2xl font-bold text-white sm:text-3xl mb-6">
              {screenshot.title}
            </h3>
            <ul className="space-y-4">
              {screenshot.bullets.map((bullet, index) => (
                <li key={index} className="flex items-start">
                  <span className="flex-shrink-0 flex h-6 w-6 items-center justify-center rounded-full bg-cyan-500/10 text-cyan-400 mr-3 mt-0.5">
                    <span className="text-sm">•</span>
                  </span>
                  <span className="text-gray-300 leading-relaxed">{bullet}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
