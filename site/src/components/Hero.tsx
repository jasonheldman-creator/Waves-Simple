import Link from "next/link";

interface HeroProps {
  title: string;
  subtitle: string;
  ctaText?: string;
  ctaLink?: string;
  secondaryCtaText?: string;
  secondaryCtaLink?: string;
}

export default function Hero({
  title,
  subtitle,
  ctaText,
  ctaLink,
  secondaryCtaText,
  secondaryCtaLink,
}: HeroProps) {
  return (
    <section className="relative overflow-hidden bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-32">
      {/* Animated background effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-green-500/10" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,255,255,0.1),transparent_50%)]" />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl md:text-6xl lg:text-7xl">
            {title.split(" ").map((word, i) => {
              // Highlight certain keywords with gradient
              const highlightWords = ["WAVES", "Intelligence", "Platform", "Security", "Architecture"];
              if (highlightWords.some((hw) => word.includes(hw))) {
                return (
                  <span
                    key={i}
                    className="bg-gradient-to-r from-cyan-400 to-green-400 bg-clip-text text-transparent"
                  >
                    {word}{" "}
                  </span>
                );
              }
              return <span key={i}>{word} </span>;
            })}
          </h1>
          <p className="mx-auto mt-6 max-w-3xl text-lg text-gray-300 sm:text-xl md:text-2xl">
            {subtitle}
          </p>

          {(ctaText || secondaryCtaText) && (
            <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:justify-center">
              {ctaText && ctaLink && (
                <Link
                  href={ctaLink}
                  className="rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-3 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400"
                >
                  {ctaText}
                </Link>
              )}
              {secondaryCtaText && secondaryCtaLink && (
                <Link
                  href={secondaryCtaLink}
                  className="rounded-md border border-cyan-500 px-8 py-3 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10"
                >
                  {secondaryCtaText}
                </Link>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
