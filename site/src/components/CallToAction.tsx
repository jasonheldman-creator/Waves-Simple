import Link from "next/link";

interface CallToActionProps {
  title: string;
  description: string;
  primaryButtonText: string;
  primaryButtonLink: string;
  secondaryButtonText?: string;
  secondaryButtonLink?: string;
}

export default function CallToAction({
  title,
  description,
  primaryButtonText,
  primaryButtonLink,
  secondaryButtonText,
  secondaryButtonLink,
}: CallToActionProps) {
  return (
    <section className="bg-gradient-to-r from-cyan-900/20 to-green-900/20 py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="relative overflow-hidden rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-gray-900 to-black p-8 shadow-2xl sm:p-12 lg:p-16">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-green-500/5" />
          <div className="relative">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
                {title}
              </h2>
              <p className="mx-auto mt-6 max-w-2xl text-lg text-gray-300 sm:text-xl">
                {description}
              </p>
              <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:justify-center">
                <Link
                  href={primaryButtonLink}
                  className="rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-3 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400"
                >
                  {primaryButtonText}
                </Link>
                {secondaryButtonText && secondaryButtonLink && (
                  <Link
                    href={secondaryButtonLink}
                    className="rounded-md border border-cyan-500 px-8 py-3 text-base font-semibold text-cyan-400 transition-all hover:bg-cyan-500/10"
                  >
                    {secondaryButtonText}
                  </Link>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
