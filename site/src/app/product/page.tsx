import Hero from "@/components/Hero";
import ScreenshotSection from "@/components/ScreenshotSection";
import { siteContent } from "@/content/newSiteContent";

export default function ProductPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero title={siteContent.product.hero.title} subtitle={siteContent.product.hero.subtitle} />

      {siteContent.product.screenshots.map((screenshot, index) => (
        <ScreenshotSection key={index} screenshot={screenshot} reverse={index % 2 === 1} />
      ))}
    </main>
  );
}
