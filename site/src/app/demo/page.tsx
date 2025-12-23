import Hero from "@/components/Hero";
import DemoForm from "@/components/DemoForm";
import { siteContent } from "@/content/newSiteContent";

export default function DemoPage() {
  return (
    <main className="min-h-screen bg-black">
      <Hero
        title={siteContent.demo.hero.title}
        subtitle={siteContent.demo.hero.subtitle}
      />
      
      <DemoForm
        title={siteContent.demo.form.title}
        subtitle={siteContent.demo.form.subtitle}
      />
    </main>
  );
}
