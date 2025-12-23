import { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  const baseUrl = "https://www.wavesintelligence.app";

  return {
    rules: {
      userAgent: "*",
      allow: "/",
      disallow: ["/api/", "/console/"],
    },
    sitemap: `${baseUrl}/sitemap.xml`,
  };
}
