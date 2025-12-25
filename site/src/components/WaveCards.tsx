"use client";

import React, { useEffect, useMemo, useState } from "react";
import WaveCard from "../WaveCard";

type LiveWaveData = {
  wave_id: string;
  wave_name: string;
  status: string; // LIVE / DEMO
  performance_1d: string;
  performance_30d: string;
  performance_ytd: string;
  last_updated: string;
};

type CardModel = {
  title: string;
  description: string;
};

const PREVIEW_FALLBACK_URL =
  "https://waves-simple-git-copilot-70ae28-jasonheldman-creators-projects.vercel.app/api/live_snapshot.csv";

// Quote-safe CSV line parser (handles commas inside quotes)
function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (ch === '"') {
      // handle escaped quotes ""
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === "," && !inQuotes) {
      out.push(cur.trim());
      cur = "";
      continue;
    }

    cur += ch;
  }

  out.push(cur.trim());
  return out;
}

function parseLiveSnapshotCsv(csvText: string): LiveWaveData[] {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  // Expect header:
  // wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated
  return lines.slice(1).map((line) => {
    const v = parseCsvLine(line);

    return {
      wave_id: v[0] ?? "",
      wave_name: (v[1] ?? "").replace(/^"|"$/g, ""),
      status: v[2] ?? "",
      performance_1d: v[3] ?? "",
      performance_30d: v[4] ?? "",
      performance_ytd: v[5] ?? "",
      last_updated: v[6] ?? "",
    };
  });
}

export default function WaveCards() {
  // Your existing static cards remain the base UI structure
  const staticCards: CardModel[] = useMemo(
    () => [
      {
        title: "Large Cap Growth",
        description:
          "This wave primarily emphasizes stocks with a market cap in the $10 billion+ range. Most are in the growth sectors.",
      },
      {
        title: "Small–Mid Cap Blend",
        description:
          "Focuses on potential gainers in the $1–10 billion range. Looks at both growth and value characteristics.",
      },
      {
        title: "Non-U.S. International",
        description:
          "Diversifies assets outside the U.S. by adding high-performing international stocks. Average yield: 40–60%.",
      },
    ],
    []
  );

  const [liveData, setLiveData] = useState<LiveWaveData[] | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchCsvFrom = async (url: string) => {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`Fetch failed: ${r.status}`);
      return r.text();
    };

    const fetchLive = async () => {
      const envUrl = process.env.NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL;

      const sources: Array<{ label: string; url: string | undefined }> = [
        { label: "ENV", url: envUrl },
        { label: "LOCAL_API", url: "/api/live_snapshot.csv" },
        { label: "PREVIEW_FALLBACK", url: PREVIEW_FALLBACK_URL },
      ];

      for (const s of sources) {
        if (!s.url) continue;
        try {
          const csvText = await fetchCsvFrom(s.url);
          const parsed = parseLiveSnapshotCsv(csvText);
          if (parsed.length > 0) {
            if (!cancelled) setLiveData(parsed);
            return;
          }
        } catch (e) {
          // keep trying next source
          console.warn(`Live snapshot source failed: ${s.label}`, e);
        }
      }

      if (!cancelled) setLiveData([]); // indicates tried but empty
    };

    fetchLive();
    const id = setInterval(fetchLive, 60000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // If we have liveData, append a short live metrics line to the description
  // without changing the WaveCard component structure.
  const cardsToRender: CardModel[] = useMemo(() => {
    if (!liveData || liveData.length === 0) return staticCards;

    // Create a quick lookup by normalized name
    const byName = new Map(
      liveData.map((d) => [d.wave_name.toLowerCase(), d] as const)
    );

    return staticCards.map((c) => {
      const match = byName.get(c.title.toLowerCase());
      if (!match) return c;

      const liveLine = `Live (1D ${match.performance_1d} | 30D ${match.performance_30d} | YTD ${match.performance_ytd}) — ${match.status}`;

      return {
        ...c,
        description: `${c.description} ${liveLine}`,
      };
    });
  }, [staticCards, liveData]);

  return (
    <div>
      {cardsToRender.map((c) => (
        <WaveCard key={c.title} title={c.title} description={c.description} />
      ))}
    </div>
  );
}