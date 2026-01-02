"use client";

import { useEffect, useMemo, useState } from "react";

interface WaveCard {
  id: number;
  name: string;
  description: string;
  performance?: string;
  status?: string;
  performance1d?: string;
  performance30d?: string;
  performanceYtd?: string;
}

interface LiveWaveData {
  wave_id: string;
  wave_name: string;
  status: string;
  performance_1d: string;
  performance_30d: string;
  performance_ytd: string;
  alpha_1d: string;
  alpha_30d: string;
  alpha_ytd: string;
  last_updated: string;
}

/**
 * Robust CSV line parser:
 * - Handles quoted fields with commas
 * - Handles escaped quotes inside quotes: "" -> "
 */
function parseCSVLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (ch === '"') {
      // If we are inside quotes and the next char is also a quote, it's an escaped quote
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++; // skip next quote
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }

    cur += ch;
  }

  out.push(cur);
  return out.map((v) => v.trim());
}

function normalizeDash(value: string | undefined | null) {
  const v = (value ?? "").trim();
  return v === "" ? "—" : v;
}

export default function WaveCards() {
  const [liveData, setLiveData] = useState<LiveWaveData[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const fetchLiveData = async () => {
      try {
        setIsLoading(true);

        const url = `/data/live_snapshot.csv?ts=${Date.now()}`;
        const response = await fetch(url, { cache: "no-store" });

        if (!response.ok) {
          throw new Error(`CSV fetch failed: ${response.status}`);
        }

        const csvText = await response.text();
        const lines = csvText.split(/\r?\n/).filter((l) => l.trim() !== "");

        if (lines.length < 2) {
          throw new Error("CSV file is empty or missing rows");
        }

        const rows = lines.slice(1);

        const parsed: LiveWaveData[] = rows
          .map((line) => {
            const values = parseCSVLine(line);

            return {
              wave_id: values[0] ?? "",
              wave_name: values[1] ?? "",
              status: values[2] ?? "",
              performance_1d: values[3] ?? "",
              performance_30d: values[4] ?? "",
              performance_ytd: values[5] ?? "",
              alpha_1d: values[6] ?? "",
              alpha_30d: values[7] ?? "",
              alpha_ytd: values[8] ?? "",
              last_updated: values[9] ?? "",
            };
          })
          // remove any accidental blank row
          .filter((r) => r.wave_id.trim() !== "" || r.wave_name.trim() !== "");

        if (parsed.length === 0) {
          throw new Error("CSV parsed to 0 rows");
        }

        // De-dupe by wave_id (just in case)
        const seen = new Set<string>();
        const deduped = parsed.filter((r) => {
          const key = r.wave_id.trim() || r.wave_name.trim();
          if (!key) return false;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });

        if (!cancelled) {
          setLiveData(deduped);
          setIsLoading(false);
        }
      } catch (err) {
        console.error("Failed to fetch live snapshot CSV:", err);
        if (!cancelled) {
          setLiveData([]);
          setIsLoading(false);
        }
      }
    };

    fetchLiveData();
    const intervalId = setInterval(fetchLiveData, 60000);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, []);

  // Render waves from CSV data
  const displayWaves: WaveCard[] = useMemo(() => {
    if (!liveData || liveData.length === 0) {
      return [];
    }

    return liveData.map((row, idx) => ({
      id: idx + 1,
      name: row.wave_name || row.wave_id,
      description: `Wave ID: ${row.wave_id}`,
      performance: normalizeDash(row.performance_1d),
      performance1d: normalizeDash(row.performance_1d),
      performance30d: normalizeDash(row.performance_30d),
      performanceYtd: normalizeDash(row.performance_ytd),
      status: row.status || "LIVE",
    }));
  }, [liveData]);

  // Calculate top 5 and bottom 5 waves by performance_30d for charts
  const chartData = useMemo(() => {
    if (!liveData || liveData.length === 0) {
      return { top5: [], bottom5: [] };
    }

    // Filter waves with valid performance_30d values
    const wavesWithPerf = liveData
      .map((w) => {
        const perfStr = (w.performance_30d || "").trim();
        if (perfStr === "" || perfStr === "—") {
          return { name: w.wave_name, value: null };
        }
        const numValue = parseFloat(perfStr.replace("%", ""));
        return {
          name: w.wave_name,
          value: !isNaN(numValue) ? numValue : null,
        };
      })
      .filter((w) => w.value !== null);

    // Sort by performance
    const sorted = [...wavesWithPerf].sort((a, b) => (b.value as number) - (a.value as number));

    return {
      top5: sorted.slice(0, 5),
      bottom5: sorted.slice(-5).reverse(),
    };
  }, [liveData]);

  // Helper function to render a performance chart
  const renderPerformanceChart = (
    waves: Array<{ name: string; value: number | null }>,
    title: string,
    borderColor: string
  ) => {
    if (waves.length === 0) return null;

    const maxValue = Math.max(...waves.map((w) => Math.abs(w.value as number)));

    return (
      <div className={`rounded-lg border ${borderColor} bg-gray-900/50 p-6`}>
        <h3 className="text-xl font-semibold text-white mb-4">{title}</h3>
        <div className="space-y-3">
          {waves.map((wave, idx) => {
            const barWidth = (Math.abs(wave.value as number) / maxValue) * 100;
            const isPositive = (wave.value as number) >= 0;

            return (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-300 truncate">{wave.name}</span>
                  <span className={isPositive ? "text-green-400" : "text-red-400"}>
                    {wave.value?.toFixed(2)}%
                  </span>
                </div>
                <div className="h-6 bg-gray-800 rounded overflow-hidden">
                  <div
                    className={`h-full ${isPositive ? "bg-green-500" : "bg-red-500"}`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">
            Investment <span className="text-cyan-400">Waves</span>
          </h2>
          <p className="mt-4 text-lg text-gray-400">
            Explore our portfolio of strategic investment waves
          </p>

          {isLoading && <p className="mt-2 text-sm text-gray-500">Loading wave data...</p>}
        </div>

        {/* Performance Charts */}
        {!isLoading && chartData.top5.length > 0 && (
          <div className="mb-12 grid grid-cols-1 gap-8 lg:grid-cols-2">
            {renderPerformanceChart(
              chartData.top5,
              "Top 5 Waves by 30-Day Performance",
              "border-green-500/20"
            )}
            {renderPerformanceChart(
              chartData.bottom5,
              "Bottom 5 Waves by 30-Day Performance",
              "border-red-500/20"
            )}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {displayWaves.map((wave) => (
            <div
              key={wave.id}
              className="group rounded-lg border border-gray-800 bg-gray-900/30 p-6 backdrop-blur-sm transition-all hover:border-cyan-500/50 hover:bg-gray-900/50"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400">
                    {wave.name}
                  </h3>

                  <span
                    className={`mt-1 inline-block rounded-full px-2 py-1 text-xs font-medium ${
                      wave.status === "LIVE"
                        ? "bg-green-500/20 text-green-400 border border-green-500/30"
                        : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
                    }`}
                  >
                    {wave.status}
                  </span>
                </div>

                {/* Always show the metrics block (even if it's "—") */}
                <div className="text-right">
                  <div
                    className={`text-lg font-bold ${
                      wave.performance && wave.performance !== "—" && wave.performance.startsWith("-")
                        ? "text-red-400"
                        : wave.performance && wave.performance !== "—"
                        ? "text-green-400"
                        : "text-gray-400"
                    }`}
                  >
                    {wave.performance ?? "—"}
                  </div>
                  <div className="text-xs text-gray-500">1D</div>

                  <div
                    className={`text-sm font-semibold mt-1 ${
                      wave.performance30d && wave.performance30d !== "—" && wave.performance30d.startsWith("-")
                        ? "text-red-400"
                        : wave.performance30d && wave.performance30d !== "—"
                        ? "text-green-400"
                        : "text-gray-400"
                    }`}
                  >
                    {wave.performance30d ?? "—"}
                  </div>
                  <div className="text-xs text-gray-500">30D</div>

                  <div
                    className={`text-sm font-semibold mt-1 ${
                      wave.performanceYtd && wave.performanceYtd !== "—" && wave.performanceYtd.startsWith("-")
                        ? "text-red-400"
                        : wave.performanceYtd && wave.performanceYtd !== "—"
                        ? "text-green-400"
                        : "text-gray-400"
                    }`}
                  >
                    {wave.performanceYtd ?? "—"}
                  </div>
                  <div className="text-xs text-gray-500">YTD</div>
                </div>
              </div>

              <p className="mt-4 text-sm text-gray-400">{wave.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}