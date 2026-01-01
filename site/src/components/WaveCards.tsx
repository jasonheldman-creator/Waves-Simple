"use client";

import { useState, useEffect } from "react";
import { WAVE_REGISTRY } from "@/lib/waveRegistry";

interface WaveCardData {
  wave_id: string;
  wave_name: string;
  status: string;
  performance_1d: string;
  performance_30d: string;
  performance_ytd: string;
  last_updated: string;
}

/**
 * Simple CSV parser without external dependencies
 */
function parseCSV(csvText: string): WaveCardData[] {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return [];
  
  const data: WaveCardData[] = [];
  
  // Skip header (line 0) and parse data rows
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    // Simple CSV parsing - handles quoted fields
    const values: string[] = [];
    let currentValue = '';
    let insideQuotes = false;
    
    for (let j = 0; j < line.length; j++) {
      const char = line[j];
      
      if (char === '"') {
        insideQuotes = !insideQuotes;
      } else if (char === ',' && !insideQuotes) {
        values.push(currentValue);
        currentValue = '';
      } else {
        currentValue += char;
      }
    }
    values.push(currentValue); // Push the last value
    
    if (values.length >= 7) {
      data.push({
        wave_id: values[0],
        wave_name: values[1],
        status: values[2],
        performance_1d: values[3],
        performance_30d: values[4],
        performance_ytd: values[5],
        last_updated: values[6],
      });
    }
  }
  
  return data;
}

/**
 * WaveCards Component
 * 
 * Displays all investment waves with live performance metrics
 * Fetches data client-side after mount to avoid build-time dependencies
 */
export default function WaveCards() {
  const [waveData, setWaveData] = useState<Map<string, WaveCardData>>(new Map());
  const [dataSource, setDataSource] = useState<string>("loading");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    async function fetchWaveData() {
      // Fetch priority:
      // 1. NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL (if set)
      // 2. /api/live_snapshot.csv
      // 3. Preview fallback (render registry with no metrics)
      
      const externalUrl = process.env.NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL;
      const urls = externalUrl 
        ? [externalUrl, "/api/live_snapshot.csv"]
        : ["/api/live_snapshot.csv"];

      for (const url of urls) {
        try {
          console.log(`[WaveCards] Attempting to fetch from: ${url}`);
          const response = await fetch(url, {
            cache: 'no-store',
          });

          if (!response.ok) {
            console.warn(`[WaveCards] Fetch failed from ${url}: ${response.status}`);
            continue;
          }

          const csvText = await response.text();
          const parsed = parseCSV(csvText);
          
          if (!isMounted) return;

          // Create map of wave_id -> data
          const dataMap = new Map<string, WaveCardData>();
          for (const wave of parsed) {
            dataMap.set(wave.wave_id, wave);
          }

          setWaveData(dataMap);
          setDataSource(url === externalUrl ? "external" : "api");
          
          console.log(`[WaveCards] Successfully loaded ${parsed.length} waves from ${url}`);
          
          // Invariant check: warn if parsed rows < registry rows
          if (parsed.length < WAVE_REGISTRY.length) {
            const parsedIds = new Set(parsed.map(w => w.wave_id));
            const missingWaves = WAVE_REGISTRY
              .filter(w => !parsedIds.has(w.wave_id))
              .map(w => w.wave_id);
            
            console.warn(
              `[WaveCards] WARNING: Parsed ${parsed.length} rows but registry has ${WAVE_REGISTRY.length} waves.`,
              `Missing wave_ids:`, missingWaves
            );
          }
          
          setIsLoading(false);
          return;
          
        } catch (error) {
          console.error(`[WaveCards] Error fetching from ${url}:`, error);
        }
      }

      // All fetches failed - use preview fallback
      if (!isMounted) return;
      
      console.log("[WaveCards] All fetch attempts failed. Using PREVIEW_FALLBACK.");
      setDataSource("preview_fallback");
      setIsLoading(false);
    }

    fetchWaveData();

    return () => {
      isMounted = false;
    };
  }, []);

  // Render all waves from canonical registry
  const wavesToRender = WAVE_REGISTRY;

  return (
    <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
            Investment Waves Portfolio
          </h2>
          <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
            Strategic themes capturing alpha across market conditions
          </p>
          {!isLoading && (
            <p className="mt-2 text-xs text-gray-500">
              Data source: {dataSource} • {wavesToRender.length} waves
            </p>
          )}
        </div>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {wavesToRender.map((registryWave) => {
            const liveData = waveData.get(registryWave.wave_id);
            const displayName = liveData?.wave_name || registryWave.display_name;
            const status = liveData?.status || "–";
            const perf1d = liveData?.performance_1d || "–";
            const perf30d = liveData?.performance_30d || "–";
            const perfYtd = liveData?.performance_ytd || "–";

            // Determine card color based on status
            const isLive = status === "LIVE";
            const borderColor = isLive ? "border-cyan-500/30" : "border-gray-700/50";
            const statusColor = isLive ? "text-cyan-400" : "text-gray-500";

            return (
              <div
                key={registryWave.wave_id}
                className={`rounded-lg border ${borderColor} bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 transition-all hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10`}
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white leading-tight">
                    {displayName}
                  </h3>
                  <span className={`text-xs font-medium ${statusColor} px-2 py-1 rounded border ${borderColor}`}>
                    {status}
                  </span>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">1-Day</span>
                    <span className={`text-sm font-medium ${getPerformanceColor(perf1d)}`}>
                      {formatPerformance(perf1d)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">30-Day</span>
                    <span className={`text-sm font-medium ${getPerformanceColor(perf30d)}`}>
                      {formatPerformance(perf30d)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">YTD</span>
                    <span className={`text-sm font-medium ${getPerformanceColor(perfYtd)}`}>
                      {formatPerformance(perfYtd)}
                    </span>
                  </div>
                </div>

                {liveData?.last_updated && (
                  <div className="mt-4 pt-4 border-t border-gray-700">
                    <span className="text-xs text-gray-500">
                      Updated: {new Date(liveData.last_updated).toLocaleDateString()}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function formatPerformance(value: string): string {
  if (value === "–" || value === "") return "–";
  const num = parseFloat(value);
  if (isNaN(num)) return value;
  return num >= 0 ? `+${num.toFixed(2)}%` : `${num.toFixed(2)}%`;
}

function getPerformanceColor(value: string): string {
  if (value === "–" || value === "") return "text-gray-500";
  const num = parseFloat(value);
  if (isNaN(num)) return "text-gray-500";
  if (num > 0) return "text-green-400";
  if (num < 0) return "text-red-400";
  return "text-gray-400";
}