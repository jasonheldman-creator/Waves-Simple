"use client";

import { useEffect, useState } from "react";

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

interface WaveCardsProps {
  waves?: WaveCard[];
}

interface LiveWaveData {
  wave_id: string;
  wave_name: string;
  status: string;
  performance_1d: string;
  performance_30d: string;
  performance_ytd: string;
  last_updated: string;
}

export default function WaveCards({ waves }: WaveCardsProps) {
  const [liveData, setLiveData] = useState<LiveWaveData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch live snapshot data from CSV endpoint
  useEffect(() => {
    const fetchLiveData = async () => {
      try {
        const csvUrl = process.env.NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL || "/api/live_snapshot.csv";
        const response = await fetch(csvUrl);
        const csvText = await response.text();
        
        // Parse CSV data
        const lines = csvText.trim().split("\n");
        
        const parsedData: LiveWaveData[] = lines.slice(1).map((line) => {
          const values = line.split(",");
          return {
            wave_id: values[0],
            wave_name: values[1],
            status: values[2],
            performance_1d: values[3],
            performance_30d: values[4],
            performance_ytd: values[5],
            last_updated: values[6],
          };
        });
        
        setLiveData(parsedData);
        setIsLoading(false);
      } catch (error) {
        console.error("Failed to fetch live snapshot data:", error);
        setIsLoading(false);
      }
    };

    // Initial fetch
    fetchLiveData();

    // Set up 60-second interval
    const intervalId = setInterval(fetchLiveData, 60000);

    // Cleanup on unmount
    return () => clearInterval(intervalId);
  }, []);

  // Convert live data to display waves - show ALL waves from CSV
  const displayWaves: WaveCard[] = liveData.map((live, index) => {
    return {
      id: index + 1,
      name: live.wave_name,
      description: `Wave ID: ${live.wave_id}`,
      performance: live.performance_1d,
      performance1d: live.performance_1d,
      performance30d: live.performance_30d,
      performanceYtd: live.performance_ytd,
      status: live.status,
    };
  });

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
          {isLoading && (
            <p className="mt-2 text-sm text-gray-500">
              Loading wave data...
            </p>
          )}
        </div>

        {displayWaves.length === 0 && !isLoading ? (
          <div className="text-center text-gray-400">
            <p>No waves available at the moment.</p>
          </div>
        ) : (
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
                        wave.status === "DEMO"
                          ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
                          : "bg-green-500/20 text-green-400"
                      }`}
                    >
                      {wave.status}
                    </span>
                  </div>
                  {wave.performance && wave.performance !== "--" && (
                    <div className="text-right">
                      <div className={`text-lg font-bold ${
                        wave.performance.startsWith("-") ? "text-red-400" : "text-green-400"
                      }`}>{wave.performance}</div>
                      <div className="text-xs text-gray-500">1D</div>
                      {wave.performance30d && wave.performance30d !== "--" && (
                        <>
                          <div className={`text-sm font-semibold mt-1 ${
                            wave.performance30d.startsWith("-") ? "text-red-400" : "text-green-400"
                          }`}>{wave.performance30d}</div>
                          <div className="text-xs text-gray-500">30D</div>
                        </>
                      )}
                      {wave.performanceYtd && wave.performanceYtd !== "--" && (
                        <>
                          <div className={`text-sm font-semibold mt-1 ${
                            wave.performanceYtd.startsWith("-") ? "text-red-400" : "text-green-400"
                          }`}>{wave.performanceYtd}</div>
                          <div className="text-xs text-gray-500">YTD</div>
                        </>
                      )}
                    </div>
                  )}
                </div>
                <p className="mt-4 text-sm text-gray-400">{wave.description}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
