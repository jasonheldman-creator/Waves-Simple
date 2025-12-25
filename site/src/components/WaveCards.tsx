"use client";

import React, { useEffect, useState } from "react";

interface WaveData {
  wave_id: string;
  wave_name: string;
  status: "LIVE" | "DEMO";
  performance_1d: number;
  performance_30d: number;
  performance_ytd: number;
  last_updated: string;
}

type DataSource = "external" | "internal" | "demo";

// Helper function to parse CSV data
const parseCSV = (csvText: string): WaveData[] => {
  const lines = csvText.trim().split("\n");
  if (lines.length < 2) return [];

  return lines
    .slice(1)
    .map((line) => {
      // Handle quoted fields that might contain commas
      const regex = /,(?=(?:[^"]*"[^"]*")*[^"]*$)/;
      const values = line.split(regex).map((v) => v.replace(/^"|"$/g, "").trim());

      if (values.length < 7) return null;

      return {
        wave_id: values[0],
        wave_name: values[1],
        status: values[2] === "LIVE" ? "LIVE" : "DEMO",
        performance_1d: parseFloat(values[3]),
        performance_30d: parseFloat(values[4]),
        performance_ytd: parseFloat(values[5]),
        last_updated: values[6],
      };
    })
    .filter((wave): wave is WaveData => wave !== null);
};

// Helper function to get deterministic demo data
const getDemoWaves = (): WaveData[] => {
  const now = new Date().toISOString();
  return [
    {
      wave_id: "core_equity_wave",
      wave_name: "Core Equity Wave",
      status: "DEMO",
      performance_1d: 0.45,
      performance_30d: 2.31,
      performance_ytd: 12.45,
      last_updated: now,
    },
    {
      wave_id: "growth_alpha_wave",
      wave_name: "Growth Alpha Wave",
      status: "DEMO",
      performance_1d: 0.82,
      performance_30d: 4.56,
      performance_ytd: 18.92,
      last_updated: now,
    },
    {
      wave_id: "value_recovery_wave",
      wave_name: "Value Recovery Wave",
      status: "DEMO",
      performance_1d: -0.23,
      performance_30d: 1.87,
      performance_ytd: 8.34,
      last_updated: now,
    },
  ];
};

const WaveCards = () => {
  const [waves, setWaves] = useState<WaveData[]>([]);
  const [dataSource, setDataSource] = useState<DataSource>("demo");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchWaveData = async () => {
      try {
        // Priority 1: External CSV from NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL
        const externalUrl = process.env.NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL;
        if (externalUrl) {
          try {
            const response = await fetch(externalUrl, {
              cache: "no-store",
            });
            if (response.ok) {
              const csvText = await response.text();
              const parsedWaves = parseCSV(csvText);
              if (parsedWaves.length > 0) {
                setWaves(parsedWaves);
                setDataSource("external");
                setIsLoading(false);
                return;
              }
            }
          } catch (error) {
            console.warn("External CSV fetch failed:", error);
          }
        }

        // Priority 2: Internal fallback API /api/live_snapshot.csv
        try {
          const response = await fetch("/api/live_snapshot.csv", {
            cache: "no-store",
          });
          if (response.ok) {
            const csvText = await response.text();
            const parsedWaves = parseCSV(csvText);
            if (parsedWaves.length > 0) {
              setWaves(parsedWaves);
              setDataSource("internal");
              setIsLoading(false);
              return;
            }
          }
        } catch (error) {
          console.warn("Internal CSV fetch failed:", error);
        }

        // Priority 3: Deterministic demo data (final fallback)
        setWaves(getDemoWaves());
        setDataSource("demo");
        setIsLoading(false);
      } catch (error) {
        console.error("Error fetching wave data:", error);
        setWaves(getDemoWaves());
        setDataSource("demo");
        setIsLoading(false);
      }
    };

    fetchWaveData();
    
    // Auto-refresh every 60 seconds
    const interval = setInterval(fetchWaveData, 60000);
    return () => clearInterval(interval);
  }, []);

  const getDataSourceBadge = () => {
    if (dataSource === "external" || dataSource === "internal") {
      return (
        <div className="inline-flex items-center gap-1.5 rounded border border-green-500/50 bg-green-500/20 px-2 py-0.5 text-xs font-semibold text-green-400">
          <span className="animate-pulse">●</span>
          <span>LIVE</span>
        </div>
      );
    }
    return (
      <div className="inline-flex items-center gap-1.5 rounded border border-gray-500/50 bg-gray-500/20 px-2 py-0.5 text-xs font-semibold text-gray-400">
        <span>○</span>
        <span>DEMO</span>
      </div>
    );
  };

  if (isLoading) {
    return (
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="animate-pulse">
            <div className="h-8 w-48 bg-gray-800 rounded mb-8 mx-auto"></div>
            <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-64 bg-gray-800 rounded-lg"></div>
              ))}
            </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="bg-black py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <h2 className="text-3xl font-bold text-white sm:text-4xl">
              Active Investment Waves
            </h2>
            {getDataSourceBadge()}
          </div>
          <p className="text-gray-400 text-sm">
            {dataSource === "external"
              ? "Data from external live feed"
              : dataSource === "internal"
              ? "Data from internal snapshot"
              : "Displaying demonstration data"}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
          {waves.map((wave) => (
            <div
              key={wave.wave_id}
              className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8"
            >
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xl font-semibold text-white">
                    {wave.wave_name}
                  </h3>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      wave.status === "LIVE"
                        ? "bg-green-500/20 text-green-400"
                        : "bg-gray-500/20 text-gray-400"
                    }`}
                  >
                    {wave.status}
                  </span>
                </div>
                <p className="text-xs text-gray-500">ID: {wave.wave_id}</p>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">1-Day</span>
                  <span
                    className={`text-sm font-semibold ${
                      wave.performance_1d >= 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {wave.performance_1d >= 0 ? "+" : ""}
                    {wave.performance_1d.toFixed(2)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">30-Day</span>
                  <span
                    className={`text-sm font-semibold ${
                      wave.performance_30d >= 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {wave.performance_30d >= 0 ? "+" : ""}
                    {wave.performance_30d.toFixed(2)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">YTD</span>
                  <span
                    className={`text-sm font-semibold ${
                      wave.performance_ytd >= 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {wave.performance_ytd >= 0 ? "+" : ""}
                    {wave.performance_ytd.toFixed(2)}%
                  </span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-700">
                <p className="text-xs text-gray-500">
                  Updated: {new Date(wave.last_updated).toLocaleString()}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default WaveCards;