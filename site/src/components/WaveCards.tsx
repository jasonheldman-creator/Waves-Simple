"use client";

import { useEffect, useState } from "react";
import type { WaveMetrics } from "@/types/waves";

export default function WaveCards() {
  const [waves, setWaves] = useState<WaveMetrics[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchWaves = async () => {
      try {
        setIsLoading(true);
        const response = await fetch("/api/waves");
        if (!response.ok) {
          throw new Error("Failed to fetch waves data");
        }
        const data = await response.json();
        setWaves(data.waves || []);
        setError(null);
      } catch (err) {
        console.error("Error fetching waves:", err);
        setError(err instanceof Error ? err.message : "Failed to load waves");
      } finally {
        setIsLoading(false);
      }
    };

    fetchWaves();
    // Refresh every 60 seconds
    const interval = setInterval(fetchWaves, 60000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Active Investment Waves
            </h2>
            <p className="mt-4 text-lg text-gray-400">
              Loading wave performance data...
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className="rounded-lg border border-gray-800 bg-gray-900/50 p-6 animate-pulse"
              >
                <div className="h-6 w-3/4 bg-gray-800 rounded mb-4"></div>
                <div className="h-4 w-full bg-gray-800 rounded mb-2"></div>
                <div className="h-4 w-5/6 bg-gray-800 rounded"></div>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="bg-black py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Active Investment Waves
            </h2>
            <p className="mt-4 text-lg text-red-400">{error}</p>
          </div>
        </div>
      </section>
    );
  }

  const getReturnColor = (value: number) => {
    if (value > 0) return "text-green-400";
    if (value < 0) return "text-red-400";
    return "text-gray-400";
  };

  const formatPercent = (value: number) => {
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}%`;
  };

  return (
    <section className="bg-black py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
            Active Investment Waves
          </h2>
          <p className="mt-4 text-lg text-gray-400">
            Real-time performance metrics for institutional-grade investment strategies
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {waves.map((wave) => (
            <div
              key={wave.wave_id}
              className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 hover:border-cyan-500/40 transition-all"
            >
              <div className="mb-4">
                <h3 className="text-xl font-semibold text-white mb-2">
                  {wave.display_name}
                </h3>
                {wave.isSynthetic && (
                  <span className="inline-block rounded-md px-2 py-1 text-xs font-semibold bg-yellow-500/20 text-yellow-400">
                    DEMO DATA
                  </span>
                )}
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Today:</span>
                  <span className={`text-sm font-semibold ${getReturnColor(wave.todayReturn)}`}>
                    {formatPercent(wave.todayReturn)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">30-Day:</span>
                  <span className={`text-sm font-semibold ${getReturnColor(wave.monthReturn)}`}>
                    {formatPercent(wave.monthReturn)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">YTD:</span>
                  <span className={`text-sm font-semibold ${getReturnColor(wave.ytdReturn)}`}>
                    {formatPercent(wave.ytdReturn)}
                  </span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Alpha</div>
                    <div className={`text-sm font-semibold ${getReturnColor(wave.alpha)}`}>
                      {formatPercent(wave.alpha)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Beta</div>
                    <div className="text-sm font-semibold text-gray-300">
                      {wave.beta.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Sharpe</div>
                    <div className="text-sm font-semibold text-gray-300">
                      {wave.sharpeRatio.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>

              {wave.alerts.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  {wave.alerts.slice(0, 2).map((alert, idx) => (
                    <div
                      key={idx}
                      className="text-xs text-gray-400 mb-1 flex items-start gap-2"
                    >
                      <span className="text-yellow-400">⚠</span>
                      <span>{alert.message}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-4 text-xs text-gray-600">
                Updated: {new Date(wave.lastUpdate).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>

        {waves.length === 0 && !isLoading && !error && (
          <div className="text-center text-gray-400">
            No wave data available at this time.
          </div>
        )}
      </div>
    </section>
  );
}
