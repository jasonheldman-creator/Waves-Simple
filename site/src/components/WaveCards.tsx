"use client";

import { useEffect, useState } from "react";
import type { WavesData, WaveMetrics } from "../types/waves";
import type { DataState } from "../types/market";

export default function WaveCards() {
  const [wavesData, setWavesData] = useState<WavesData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchWavesData = async () => {
      try {
        const response = await fetch("/api/waves");
        if (response.ok) {
          const data = await response.json();
          setWavesData(data);
        }
      } catch (error) {
        console.error("Failed to fetch waves data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchWavesData();
    // Refresh every 3 minutes
    const interval = setInterval(fetchWavesData, 180000);
    return () => clearInterval(interval);
  }, []);

  const getDataStateColor = (state: DataState) => {
    switch (state) {
      case "LIVE":
        return "bg-green-500/20 text-green-400 border-green-500/50";
      case "SNAPSHOT":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/50";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/50";
    }
  };

  const formatPercentage = (value: number): string => {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}%`;
  };

  const getPerformanceColor = (value: number): string => {
    if (value > 0) return "text-green-400";
    if (value < 0) return "text-red-400";
    return "text-gray-400";
  };

  if (isLoading) {
    return (
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Live Wave Performance
            </h2>
            <p className="mt-4 text-lg text-gray-400">
              Loading wave metrics...
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="rounded-lg border border-gray-800 bg-gray-900/50 p-6 animate-pulse">
                <div className="h-6 w-3/4 bg-gray-800 rounded mb-4"></div>
                <div className="h-4 w-1/2 bg-gray-800 rounded mb-3"></div>
                <div className="h-4 w-full bg-gray-800 rounded"></div>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  if (!wavesData || wavesData.waves.length === 0) {
    return (
      <section className="bg-black py-20 sm:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
              Live Wave Performance
            </h2>
            <p className="mt-4 text-lg text-gray-400">
              No wave data available at this time.
            </p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="bg-black py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
            Live Wave Performance
          </h2>
          <p className="mt-4 text-lg text-gray-400">
            Real-time portfolio metrics and analytics
          </p>
          <div className="mt-6 flex justify-center gap-4">
            <span className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium border ${getDataStateColor(wavesData.dataState)}`}>
              {wavesData.dataState}
            </span>
            <span className="text-sm text-gray-500">
              As of: {new Date(wavesData.asOf).toLocaleString()}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {wavesData.waves.map((wave: WaveMetrics) => (
            <div
              key={wave.wave_id}
              className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6 hover:border-cyan-500/40 transition-colors"
            >
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-semibold text-white">
                  {wave.display_name}
                </h3>
                {wave.isSynthetic && (
                  <span className="text-xs px-2 py-1 rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/50">
                    DEMO
                  </span>
                )}
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Today</span>
                  <span className={`text-sm font-medium ${getPerformanceColor(wave.todayReturn)}`}>
                    {formatPercentage(wave.todayReturn)}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">1 Month</span>
                  <span className={`text-sm font-medium ${getPerformanceColor(wave.monthReturn)}`}>
                    {formatPercentage(wave.monthReturn)}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">YTD</span>
                  <span className={`text-sm font-medium ${getPerformanceColor(wave.ytdReturn)}`}>
                    {formatPercentage(wave.ytdReturn)}
                  </span>
                </div>

                <div className="pt-3 border-t border-gray-700">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs text-gray-500">Alpha</span>
                    <span className="text-xs text-cyan-400">{formatPercentage(wave.alpha)}</span>
                  </div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs text-gray-500">Beta</span>
                    <span className="text-xs text-gray-300">{wave.beta.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-500">Sharpe</span>
                    <span className="text-xs text-gray-300">{wave.sharpeRatio.toFixed(2)}</span>
                  </div>
                </div>

                {wave.alerts.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-700">
                    {wave.alerts.slice(0, 2).map((alert, idx) => (
                      <div key={idx} className="text-xs text-orange-400 mb-1">
                        ⚠ {alert.message}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {wavesData.alerts.length > 0 && (
          <div className="mt-12 rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-6">
            <h3 className="text-lg font-semibold text-yellow-400 mb-4">System Alerts</h3>
            <div className="space-y-2">
              {wavesData.alerts.map((alert, idx) => (
                <div key={idx} className="text-sm text-gray-300">
                  <span className="text-yellow-400">•</span> {alert.message}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}