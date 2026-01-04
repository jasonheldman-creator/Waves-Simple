"use client";

import { useEffect, useState } from "react";
import type { MarketData, DataState, ConfidenceLevel } from "@/types/market";
import { DEFAULT_MARKET_DATA } from "@/types/market";

export default function GovernancePanel() {
  const [marketData, setMarketData] = useState<MarketData>(DEFAULT_MARKET_DATA);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        const response = await fetch("/api/market");
        if (response.ok) {
          const data = await response.json();
          setMarketData(data);
        }
      } catch (error) {
        console.error("Failed to fetch market data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchMarketData();
    // Refresh every 60 seconds
    const interval = setInterval(fetchMarketData, 60000);
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

  const getConfidenceColor = (level: ConfidenceLevel) => {
    switch (level) {
      case "HIGH":
        return "text-green-400";
      case "MED":
        return "text-yellow-400";
      default:
        return "text-orange-400";
    }
  };

  if (isLoading) {
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-4">
        <div className="animate-pulse">
          <div className="h-5 w-40 bg-gray-800 rounded mb-3"></div>
          <div className="space-y-2">
            <div className="h-4 w-full bg-gray-800 rounded"></div>
            <div className="h-4 w-full bg-gray-800 rounded"></div>
            <div className="h-4 w-full bg-gray-800 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-cyan-500/20 bg-gray-900/50 p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className="text-2xl">⚙️</div>
        <h3 className="text-base font-semibold text-white">Governance Signals</h3>
      </div>

      <div className="space-y-2">
        {/* Data State */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Data State</span>
          <div
            className={`rounded border px-2 py-0.5 text-xs font-semibold ${getDataStateColor(marketData.governance.dataState)}`}
          >
            {marketData.governance.dataState}
          </div>
        </div>

        {/* Attribution Confidence */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Attribution Confidence</span>
          <span
            className={`text-xs font-semibold ${getConfidenceColor(marketData.governance.attributionConfidence)}`}
          >
            {marketData.governance.attributionConfidence}
          </span>
        </div>

        {/* Benchmark Stability */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Benchmark Stability</span>
          <span
            className={`text-xs font-semibold ${getConfidenceColor(marketData.governance.benchmarkStability)}`}
          >
            {marketData.governance.benchmarkStability}
          </span>
        </div>

        {/* Last Update */}
        <div className="pt-2 mt-2 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">Last Update</span>
            <span className="text-xs text-gray-500">
              {new Date(marketData.governance.lastUpdate).toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
