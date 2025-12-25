"use client";

import { useEffect, useState } from "react";
import type { MarketData, RegimeType } from "@/types/market";
import { DEFAULT_MARKET_DATA } from "@/types/market";

export default function RegimeCard() {
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

  const getRegimeIcon = (regime: RegimeType) => {
    switch (regime) {
      case "Risk-On":
        return "📈";
      case "Risk-Off":
        return "📉";
      default:
        return "⚖️";
    }
  };

  const getRegimeColor = (regime: RegimeType) => {
    switch (regime) {
      case "Risk-On":
        return "border-green-500/30 bg-green-500/5";
      case "Risk-Off":
        return "border-red-500/30 bg-red-500/5";
      default:
        return "border-yellow-500/30 bg-yellow-500/5";
    }
  };

  const getRegimeBadgeColor = (regime: RegimeType) => {
    switch (regime) {
      case "Risk-On":
        return "bg-green-500/20 text-green-400";
      case "Risk-Off":
        return "bg-red-500/20 text-red-400";
      default:
        return "bg-yellow-500/20 text-yellow-400";
    }
  };

  if (isLoading) {
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-6">
        <div className="animate-pulse">
          <div className="h-6 w-32 bg-gray-800 rounded mb-3"></div>
          <div className="h-4 w-full bg-gray-800 rounded mb-2"></div>
          <div className="h-4 w-3/4 bg-gray-800 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`rounded-lg border p-6 transition-all ${getRegimeColor(marketData.regime.regime)}`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="text-3xl">{getRegimeIcon(marketData.regime.regime)}</div>
          <div>
            <h3 className="text-lg font-semibold text-white mb-1">Today&apos;s Regime</h3>
            <div
              className={`inline-block rounded-md px-2 py-1 text-sm font-semibold ${getRegimeBadgeColor(marketData.regime.regime)}`}
            >
              {marketData.regime.regime}
            </div>
          </div>
        </div>
        <div className="text-xs text-gray-500">
          Confidence: {marketData.regime.confidence}
        </div>
      </div>
      <p className="text-sm text-gray-300 mb-3">{marketData.regime.description}</p>
      <div className="pt-3 border-t border-gray-700">
        <p className="text-xs font-semibold text-gray-400 mb-1">Exposure Implications</p>
        <p className="text-sm text-gray-300">{marketData.regime.exposureImplications}</p>
      </div>
    </div>
  );
}
