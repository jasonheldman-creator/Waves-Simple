"use client";

import { useEffect, useState } from "react";
import type { MarketData, RegimeType } from "@/types/market";
import { DEFAULT_MARKET_DATA } from "@/types/market";

export default function MarketTape() {
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

  const getRegimeBadgeColor = (regime: RegimeType) => {
    switch (regime) {
      case "Risk-On":
        return "bg-green-500/20 text-green-400 border-green-500/50";
      case "Risk-Off":
        return "bg-red-500/20 text-red-400 border-red-500/50";
      default:
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/50";
    }
  };

  const formatChange = (change: number, percent: number) => {
    const sign = change >= 0 ? "+" : "";
    const color = change >= 0 ? "text-green-400" : "text-red-400";
    return (
      <span className={color}>
        {sign}
        {percent.toFixed(2)}%
      </span>
    );
  };

  // Get key symbols from different categories
  const keySymbols = [
    marketData.symbols.macro.find((s) => s.symbol === "SPY"),
    marketData.symbols.macro.find((s) => s.symbol === "QQQ"),
    marketData.symbols.macro.find((s) => s.symbol === "IWM"),
    marketData.symbols.rates.find((s) => s.symbol === "TLT"),
    marketData.symbols.crypto.find((s) => s.symbol === "GLD"),
    marketData.symbols.crypto.find((s) => s.symbol === "BTC"),
  ].filter(Boolean);

  if (isLoading) {
    return (
      <div className="border-b border-gray-800 bg-gray-950/95 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-2">
          <div className="flex items-center justify-center">
            <div className="text-xs text-gray-500">Loading market data...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="border-b border-gray-800 bg-gray-950/95 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 py-2">
        <div className="flex items-center justify-between gap-4 overflow-x-auto">
          {/* WAVES Regime Badge */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-xs font-semibold text-gray-400">WAVES™</span>
            <div
              className={`rounded-md border px-2 py-1 text-xs font-semibold ${getRegimeBadgeColor(marketData.regime.regime)}`}
            >
              {marketData.regime.regime}
            </div>
          </div>

          {/* Market Symbols Ticker */}
          <div className="flex items-center gap-4 overflow-x-auto scrollbar-hide flex-1 min-w-0">
            {keySymbols.map((symbol) =>
              symbol ? (
                <div key={symbol.symbol} className="flex items-center gap-2 flex-shrink-0">
                  <span className="text-xs font-semibold text-gray-300">
                    {symbol.symbol}
                  </span>
                  <span className="text-xs text-gray-400">
                    {symbol.value.toLocaleString("en-US", {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </span>
                  <span className="text-xs">
                    {formatChange(symbol.dailyChange, symbol.dailyChangePercent)}
                  </span>
                </div>
              ) : null
            )}
          </div>

          {/* Data State Indicator */}
          {marketData.isDelayed && (
            <div className="flex items-center gap-2 flex-shrink-0">
              <div className="h-2 w-2 rounded-full bg-yellow-500 animate-pulse"></div>
              <span className="text-xs text-gray-500">Delayed</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
