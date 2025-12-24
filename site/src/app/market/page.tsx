"use client";

import { useEffect, useState } from "react";
import type { Metadata } from "next";
import type { MarketData, MarketSymbol } from "@/types/market";
import { DEFAULT_MARKET_DATA } from "@/types/market";
import GovernancePanel from "@/components/GovernancePanel";
import RegimeCard from "@/components/RegimeCard";

type WatchlistTab = "macro" | "tech" | "rates" | "crypto";

export default function MarketPage() {
  const [activeTab, setActiveTab] = useState<WatchlistTab>("macro");
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

  const formatValue = (value: number, symbol: string) => {
    if (symbol === "BTC") {
      return value.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });
    }
    return value.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
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

  const renderSymbolRow = (symbol: MarketSymbol) => (
    <tr key={symbol.symbol} className="border-b border-gray-800 hover:bg-gray-800/30">
      <td className="px-4 py-3">
        <div>
          <div className="font-semibold text-white">{symbol.symbol}</div>
          <div className="text-xs text-gray-500">{symbol.name}</div>
        </div>
      </td>
      <td className="px-4 py-3 text-right font-mono text-white">
        {formatValue(symbol.value, symbol.symbol)}
      </td>
      <td className="px-4 py-3 text-right text-sm">
        {formatChange(symbol.dailyChange, symbol.dailyChangePercent)}
      </td>
      <td className="px-4 py-3 text-right text-sm">
        {formatChange(symbol.weeklyChange, symbol.weeklyChangePercent)}
      </td>
      <td className="px-4 py-3 text-right text-sm">
        {formatChange(symbol.monthlyChange, symbol.monthlyChangePercent)}
      </td>
    </tr>
  );

  const tabs = [
    { id: "macro" as WatchlistTab, label: "Macro" },
    { id: "tech" as WatchlistTab, label: "Tech" },
    { id: "rates" as WatchlistTab, label: "Rates" },
    { id: "crypto" as WatchlistTab, label: "Crypto" },
  ];

  return (
    <main className="min-h-screen bg-black">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-gray-900 to-black py-16 sm:py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-white sm:text-5xl lg:text-6xl">
              Market Intelligence Terminal
            </h1>
            <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto">
              Real-time market data, regime analysis, and governance signals for institutional decision-making
            </p>
          </div>

          {/* Data Delayed Notice */}
          {marketData.isDelayed && (
            <div className="mt-6 mx-auto max-w-2xl">
              <div className="rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-4 text-center">
                <div className="flex items-center justify-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-yellow-500 animate-pulse"></div>
                  <p className="text-sm text-yellow-400">{marketData.delayMessage}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Main Content */}
      <section className="bg-black py-12">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Regime Card - Takes 2 columns on large screens */}
            <div className="lg:col-span-2">
              <RegimeCard />
            </div>

            {/* Governance Panel - Takes 1 column */}
            <div>
              <GovernancePanel />
            </div>
          </div>

          {/* Market Watchlists */}
          <div className="rounded-lg border border-gray-800 bg-gray-900/50 overflow-hidden">
            {/* Tabs */}
            <div className="border-b border-gray-800">
              <div className="flex">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex-1 px-4 py-3 text-sm font-semibold transition-colors ${
                      activeTab === tab.id
                        ? "bg-cyan-500/10 text-cyan-400 border-b-2 border-cyan-500"
                        : "text-gray-400 hover:text-gray-300 hover:bg-gray-800/50"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Watchlist Table */}
            {isLoading ? (
              <div className="p-12 text-center">
                <div className="text-gray-500">Loading market data...</div>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase">
                        Symbol
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase">
                        Value
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase">
                        Daily
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase">
                        Weekly
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400 uppercase">
                        Monthly
                      </th>
                    </tr>
                  </thead>
                  <tbody>{marketData.symbols[activeTab].map(renderSymbolRow)}</tbody>
                </table>
              </div>
            )}
          </div>

          {/* Additional Info */}
          <div className="mt-6 text-center">
            <p className="text-xs text-gray-500">
              Last updated: {new Date(marketData.timestamp).toLocaleString()} • Data refresh
              interval: 60 seconds
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
