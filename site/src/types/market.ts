/**
 * Market Intelligence Data Types
 */

export type RegimeType = "Risk-On" | "Neutral" | "Risk-Off";

export type ConfidenceLevel = "HIGH" | "MED" | "LOW";

export type DataState = "LIVE" | "SNAPSHOT" | "FALLBACK";

export interface MarketSymbol {
  symbol: string;
  name: string;
  value: number;
  dailyChange: number;
  dailyChangePercent: number;
  weeklyChange: number;
  weeklyChangePercent: number;
  monthlyChange: number;
  monthlyChangePercent: number;
}

export interface RegimeData {
  regime: RegimeType;
  description: string;
  exposureImplications: string;
  confidence: ConfidenceLevel;
}

export interface GovernanceSignals {
  dataState: DataState;
  attributionConfidence: ConfidenceLevel;
  benchmarkStability: ConfidenceLevel;
  lastUpdate: string;
}

export interface MarketData {
  timestamp: string;
  regime: RegimeData;
  governance: GovernanceSignals;
  symbols: {
    macro: MarketSymbol[];
    tech: MarketSymbol[];
    rates: MarketSymbol[];
    crypto: MarketSymbol[];
  };
  isDelayed: boolean;
  delayMessage?: string;
}

export const DEFAULT_MARKET_DATA: MarketData = {
  timestamp: new Date().toISOString(),
  regime: {
    regime: "Neutral",
    description: "Market conditions showing balanced risk/reward dynamics with moderate volatility.",
    exposureImplications: "Maintain balanced exposure across growth and defensive sectors. Monitor regime transitions.",
    confidence: "MED",
  },
  governance: {
    dataState: "FALLBACK",
    attributionConfidence: "MED",
    benchmarkStability: "HIGH",
    lastUpdate: new Date().toISOString(),
  },
  symbols: {
    macro: [
      {
        symbol: "SPY",
        name: "S&P 500 ETF",
        value: 475.23,
        dailyChange: 2.15,
        dailyChangePercent: 0.45,
        weeklyChange: -3.42,
        weeklyChangePercent: -0.72,
        monthlyChange: 12.58,
        monthlyChangePercent: 2.72,
      },
      {
        symbol: "QQQ",
        name: "Nasdaq-100 ETF",
        value: 398.67,
        dailyChange: 3.22,
        dailyChangePercent: 0.81,
        weeklyChange: -5.18,
        weeklyChangePercent: -1.28,
        monthlyChange: 18.94,
        monthlyChangePercent: 4.99,
      },
      {
        symbol: "IWM",
        name: "Russell 2000 ETF",
        value: 195.42,
        dailyChange: 1.08,
        dailyChangePercent: 0.55,
        weeklyChange: -2.37,
        weeklyChangePercent: -1.20,
        monthlyChange: 8.76,
        monthlyChangePercent: 4.69,
      },
    ],
    tech: [
      {
        symbol: "AAPL",
        name: "Apple Inc.",
        value: 185.92,
        dailyChange: 2.14,
        dailyChangePercent: 1.16,
        weeklyChange: -3.87,
        weeklyChangePercent: -2.04,
        monthlyChange: 14.23,
        monthlyChangePercent: 8.29,
      },
      {
        symbol: "MSFT",
        name: "Microsoft Corp.",
        value: 378.91,
        dailyChange: 4.52,
        dailyChangePercent: 1.21,
        weeklyChange: -6.14,
        weeklyChangePercent: -1.59,
        monthlyChange: 22.45,
        monthlyChangePercent: 6.30,
      },
      {
        symbol: "NVDA",
        name: "NVIDIA Corp.",
        value: 495.22,
        dailyChange: 8.73,
        dailyChangePercent: 1.79,
        weeklyChange: -12.44,
        weeklyChangePercent: -2.45,
        monthlyChange: 48.67,
        monthlyChangePercent: 10.90,
      },
    ],
    rates: [
      {
        symbol: "TLT",
        name: "20+ Yr Treasury ETF",
        value: 92.15,
        dailyChange: -0.42,
        dailyChangePercent: -0.45,
        weeklyChange: 1.23,
        weeklyChangePercent: 1.35,
        monthlyChange: -3.67,
        monthlyChangePercent: -3.83,
      },
      {
        symbol: "^TNX",
        name: "10-Year Treasury Yield",
        value: 4.25,
        dailyChange: 0.08,
        dailyChangePercent: 1.92,
        weeklyChange: -0.15,
        weeklyChangePercent: -3.41,
        monthlyChange: 0.42,
        monthlyChangePercent: 10.96,
      },
    ],
    crypto: [
      {
        symbol: "BTC",
        name: "Bitcoin",
        value: 42873.52,
        dailyChange: 892.43,
        dailyChangePercent: 2.12,
        weeklyChange: -1248.76,
        weeklyChangePercent: -2.83,
        monthlyChange: 5432.18,
        monthlyChangePercent: 14.51,
      },
      {
        symbol: "GLD",
        name: "Gold ETF",
        value: 187.34,
        dailyChange: 1.23,
        dailyChangePercent: 0.66,
        weeklyChange: -2.14,
        weeklyChangePercent: -1.13,
        monthlyChange: 8.92,
        monthlyChangePercent: 5.00,
      },
    ],
  },
  isDelayed: true,
  delayMessage: "Data delayed - Fallback market intelligence in use",
};
