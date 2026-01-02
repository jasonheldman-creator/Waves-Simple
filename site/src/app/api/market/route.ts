import { NextResponse } from "next/server";
import {
  DEFAULT_MARKET_DATA,
  type MarketData,
  type MarketSymbol,
  type DataState,
} from "@/types/market";
import { fetchMultipleSymbols, fetchBitcoinPrice } from "@/lib/marketDataFetcher";

// Cache configuration (60-300 seconds as specified)
const CACHE_DURATION = 120; // 2 minutes for balance between freshness and API load

// In-memory cache
let cachedData: MarketData | null = null;
let cacheTimestamp: number = 0;

/**
 * Determine market regime based on VIX and market conditions
 */
function determineMarketRegime(symbols: MarketSymbol[]): MarketData["regime"] {
  // Find VIX if available
  const spy = symbols.find((s) => s.symbol === "SPY");
  const vix = symbols.find((s) => s.symbol === "^VIX");

  // Simple regime logic based on SPY performance and VIX
  let regime: MarketData["regime"]["regime"] = "Neutral";
  let description =
    "Market conditions showing balanced risk/reward dynamics with moderate volatility.";
  let exposureImplications =
    "Maintain balanced exposure across growth and defensive sectors. Monitor regime transitions.";

  if (spy) {
    const dailyChange = spy.dailyChangePercent;

    if (vix && vix.value > 25) {
      regime = "Risk-Off";
      description =
        "Elevated volatility with defensive market behavior. Heightened uncertainty driving risk aversion.";
      exposureImplications =
        "Reduce growth exposure. Increase defensive positions and consider hedging strategies.";
    } else if (dailyChange > 1.0) {
      regime = "Risk-On";
      description =
        "Strong market momentum with risk asset outperformance. Bullish sentiment prevailing.";
      exposureImplications =
        "Favor growth and cyclical sectors. Monitor for overextension and prepare for rotation.";
    } else if (dailyChange < -1.0) {
      regime = "Risk-Off";
      description =
        "Market showing defensive characteristics. Risk assets underperforming safe havens.";
      exposureImplications =
        "Reduce growth exposure. Increase defensive positions and quality bias.";
    }
  }

  return {
    regime,
    description,
    exposureImplications,
    confidence: "MED",
  };
}

/**
 * GET /api/market
 *
 * Returns live market intelligence data with caching and fallback.
 * Fetches from Stooq (equities/ETFs) and CoinGecko (BTC) without API keys.
 */
export async function GET() {
  const now = Date.now();

  // Check cache validity
  if (cachedData && now - cacheTimestamp < CACHE_DURATION * 1000) {
    return NextResponse.json(cachedData, {
      headers: {
        "Cache-Control": `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
      },
    });
  }

  try {
    // Fetch live market data
    const macroSymbols = ["SPY", "QQQ", "IWM", "GLD"];
    const techSymbols = ["AAPL", "MSFT", "NVDA"];
    const ratesSymbols = ["TLT", "^TNX", "^VIX"];

    // Fetch all symbol groups in parallel
    const [macroResult, techResult, ratesResult, btcData] = await Promise.allSettled([
      fetchMultipleSymbols(macroSymbols),
      fetchMultipleSymbols(techSymbols),
      fetchMultipleSymbols(ratesSymbols),
      fetchBitcoinPrice(),
    ]);

    // Collect successful results
    const macro = macroResult.status === "fulfilled" ? macroResult.value.symbols : [];
    const tech = techResult.status === "fulfilled" ? techResult.value.symbols : [];
    const rates = ratesResult.status === "fulfilled" ? ratesResult.value.symbols : [];
    const crypto = btcData.status === "fulfilled" ? [btcData.value] : [];

    // Determine overall data state
    let dataState: DataState = "LIVE";
    const totalExpected = macroSymbols.length + techSymbols.length + ratesSymbols.length + 1; // +1 for BTC
    const totalReceived = macro.length + tech.length + rates.length + crypto.length;
    const successRate = totalReceived / totalExpected;

    if (successRate === 0) {
      // Complete failure - use fallback
      throw new Error("All market data sources failed");
    } else if (successRate < 0.8) {
      dataState = "SNAPSHOT"; // Partial data
    }

    // Merge with fallback data for any missing symbols
    const macroFallback = DEFAULT_MARKET_DATA.symbols.macro.filter(
      (fb) => !macro.find((s) => s.symbol === fb.symbol)
    );
    const techFallback = DEFAULT_MARKET_DATA.symbols.tech.filter(
      (fb) => !tech.find((s) => s.symbol === fb.symbol)
    );
    const ratesFallback = DEFAULT_MARKET_DATA.symbols.rates.filter(
      (fb) => !rates.find((s) => s.symbol === fb.symbol)
    );
    const cryptoFallback = DEFAULT_MARKET_DATA.symbols.crypto.filter(
      (fb) => !crypto.find((s) => s.symbol === fb.symbol)
    );

    const allSymbols = [...macro, ...tech, ...rates, ...crypto];
    const regime = determineMarketRegime(allSymbols);

    const marketData: MarketData = {
      timestamp: new Date().toISOString(),
      regime,
      governance: {
        dataState,
        attributionConfidence: dataState === "LIVE" ? "HIGH" : "MED",
        benchmarkStability: "HIGH",
        lastUpdate: new Date().toISOString(),
      },
      symbols: {
        macro: [...macro, ...macroFallback],
        tech: [...tech, ...techFallback],
        rates: [...rates, ...ratesFallback],
        crypto: [...crypto, ...cryptoFallback],
      },
      isDelayed: dataState !== "LIVE",
      delayMessage:
        dataState === "LIVE"
          ? undefined
          : dataState === "SNAPSHOT"
            ? "Using last validated snapshot. Live feed delayed."
            : "Data delayed - Fallback market intelligence in use",
    };

    // Update cache
    cachedData = marketData;
    cacheTimestamp = now;

    return NextResponse.json(marketData, {
      headers: {
        "Cache-Control": `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
      },
    });
  } catch (error) {
    console.error("Market data fetch error:", error);

    // Return last cached data if available
    if (cachedData) {
      return NextResponse.json(
        {
          ...cachedData,
          governance: {
            ...cachedData.governance,
            dataState: "FALLBACK" as DataState,
          },
          isDelayed: true,
          delayMessage: "Using last validated snapshot. Live feed delayed.",
        },
        {
          headers: {
            "Cache-Control": `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
          },
        }
      );
    }

    // Ultimate fallback to default data
    const fallbackData: MarketData = {
      ...DEFAULT_MARKET_DATA,
      timestamp: new Date().toISOString(),
      governance: {
        ...DEFAULT_MARKET_DATA.governance,
        dataState: "FALLBACK",
        lastUpdate: new Date().toISOString(),
      },
    };

    cachedData = fallbackData;
    cacheTimestamp = now;

    return NextResponse.json(fallbackData, {
      headers: {
        "Cache-Control": `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
      },
    });
  }
}
