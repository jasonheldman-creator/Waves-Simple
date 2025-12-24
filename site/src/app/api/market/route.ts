import { NextResponse } from "next/server";
import { DEFAULT_MARKET_DATA, type MarketData } from "@/types/market";

// Cache configuration
const CACHE_DURATION = 60; // 60 seconds

// In-memory cache
let cachedData: MarketData | null = null;
let cacheTimestamp: number = 0;

/**
 * GET /api/market
 * 
 * Returns market intelligence data with caching.
 * Uses stub data with "Data delayed" fallback when no live data is available.
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

  // In a production environment, you would fetch live data here
  // For now, we return stub data with current timestamp
  const marketData: MarketData = {
    ...DEFAULT_MARKET_DATA,
    timestamp: new Date().toISOString(),
    governance: {
      ...DEFAULT_MARKET_DATA.governance,
      lastUpdate: new Date().toISOString(),
    },
  };

  // Update cache
  cachedData = marketData;
  cacheTimestamp = now;

  return NextResponse.json(marketData, {
    headers: {
      "Cache-Control": `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
    },
  });
}
