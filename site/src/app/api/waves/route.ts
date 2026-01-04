import { NextResponse } from "next/server";
import { readWavesData } from "@/lib/wavesDataReader";
import type { WavesData, WaveAlert } from "@/types/waves";

// Cache configuration (60-300 seconds as specified)
const CACHE_DURATION = 180; // 3 minutes for WAVES data

// In-memory cache
let cachedData: WavesData | null = null;
let cacheTimestamp: number = 0;

// Fallback data for when CSV is not available
const FALLBACK_WAVES_DATA: WavesData = {
  timestamp: new Date().toISOString(),
  dataState: "FALLBACK",
  asOf: new Date().toISOString(),
  waves: [],
  alerts: [
    {
      type: "data_quality",
      severity: "medium",
      message: "WAVES data source unavailable. Using fallback mode.",
      timestamp: new Date().toISOString(),
    },
  ],
  governance: {
    dataState: "FALLBACK",
    confidenceLevel: "LOW",
    syntheticDataPercentage: 100,
    lastUpdate: new Date().toISOString(),
  },
};

/**
 * GET /api/waves
 *
 * Returns WAVES portfolio metrics with caching and fallback.
 * Primary source: wave_history.csv from repository
 * Fallback: Returns empty waves with clear messaging
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
    // Read WAVES data from CSV
    const { waves, dataState, syntheticPercentage } = await readWavesData();

    // Collect all alerts from waves
    const allAlerts: WaveAlert[] = [];
    for (const wave of waves) {
      allAlerts.push(...wave.alerts);
    }

    // Add global alerts
    if (syntheticPercentage === 100) {
      allAlerts.push({
        type: "data_quality",
        severity: "medium",
        message: "All WAVES data is synthetic. Live portfolio data pending.",
        timestamp: new Date().toISOString(),
      });
    } else if (syntheticPercentage > 50) {
      allAlerts.push({
        type: "data_quality",
        severity: "low",
        message: `${syntheticPercentage.toFixed(0)}% of WAVES using synthetic data`,
        timestamp: new Date().toISOString(),
      });
    }

    const wavesData: WavesData = {
      timestamp: new Date().toISOString(),
      dataState,
      asOf: waves.length > 0 ? waves[0].lastUpdate : new Date().toISOString(),
      waves,
      alerts: allAlerts,
      governance: {
        dataState,
        confidenceLevel:
          syntheticPercentage === 0 ? "HIGH" : syntheticPercentage < 50 ? "MED" : "LOW",
        syntheticDataPercentage: syntheticPercentage,
        lastUpdate: new Date().toISOString(),
      },
    };

    // Update cache
    cachedData = wavesData;
    cacheTimestamp = now;

    return NextResponse.json(wavesData, {
      headers: {
        "Cache-Control": `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
      },
    });
  } catch (error) {
    console.error("WAVES data fetch error:", error);

    // Return last cached data if available
    if (cachedData) {
      return NextResponse.json(
        {
          ...cachedData,
          governance: {
            ...cachedData.governance,
            dataState: "FALLBACK",
          },
        },
        {
          headers: {
            "Cache-Control": `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
          },
        }
      );
    }

    // Ultimate fallback
    cachedData = FALLBACK_WAVES_DATA;
    cacheTimestamp = now;

    return NextResponse.json(FALLBACK_WAVES_DATA, {
      headers: {
        "Cache-Control": `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
      },
    });
  }
}
