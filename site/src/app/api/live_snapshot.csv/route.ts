import { NextResponse } from "next/server";
import { readWavesData } from "@/lib/wavesDataReader";
import { WAVE_REGISTRY } from "@/lib/waveRegistry";

/**
 * GET /api/live_snapshot.csv
 * 
 * Returns wave performance data as CSV with:
 * - wave_id, wave_name, status, performance_1d, performance_30d, performance_ytd, last_updated
 * - Primary source: wave_history.csv metrics left-joined with canonical wave registry
 * - Fallback: Deterministic demo data for all registry waves
 * - NO OUTPUT FILTERING - always returns all waves from registry
 * - NO EXTERNAL NETWORK CALLS - local data only
 */
export async function GET() {
  const csvHeader = "wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated";
  
  try {
    // Try to read wave data from local wave_history.csv
    const { waves, syntheticPercentage } = await readWavesData();
    
    // Create a map of computed metrics by wave_id
    const metricsMap = new Map<string, typeof waves[0]>();
    for (const wave of waves) {
      metricsMap.set(wave.wave_id, wave);
    }
    
    // Build CSV content - one row per wave in registry (left join)
    const csvLines: string[] = [csvHeader];
    const now = new Date().toISOString();
    
    for (const registryWave of WAVE_REGISTRY) {
      const metrics = metricsMap.get(registryWave.wave_id);
      
      if (metrics) {
        // Wave has computed metrics from wave_history.csv
        const status = metrics.isSynthetic ? "DEMO" : "LIVE";
        const perf1d = metrics.todayReturn.toFixed(2);
        const perf30d = metrics.monthReturn.toFixed(2);
        const perfYtd = metrics.ytdReturn.toFixed(2);
        const waveName = `"${metrics.display_name}"`;
        
        csvLines.push(
          `${registryWave.wave_id},${waveName},${status},${perf1d},${perf30d},${perfYtd},${metrics.lastUpdate}`
        );
      } else {
        // Wave exists in registry but has no history data - output placeholder row
        const waveName = `"${registryWave.display_name}"`;
        csvLines.push(
          `${registryWave.wave_id},${waveName},DEMO,–,–,–,${now}`
        );
      }
    }
    
    const csvContent = csvLines.join('\n');
    
    return new NextResponse(csvContent, {
      headers: {
        'Content-Type': 'text/csv; charset=utf-8',
        'Cache-Control': 'no-store',
      },
    });
    
  } catch (error) {
    console.error("Error generating live_snapshot.csv from wave_history.csv:", error);
    console.log("Falling back to DEMO data with full registry coverage");
    
    // Fallback: Return all registry waves with demo/placeholder data
    const now = new Date().toISOString();
    const csvLines: string[] = [csvHeader];
    
    // Generate deterministic demo data for ALL waves in registry
    for (let i = 0; i < WAVE_REGISTRY.length; i++) {
      const wave = WAVE_REGISTRY[i];
      const waveName = `"${wave.display_name}"`;
      
      // Deterministic but varied demo performance based on index
      const seed = i * 123.456;
      const perf1d = (Math.sin(seed) * 0.5).toFixed(2);
      const perf30d = (Math.sin(seed * 1.5) * 3.0 + 2.0).toFixed(2);
      const perfYtd = (Math.sin(seed * 2.0) * 8.0 + 10.0).toFixed(2);
      
      csvLines.push(
        `${wave.wave_id},${waveName},DEMO,${perf1d},${perf30d},${perfYtd},${now}`
      );
    }
    
    const csvContent = csvLines.join('\n');
    
    return new NextResponse(csvContent, {
      headers: {
        'Content-Type': 'text/csv; charset=utf-8',
        'Cache-Control': 'no-store',
      },
    });
  }
}
