import { NextResponse } from "next/server";
import { readWavesData } from "@/lib/wavesDataReader";

// Force Node.js runtime for file system access
export const runtime = "nodejs";

/**
 * GET /api/live_snapshot.csv
 * 
 * Returns wave performance data as CSV with:
 * - wave_id, wave_name, status, performance_1d, performance_30d, performance_ytd, last_updated
 * - Primary source: wave_history.csv metrics
 * - Fallback: Deterministic demo data
 */
export async function GET() {
  try {
    // Try to read real wave data
    const { waves, syntheticPercentage } = await readWavesData();
    
    // Build CSV content
    const csvLines: string[] = [
      "wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated"
    ];
    
    // Deduplicate by wave_id (use Map to preserve order)
    const waveMap = new Map<string, typeof waves[0]>();
    for (const wave of waves) {
      if (!waveMap.has(wave.wave_id)) {
        waveMap.set(wave.wave_id, wave);
      }
    }
    
    const uniqueWaves = Array.from(waveMap.values());
    const status = syntheticPercentage === 100 ? "DEMO" : "LIVE";
    
    for (const wave of uniqueWaves) {
      // Format performance values to 2 decimal places
      const perf1d = wave.todayReturn.toFixed(2);
      const perf30d = wave.monthReturn.toFixed(2);
      const perfYtd = wave.ytdReturn.toFixed(2);
      
      // Escape fields that might contain commas
      const waveName = `"${wave.display_name}"`;
      
      csvLines.push(
        `${wave.wave_id},${waveName},${status},${perf1d},${perf30d},${perfYtd},${wave.lastUpdate}`
      );
    }
    
    const csvContent = csvLines.join('\n');
    
    return new NextResponse(csvContent, {
      headers: {
        'Content-Type': 'text/csv; charset=utf-8',
        'Cache-Control': 'no-store',
      },
    });
    
  } catch (error) {
    console.error("Error generating live_snapshot.csv:", error);
    
    // Fallback to demo data
    const now = new Date().toISOString();
    const csvLines: string[] = [
      "wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated"
    ];
    
    // Generate deterministic demo data for 15 waves
    const demoWaves = [
      { id: "core_equity_wave", name: "Core Equity Wave", p1d: 0.45, p30d: 2.31, pytd: 12.45 },
      { id: "growth_alpha_wave", name: "Growth Alpha Wave", p1d: 0.82, p30d: 4.56, pytd: 18.92 },
      { id: "value_recovery_wave", name: "Value Recovery Wave", p1d: -0.23, p30d: 1.87, pytd: 8.34 },
      { id: "income_generation_wave", name: "Income Generation Wave", p1d: 0.12, p30d: 1.45, pytd: 6.78 },
      { id: "multi_factor_wave", name: "Multi-Factor Wave", p1d: 0.56, p30d: 3.21, pytd: 14.67 },
      { id: "sector_rotation_wave", name: "Sector Rotation Wave", p1d: 0.34, p30d: 2.98, pytd: 11.23 },
      { id: "global_diversification_wave", name: "Global Diversification Wave", p1d: 0.67, p30d: 3.45, pytd: 15.89 },
      { id: "small_mid_cap_wave", name: "Small-Mid Cap Wave", p1d: 0.91, p30d: 4.12, pytd: 19.45 },
      { id: "innovation_thematic_wave", name: "Innovation Thematic Wave", p1d: 1.23, p30d: 5.67, pytd: 22.34 },
      { id: "defensive_positioning_wave", name: "Defensive Positioning Wave", p1d: 0.08, p30d: 0.89, pytd: 4.56 },
      { id: "esg_integration_wave", name: "ESG Integration Wave", p1d: 0.43, p30d: 2.76, pytd: 13.21 },
      { id: "volatility_harvesting_wave", name: "Volatility Harvesting Wave", p1d: 0.19, p30d: 1.34, pytd: 7.89 },
      { id: "macro_allocation_wave", name: "Macro Allocation Wave", p1d: 0.51, p30d: 3.09, pytd: 14.12 },
      { id: "market_neutral_wave", name: "Market Neutral Wave", p1d: 0.15, p30d: 1.23, pytd: 5.67 },
      { id: "special_situations_wave", name: "Special Situations Wave", p1d: 0.78, p30d: 3.87, pytd: 16.45 },
    ];
    
    for (const wave of demoWaves) {
      csvLines.push(
        `${wave.id},"${wave.name}",DEMO,${wave.p1d.toFixed(2)},${wave.p30d.toFixed(2)},${wave.pytd.toFixed(2)},${now}`
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
