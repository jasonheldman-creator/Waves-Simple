import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

interface WaveHistoryRow {
  wave_id: string;
  display_name: string;
  date: string;
  portfolio_return: number;
  benchmark_return: number;
  is_synthetic: boolean;
}

interface SnapshotRow {
  wave_id: string;
  wave_name: string;
  status: string;
  performance_1d: number;
  performance_30d: number;
  performance_ytd: number;
  last_updated: string;
}

function parseCSV(csvText: string): WaveHistoryRow[] {
  const lines = csvText.trim().split('\n');
  
  return lines.slice(1).map(line => {
    const values = line.split(',');
    return {
      wave_id: values[0],
      display_name: values[1],
      date: values[2],
      portfolio_return: parseFloat(values[3]),
      benchmark_return: parseFloat(values[4]),
      is_synthetic: values[5] === 'True',
    };
  }).filter(row => !isNaN(row.portfolio_return));
}

function calculateCumulativeReturn(returns: number[]): number {
  return returns.reduce((cum, ret) => cum * (1 + ret), 1) - 1;
}

function calculatePerformanceMetrics(rows: WaveHistoryRow[]): {
  performance_1d: number;
  performance_30d: number;
  performance_ytd: number;
} {
  // Sort by date descending (newest first)
  const sortedRows = [...rows].sort((a, b) => 
    new Date(b.date).getTime() - new Date(a.date).getTime()
  );

  // 1D performance (most recent day)
  const performance_1d = sortedRows.length > 0 ? sortedRows[0].portfolio_return : 0;

  // 30D performance (last 30 trading days)
  const last30Days = sortedRows.slice(0, Math.min(30, sortedRows.length));
  const performance_30d = calculateCumulativeReturn(
    last30Days.map(r => r.portfolio_return)
  );

  // YTD performance (year to date)
  const currentYear = new Date().getFullYear();
  const ytdRows = sortedRows.filter(r => new Date(r.date).getFullYear() === currentYear);
  const performance_ytd = calculateCumulativeReturn(
    ytdRows.map(r => r.portfolio_return)
  );

  return {
    performance_1d,
    performance_30d,
    performance_ytd,
  };
}

async function generateLiveSnapshot(): Promise<SnapshotRow[]> {
  try {
    const repoRoot = path.join(process.cwd(), "..");
    const historyPath = path.join(repoRoot, "wave_history.csv");
    
    const csvText = await fs.readFile(historyPath, "utf-8");
    const rows = parseCSV(csvText);
    
    if (rows.length === 0) {
      throw new Error("No wave history data found");
    }

    // Group by wave_id
    const waveGroups = new Map<string, WaveHistoryRow[]>();
    for (const row of rows) {
      if (!waveGroups.has(row.wave_id)) {
        waveGroups.set(row.wave_id, []);
      }
      waveGroups.get(row.wave_id)!.push(row);
    }

    const snapshots: SnapshotRow[] = [];

    for (const [wave_id, waveRows] of waveGroups) {
      // Sort by date descending and deduplicate (keep latest per date)
      const deduped = new Map<string, WaveHistoryRow>();
      for (const row of waveRows) {
        const existing = deduped.get(row.date);
        if (!existing) {
          deduped.set(row.date, row);
        }
        // If duplicate date exists, we keep the first one we encountered
        // (could also keep based on other criteria if needed)
      }
      
      const uniqueRows = Array.from(deduped.values());
      uniqueRows.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

      const latest = uniqueRows[0];
      const metrics = calculatePerformanceMetrics(uniqueRows);

      snapshots.push({
        wave_id,
        wave_name: latest.display_name,
        status: latest.is_synthetic ? "DEMO" : "LIVE",
        performance_1d: metrics.performance_1d,
        performance_30d: metrics.performance_30d,
        performance_ytd: metrics.performance_ytd,
        last_updated: latest.date,
      });
    }

    // Deduplicate by wave_id (keep newest last_updated)
    const dedupedSnapshots = new Map<string, SnapshotRow>();
    for (const snapshot of snapshots) {
      const existing = dedupedSnapshots.get(snapshot.wave_id);
      if (!existing || new Date(snapshot.last_updated) > new Date(existing.last_updated)) {
        dedupedSnapshots.set(snapshot.wave_id, snapshot);
      }
    }

    return Array.from(dedupedSnapshots.values());
  } catch (error) {
    console.error("Error generating live snapshot:", error);
    throw error;
  }
}

function generateDemoSnapshot(): SnapshotRow[] {
  // Generate deterministic placeholder values
  const demoWaves = [
    "ai_cloud_megacap_wave",
    "sp500_wave",
    "russell_3000_wave",
    "income_wave",
    "small_cap_growth_wave",
  ];

  const now = new Date().toISOString();
  
  return demoWaves.map((wave_id, index) => ({
    wave_id,
    wave_name: wave_id.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
    status: "DEMO",
    performance_1d: (index * 0.001) - 0.002, // Deterministic small values
    performance_30d: (index * 0.01) - 0.02,
    performance_ytd: (index * 0.05) - 0.1,
    last_updated: now,
  }));
}

function formatCSV(snapshots: SnapshotRow[]): string {
  const header = "wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated";
  const rows = snapshots.map(s => 
    `${s.wave_id},${s.wave_name},${s.status},${s.performance_1d},${s.performance_30d},${s.performance_ytd},${s.last_updated}`
  );
  
  return [header, ...rows].join('\n');
}

/**
 * GET /api/live_snapshot.csv
 * 
 * Serves a CSV snapshot of wave performance metrics.
 * Attempts to compute from wave_history.csv, falls back to demo data.
 */
export async function GET() {
  try {
    let snapshots: SnapshotRow[];
    
    try {
      snapshots = await generateLiveSnapshot();
    } catch (error) {
      console.warn("Failed to generate live snapshot, using demo data:", error);
      snapshots = generateDemoSnapshot();
    }

    const csv = formatCSV(snapshots);

    return new NextResponse(csv, {
      status: 200,
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Cache-Control": "no-store",
        "Content-Disposition": "inline; filename=live_snapshot.csv",
      },
    });
  } catch (error) {
    console.error("Error in live_snapshot.csv endpoint:", error);
    
    // Even on error, return demo data as CSV
    const demoSnapshots = generateDemoSnapshot();
    const csv = formatCSV(demoSnapshots);
    
    return new NextResponse(csv, {
      status: 200,
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Cache-Control": "no-store",
        "Content-Disposition": "inline; filename=live_snapshot.csv",
      },
    });
  }
}
