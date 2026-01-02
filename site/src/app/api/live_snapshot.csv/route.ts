import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

interface WaveRegistryRow {
  wave_id: string;
  wave_name: string;
  mode_default: string;
  benchmark_spec: string;
  holdings_source: string;
  category: string;
  active: string;
  ticker_raw: string;
  ticker_normalized: string;
  created_at: string;
  updated_at: string;
}

interface WaveHistoryRow {
  wave_id: string;
  display_name: string;
  date: string;
  portfolio_return: number;
  benchmark_return: number;
  is_synthetic: boolean;
}

function parseRegistryCSV(csvText: string): WaveRegistryRow[] {
  const lines = csvText.trim().split('\n');
  return lines.slice(1).map(line => {
    const values = line.split(',');
    return {
      wave_id: values[0],
      wave_name: values[1],
      mode_default: values[2],
      benchmark_spec: values[3],
      holdings_source: values[4],
      category: values[5],
      active: values[6],
      ticker_raw: values[7],
      ticker_normalized: values[8],
      created_at: values[9],
      updated_at: values[10],
    };
  });
}

function parseHistoryCSV(csvText: string): WaveHistoryRow[] {
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

/**
 * GET /api/live_snapshot.csv
 * 
 * Returns CSV data for all waves in the canonical registry.
 * Performs left join of metrics from wave_history.csv to the registry.
 * Missing metrics are filled with placeholder "--".
 */
export async function GET() {
  try {
    const repoRoot = path.join(process.cwd(), "..");
    const registryPath = path.join(repoRoot, "data", "wave_registry.csv");
    const historyPath = path.join(repoRoot, "wave_history.csv");
    
    // Read canonical wave registry
    const registryText = await fs.readFile(registryPath, "utf-8");
    const registryRows = parseRegistryCSV(registryText);
    
    // Read wave history and compute metrics
    let historyRows: WaveHistoryRow[] = [];
    try {
      const historyText = await fs.readFile(historyPath, "utf-8");
      historyRows = parseHistoryCSV(historyText);
    } catch (error) {
      // If history file not found, continue with empty history
      console.warn("Wave history file not found, using placeholders");
    }
    
    // Group history by wave_id
    const historyByWave = new Map<string, WaveHistoryRow[]>();
    for (const row of historyRows) {
      if (!historyByWave.has(row.wave_id)) {
        historyByWave.set(row.wave_id, []);
      }
      historyByWave.get(row.wave_id)!.push(row);
    }
    
    // Build CSV rows with left join
    const csvRows: string[] = ["wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated"];
    
    // Expected wave count (from wave_weights.csv canonical list)
    const EXPECTED_WAVE_COUNT = 28;
    
    for (const wave of registryRows) {
      const waveHistory = historyByWave.get(wave.wave_id);
      
      let status = "DEMO";
      let performance1d = "--";
      let performance30d = "--";
      let performanceYtd = "--";
      let lastUpdated = "--";
      
      if (waveHistory && waveHistory.length > 0) {
        // Sort by date descending
        waveHistory.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
        
        const latest = waveHistory[0];
        const isSynthetic = waveHistory.some(r => r.is_synthetic);
        
        status = isSynthetic ? "DEMO" : "Active";
        lastUpdated = latest.date;
        
        // Calculate 1-day performance (latest return)
        performance1d = (latest.portfolio_return * 100).toFixed(2) + "%";
        
        // Calculate 30-day performance (last 21 trading days)
        const last30Days = waveHistory.slice(0, Math.min(21, waveHistory.length));
        const returns30d = last30Days.map(r => r.portfolio_return);
        const cumReturn30d = calculateCumulativeReturn(returns30d);
        performance30d = (cumReturn30d * 100).toFixed(2) + "%";
        
        // Calculate YTD performance (all available data)
        const returnsYtd = waveHistory.map(r => r.portfolio_return);
        const cumReturnYtd = calculateCumulativeReturn(returnsYtd);
        performanceYtd = (cumReturnYtd * 100).toFixed(2) + "%";
      }
      
      csvRows.push(`${wave.wave_id},${wave.wave_name},${status},${performance1d},${performance30d},${performanceYtd},${lastUpdated}`);
    }
    
    // Hard assertion - validate exactly EXPECTED_WAVE_COUNT rows
    // Check CSV rows (excluding header) instead of registry rows for accuracy
    const actualDataRowCount = csvRows.length - 1; // -1 for header row
    if (actualDataRowCount !== EXPECTED_WAVE_COUNT) {
      const errorMsg = `VALIDATION FAILED: Expected ${EXPECTED_WAVE_COUNT} data rows but got ${actualDataRowCount} in CSV output`;
      console.error(errorMsg);
      console.error("CSV rows (first 5):", csvRows.slice(0, 6));
      throw new Error(errorMsg);
    }
    
    const csvContent = csvRows.join("\n");
    
    return new NextResponse(csvContent, {
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
    
  } catch (error) {
    console.error("Error generating live snapshot CSV:", error);
    
    // Return minimal CSV with error indication
    const errorCsv = "wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated\n";
    
    return new NextResponse(errorCsv, {
      status: 500,
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  }
}
