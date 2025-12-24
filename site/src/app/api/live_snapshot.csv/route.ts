/**
 * Live Snapshot CSV Endpoint
 * 
 * Serves a CSV file with live wave performance metrics.
 * Data is computed from wave_history.csv in the repository.
 * 
 * CSV Schema:
 * wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated
 */

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

interface LiveSnapshotRow {
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
    // Simple CSV parser - wave_history.csv uses simple comma separation without quoted fields
    const values = line.split(',');
    return {
      wave_id: values[0]?.trim() || '',
      display_name: values[1]?.trim() || '',
      date: values[2]?.trim() || '',
      portfolio_return: parseFloat(values[3] || '0'),
      benchmark_return: parseFloat(values[4] || '0'),
      is_synthetic: values[5]?.trim() === 'True',
    };
  }).filter(row => row.wave_id && !isNaN(row.portfolio_return));
}

function calculateCumulativeReturn(returns: number[]): number {
  return returns.reduce((cum, ret) => cum * (1 + ret), 1) - 1;
}

function getTradingDaysInYear(year: number): number {
  // Approximate 252 trading days per year
  return 252;
}

function computeYTDReturn(rows: WaveHistoryRow[], latestDate: Date): number {
  const yearStart = new Date(latestDate.getFullYear(), 0, 1);
  const ytdRows = rows.filter(r => {
    const rowDate = new Date(r.date);
    return rowDate >= yearStart && rowDate <= latestDate;
  });
  
  if (ytdRows.length === 0) return 0;
  
  const returns = ytdRows.map(r => r.portfolio_return);
  return calculateCumulativeReturn(returns);
}

function compute30DReturn(rows: WaveHistoryRow[]): number {
  // Take last 30 trading days (or available data if less)
  const last30 = rows.slice(0, Math.min(30, rows.length));
  if (last30.length === 0) return 0;
  
  const returns = last30.map(r => r.portfolio_return);
  return calculateCumulativeReturn(returns);
}

function compute1DReturn(rows: WaveHistoryRow[]): number {
  if (rows.length === 0) return 0;
  return rows[0].portfolio_return;
}

async function generateLiveSnapshot(): Promise<LiveSnapshotRow[]> {
  try {
    // Read wave_history.csv from repository root
    const repoRoot = path.join(process.cwd(), "..");
    const historyPath = path.join(repoRoot, "wave_history.csv");
    
    const csvText = await fs.readFile(historyPath, "utf-8");
    const allRows = parseCSV(csvText);
    
    if (allRows.length === 0) {
      throw new Error("No wave history data found");
    }
    
    // Group by wave_id
    const waveGroups = new Map<string, WaveHistoryRow[]>();
    for (const row of allRows) {
      if (!waveGroups.has(row.wave_id)) {
        waveGroups.set(row.wave_id, []);
      }
      waveGroups.get(row.wave_id)!.push(row);
    }
    
    const snapshot: LiveSnapshotRow[] = [];
    
    for (const [wave_id, rows] of waveGroups) {
      // Sort by date descending (most recent first)
      rows.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
      
      const latest = rows[0];
      const latestDate = new Date(latest.date);
      
      // Compute performance metrics
      const performance_1d = compute1DReturn(rows);
      const performance_30d = compute30DReturn(rows);
      const performance_ytd = computeYTDReturn(rows, latestDate);
      
      // Determine status
      const status = latest.is_synthetic ? "DEMO" : "ACTIVE";
      
      snapshot.push({
        wave_id,
        wave_name: latest.display_name,
        status,
        performance_1d: performance_1d * 100, // Convert to percentage
        performance_30d: performance_30d * 100,
        performance_ytd: performance_ytd * 100,
        last_updated: latest.date,
      });
    }
    
    // Deduplicate by wave_id (keep most recent based on last_updated)
    const deduped = new Map<string, LiveSnapshotRow>();
    for (const row of snapshot) {
      const existing = deduped.get(row.wave_id);
      if (!existing || new Date(row.last_updated) > new Date(existing.last_updated)) {
        deduped.set(row.wave_id, row);
      }
    }
    
    return Array.from(deduped.values()).sort((a, b) => a.wave_name.localeCompare(b.wave_name));
    
  } catch (error) {
    console.error("Error generating live snapshot:", error);
    // Return demo snapshot on error
    return generateDemoSnapshot();
  }
}

function generateDemoSnapshot(): LiveSnapshotRow[] {
  // Deterministic demo data with seeded values
  const demoWaves = [
    { id: "sp500_wave", name: "S&P 500 Wave", p1d: 0.42, p30d: 3.28, pytd: 18.45 },
    { id: "growth_wave", name: "Growth Wave", p1d: 0.68, p30d: 4.52, pytd: 22.18 },
    { id: "income_wave", name: "Income Wave", p1d: 0.15, p30d: 1.24, pytd: 8.67 },
    { id: "small_cap_growth_wave", name: "Small Cap Growth Wave", p1d: 0.91, p30d: 5.73, pytd: 24.92 },
  ];
  
  const now = new Date().toISOString().split('T')[0];
  
  return demoWaves.map(w => ({
    wave_id: w.id,
    wave_name: w.name,
    status: "DEMO",
    performance_1d: w.p1d,
    performance_30d: w.p30d,
    performance_ytd: w.pytd,
    last_updated: now,
  }));
}

function escapeCSVField(field: string): string {
  // Escape fields containing commas, quotes, or newlines by wrapping in quotes
  if (field.includes(',') || field.includes('"') || field.includes('\n')) {
    return `"${field.replace(/"/g, '""')}"`;
  }
  return field;
}

function snapshotToCSV(snapshot: LiveSnapshotRow[]): string {
  const headers = "wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated";
  const rows = snapshot.map(row => {
    const fields = [
      escapeCSVField(row.wave_id),
      escapeCSVField(row.wave_name),
      escapeCSVField(row.status),
      row.performance_1d.toFixed(2),
      row.performance_30d.toFixed(2),
      row.performance_ytd.toFixed(2),
      row.last_updated,
    ];
    return fields.join(',');
  });
  
  return [headers, ...rows].join('\n');
}

export async function GET() {
  try {
    const snapshot = await generateLiveSnapshot();
    const csvContent = snapshotToCSV(snapshot);
    
    return new NextResponse(csvContent, {
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Cache-Control": "no-store",
        "Content-Disposition": "inline; filename=live_snapshot.csv",
      },
    });
  } catch (error) {
    console.error("Failed to generate live snapshot CSV:", error);
    
    // Fallback to demo snapshot
    const demoSnapshot = generateDemoSnapshot();
    const csvContent = snapshotToCSV(demoSnapshot);
    
    return new NextResponse(csvContent, {
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Cache-Control": "no-store",
        "Content-Disposition": "inline; filename=live_snapshot.csv",
      },
    });
  }
}
