/**
 * WAVES Data Reader
 * Reads wave_history.csv and computes metrics
 * Or fetches from NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL if configured
 */

import fs from "fs/promises";
import path from "path";
import type { WaveMetrics, WaveAlert } from "@/types/waves";
import type { DataState } from "@/types/market";

interface WaveHistoryRow {
  wave_id: string;
  display_name: string;
  date: string;
  portfolio_return: number;
  benchmark_return: number;
  is_synthetic: boolean;
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

function calculateAlpha(portfolioReturns: number[], benchmarkReturns: number[]): number {
  const portfolioCum = calculateCumulativeReturn(portfolioReturns);
  const benchmarkCum = calculateCumulativeReturn(benchmarkReturns);
  return portfolioCum - benchmarkCum;
}

function calculateBeta(portfolioReturns: number[], benchmarkReturns: number[]): number {
  if (portfolioReturns.length === 0 || benchmarkReturns.length === 0) return 1.0;
  
  const avgPortfolio = portfolioReturns.reduce((sum, r) => sum + r, 0) / portfolioReturns.length;
  const avgBenchmark = benchmarkReturns.reduce((sum, r) => sum + r, 0) / benchmarkReturns.length;
  
  let covariance = 0;
  let benchmarkVariance = 0;
  
  for (let i = 0; i < portfolioReturns.length; i++) {
    const pDiff = portfolioReturns[i] - avgPortfolio;
    const bDiff = benchmarkReturns[i] - avgBenchmark;
    covariance += pDiff * bDiff;
    benchmarkVariance += bDiff * bDiff;
  }
  
  if (benchmarkVariance === 0) return 1.0;
  return covariance / benchmarkVariance;
}

function calculateSharpeRatio(returns: number[], riskFreeRate: number = 0.04): number {
  if (returns.length === 0) return 0;
  
  const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance) * Math.sqrt(252);
  
  const annualizedReturn = Math.pow(1 + avgReturn, 252) - 1;
  const excessReturn = annualizedReturn - riskFreeRate;
  
  return stdDev === 0 ? 0 : excessReturn / stdDev;
}

function calculateMaxDrawdown(returns: number[]): number {
  if (returns.length === 0) return 0;
  
  let peak = 1;
  let maxDrawdown = 0;
  let cumValue = 1;
  
  for (const ret of returns) {
    cumValue *= (1 + ret);
    if (cumValue > peak) {
      peak = cumValue;
    }
    const drawdown = (peak - cumValue) / peak;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }
  
  return maxDrawdown;
}

function generateAlerts(
  wave_id: string,
  beta: number,
  maxDrawdown: number,
  isSynthetic: boolean
): WaveAlert[] {
  const alerts: WaveAlert[] = [];
  const now = new Date().toISOString();
  
  if (Math.abs(beta - 0.9) > 0.15) {
    alerts.push({
      type: "beta_drift",
      severity: Math.abs(beta - 0.9) > 0.25 ? "high" : "medium",
      message: `Beta ${beta.toFixed(2)} deviates from target 0.90`,
      timestamp: now,
    });
  }
  
  if (maxDrawdown > 0.15) {
    alerts.push({
      type: "drawdown",
      severity: maxDrawdown > 0.25 ? "high" : "medium",
      message: `Max drawdown ${(maxDrawdown * 100).toFixed(1)}% exceeds threshold`,
      timestamp: now,
    });
  }
  
  if (isSynthetic) {
    alerts.push({
      type: "data_quality",
      severity: "low",
      message: "Using synthetic/simulated data",
      timestamp: now,
    });
  }
  
  return alerts;
}

/**
 * Fetch and parse live snapshot CSV from URL
 * CSV format: wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated
 */
async function fetchLiveSnapshotCSV(url: string): Promise<{
  waves: WaveMetrics[];
  dataState: DataState;
  syntheticPercentage: number;
}> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch CSV from ${url}: ${response.statusText}`);
  }
  
  const csvText = await response.text();
  const lines = csvText.trim().split('\n');
  
  if (lines.length < 2) {
    throw new Error("CSV is empty or has no data rows");
  }
  
  const waves: WaveMetrics[] = [];
  let syntheticCount = 0;
  
  // Parse CSV rows (skip header)
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    const values = line.split(',');
    
    if (values.length < 7) continue;
    
    const wave_id = values[0];
    const wave_name = values[1].replace(/^"|"$/g, ''); // Remove quotes
    const status = values[2];
    const performance_1d = parseFloat(values[3]);
    const performance_30d = parseFloat(values[4]);
    const performance_ytd = parseFloat(values[5]);
    const last_updated = values[6];
    
    const isSynthetic = status === "DEMO";
    if (isSynthetic) syntheticCount++;
    
    const alerts: WaveAlert[] = [];
    if (isSynthetic) {
      alerts.push({
        type: "data_quality",
        severity: "low",
        message: "Using synthetic/simulated data",
        timestamp: new Date().toISOString(),
      });
    }
    
    waves.push({
      wave_id,
      display_name: wave_name,
      todayReturn: performance_1d,
      todayReturnVsBenchmark: 0, // Not available in CSV
      weekReturn: 0, // Not available in CSV
      monthReturn: performance_30d,
      ytdReturn: performance_ytd,
      alpha: 0, // Not available in CSV
      beta: 0.9, // Default assumption
      sharpeRatio: 0, // Not available in CSV
      maxDrawdown: 0, // Not available in CSV
      cashPosition: 10, // Default assumption
      equityExposure: 90, // Default assumption
      vixLadderExposure: 5, // Default assumption
      nav: 100, // Default assumption
      navChange: 0, // Not available in CSV
      navChangePercent: performance_1d,
      alerts,
      isSynthetic,
      lastUpdate: last_updated,
    });
  }
  
  const syntheticPercentage = waves.length > 0 ? (syntheticCount / waves.length) * 100 : 0;
  const dataState: DataState = syntheticPercentage === 100 ? "SNAPSHOT" : syntheticPercentage > 0 ? "SNAPSHOT" : "LIVE";
  
  return {
    waves,
    dataState,
    syntheticPercentage,
  };
}

export async function readWavesData(): Promise<{
  waves: WaveMetrics[];
  dataState: DataState;
  syntheticPercentage: number;
}> {
  // Check if NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL is set (external URL)
  const liveSnapshotUrl = process.env.NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL;
  
  if (liveSnapshotUrl) {
    try {
      console.log(`Fetching live snapshot from: ${liveSnapshotUrl}`);
      return await fetchLiveSnapshotCSV(liveSnapshotUrl);
    } catch (error) {
      console.error(`Failed to fetch from ${liveSnapshotUrl}, falling back to local file:`, error);
      // Fall through to local file read
    }
  }
  
  // Fallback: read from local wave_history.csv
  try {
    const repoRoot = path.join(process.cwd(), "..");
    const historyPath = path.join(repoRoot, "wave_history.csv");
    
    const csvText = await fs.readFile(historyPath, "utf-8");
    const rows = parseCSV(csvText);
    
    if (rows.length === 0) {
      throw new Error("No wave history data found");
    }
    
    const waveGroups = new Map<string, WaveHistoryRow[]>();
    for (const row of rows) {
      if (!waveGroups.has(row.wave_id)) {
        waveGroups.set(row.wave_id, []);
      }
      waveGroups.get(row.wave_id)!.push(row);
    }
    
    const waves: WaveMetrics[] = [];
    let totalWaves = 0;
    let syntheticWaves = 0;
    
    for (const [wave_id, waveRows] of waveGroups) {
      waveRows.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
      
      const latest = waveRows[0];
      const isSynthetic = waveRows.some(r => r.is_synthetic);
      
      totalWaves++;
      if (isSynthetic) syntheticWaves++;
      
      const recentReturns = waveRows.slice(0, 90);
      const portfolioReturns = recentReturns.map(r => r.portfolio_return);
      const benchmarkReturns = recentReturns.map(r => r.benchmark_return);
      
      const todayReturn = latest.portfolio_return;
      const todayBenchmarkReturn = latest.benchmark_return;
      const todayReturnVsBenchmark = todayReturn - todayBenchmarkReturn;
      
      const weekReturns = portfolioReturns.slice(0, 5);
      const monthReturns = portfolioReturns.slice(0, 21);
      const ytdReturns = portfolioReturns;
      
      const alpha = calculateAlpha(portfolioReturns, benchmarkReturns);
      const beta = calculateBeta(portfolioReturns, benchmarkReturns);
      const sharpeRatio = calculateSharpeRatio(portfolioReturns);
      const maxDrawdown = calculateMaxDrawdown(portfolioReturns);
      
      const nav = 100 * (1 + calculateCumulativeReturn(portfolioReturns));
      const navChange = nav * todayReturn;
      const navChangePercent = todayReturn;
      
      const alerts = generateAlerts(wave_id, beta, maxDrawdown, isSynthetic);
      
      waves.push({
        wave_id,
        display_name: latest.display_name,
        todayReturn: todayReturn * 100,
        todayReturnVsBenchmark: todayReturnVsBenchmark * 100,
        weekReturn: calculateCumulativeReturn(weekReturns) * 100,
        monthReturn: calculateCumulativeReturn(monthReturns) * 100,
        ytdReturn: calculateCumulativeReturn(ytdReturns) * 100,
        alpha: alpha * 100,
        beta,
        sharpeRatio,
        maxDrawdown,
        cashPosition: 10,
        equityExposure: 90,
        vixLadderExposure: 5,
        nav,
        navChange,
        navChangePercent: navChangePercent * 100,
        alerts,
        isSynthetic,
        lastUpdate: latest.date,
      });
    }
    
    const syntheticPercentage = (syntheticWaves / totalWaves) * 100;
    let dataState: DataState = "LIVE";
    
    if (syntheticPercentage === 100) {
      dataState = "SNAPSHOT";
    } else if (syntheticPercentage > 0) {
      dataState = "SNAPSHOT";
    }
    
    return {
      waves,
      dataState,
      syntheticPercentage,
    };
    
  } catch (error) {
    console.error("Error reading WAVES data:", error);
    throw error;
  }
}
