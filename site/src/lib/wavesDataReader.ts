/**
 * WAVES Data Reader
 * Reads wave_history.csv and computes metrics
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
  const lines = csvText.trim().split("\n");

  return lines
    .slice(1)
    .map((line) => {
      const values = line.split(",");
      return {
        wave_id: values[0],
        display_name: values[1],
        date: values[2],
        portfolio_return: parseFloat(values[3]),
        benchmark_return: parseFloat(values[4]),
        is_synthetic: values[5] === "True",
      };
    })
    .filter((row) => !isNaN(row.portfolio_return));
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
    cumValue *= 1 + ret;
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

export async function readWavesData(): Promise<{
  waves: WaveMetrics[];
  dataState: DataState;
  syntheticPercentage: number;
}> {
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
      const isSynthetic = waveRows.some((r) => r.is_synthetic);

      totalWaves++;
      if (isSynthetic) syntheticWaves++;

      const recentReturns = waveRows.slice(0, 90);
      const portfolioReturns = recentReturns.map((r) => r.portfolio_return);
      const benchmarkReturns = recentReturns.map((r) => r.benchmark_return);

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
