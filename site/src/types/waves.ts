/**
 * WAVES Portfolio Data Types
 */

import type { DataState, ConfidenceLevel } from "./market";

export interface WaveMetrics {
  wave_id: string;
  display_name: string;
  
  // Performance metrics
  todayReturn: number;
  todayReturnVsBenchmark: number;
  weekReturn: number;
  monthReturn: number;
  ytdReturn: number;
  
  // Alpha and risk metrics
  alpha: number;
  beta: number;
  sharpeRatio: number;
  maxDrawdown: number;
  
  // Exposure metrics
  cashPosition: number;
  equityExposure: number;
  vixLadderExposure: number;
  
  // NAV and portfolio state
  nav: number;
  navChange: number;
  navChangePercent: number;
  
  // Alerts
  alerts: WaveAlert[];
  
  // Data quality
  isSynthetic: boolean;
  lastUpdate: string;
}

export interface WaveAlert {
  type: "beta_drift" | "turnover" | "drawdown" | "exposure" | "data_quality";
  severity: "low" | "medium" | "high";
  message: string;
  timestamp: string;
}

export interface WavesData {
  timestamp: string;
  dataState: DataState;
  asOf: string;
  waves: WaveMetrics[];
  alerts: WaveAlert[];
  governance: {
    dataState: DataState;
    confidenceLevel: ConfidenceLevel;
    syntheticDataPercentage: number;
    lastUpdate: string;
  };
}
