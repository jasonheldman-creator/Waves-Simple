"use client";

import React, { useState, useEffect } from "react";

interface WaveData {
  wave_id: string;
  wave_name: string;
  status: string;
  performance_1d: string;
  performance_30d: string;
  performance_ytd: string;
  last_updated: string;
}

export default function SnapshotConsole() {
  const [waves, setWaves] = useState<WaveData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("—");

  const fetchSnapshot = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch("/data/live_snapshot.csv", {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error("Failed to fetch snapshot data");
      }

      const csvText = await response.text();
      const lines = csvText.trim().split("\n");
      
      if (lines.length < 2) {
        throw new Error("Live data unavailable");
      }

      // Parse CSV manually with proper handling of empty fields
      const lines2 = lines.slice(1); // Skip header
      
      const parsedWaves: WaveData[] = lines2.map((line) => {
        // Split by comma - this assumes no commas within quoted strings
        // For the current CSV structure with empty fields, we need to handle consecutive commas
        const values = line.split(",").map(v => v.trim());
        
        return {
          wave_id: values[0] || "",
          wave_name: values[1] || "",
          status: values[2] || "",
          performance_1d: values[3] || "",
          performance_30d: values[4] || "",
          performance_ytd: values[5] || "",
          last_updated: values[6] || "",
        };
      });

      setWaves(parsedWaves);
      
      // Derive last updated from the first wave's last_updated field
      if (parsedWaves.length > 0 && parsedWaves[0].last_updated) {
        try {
          const date = new Date(parsedWaves[0].last_updated);
          setLastUpdated(date.toLocaleString());
        } catch {
          setLastUpdated("—");
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load snapshot");
      setWaves([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSnapshot();
  }, []);

  const formatReturn = (val: string) => {
    if (!val || val === "") return "—";
    const num = parseFloat(val);
    if (isNaN(num)) return "—";
    return `${(num * 100).toFixed(2)}%`;
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            WAVES Intelligence™ — Snapshot Console
          </h1>
          <p className="text-gray-600">
            Live market data snapshot for 28 canonical waves
          </p>
        </div>

        {/* Controls */}
        <div className="mb-6 flex items-center justify-between bg-white p-4 rounded-lg shadow">
          <div className="text-sm text-gray-600">
            <div>
              <strong>Total Waves:</strong> 28 |
              <strong className="ml-2">Last Updated:</strong> {lastUpdated}
            </div>
          </div>
          <div className="flex gap-3">
            <button
              onClick={fetchSnapshot}
              disabled={loading}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Loading..." : "Refresh"}
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 rounded-lg bg-red-50 border border-red-200 text-red-800">
            {error}
          </div>
        )}

        {/* Snapshot Table */}
        {loading && waves.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-gray-600">Loading snapshot data...</div>
          </div>
        ) : waves.length > 0 ? (
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Wave
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      1D Return
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      30D Return
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      YTD Return
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {waves.map((wave, index) => {
                    const ret1d = wave.performance_1d ? parseFloat(wave.performance_1d) : null;
                    const ret30d = wave.performance_30d ? parseFloat(wave.performance_30d) : null;
                    const retYtd = wave.performance_ytd ? parseFloat(wave.performance_ytd) : null;

                    return (
                      <tr
                        key={wave.wave_id}
                        className={index % 2 === 0 ? "bg-white" : "bg-gray-50"}
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {wave.wave_name}
                        </td>
                        <td
                          className={`px-6 py-4 whitespace-nowrap text-sm text-right ${
                            ret1d !== null
                              ? ret1d >= 0
                                ? "text-green-600"
                                : "text-red-600"
                              : "text-gray-400"
                          }`}
                        >
                          {formatReturn(wave.performance_1d)}
                        </td>
                        <td
                          className={`px-6 py-4 whitespace-nowrap text-sm text-right ${
                            ret30d !== null
                              ? ret30d >= 0
                                ? "text-green-600"
                                : "text-red-600"
                              : "text-gray-400"
                          }`}
                        >
                          {formatReturn(wave.performance_30d)}
                        </td>
                        <td
                          className={`px-6 py-4 whitespace-nowrap text-sm text-right ${
                            retYtd !== null
                              ? retYtd >= 0
                                ? "text-green-600"
                                : "text-red-600"
                              : "text-gray-400"
                          }`}
                        >
                          {formatReturn(wave.performance_ytd)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-center">
                          <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                            wave.status === "LIVE" 
                              ? "bg-green-100 text-green-800"
                              : wave.status === "FAILED" || wave.status === "ERROR"
                              ? "bg-red-100 text-red-800"
                              : "bg-gray-100 text-gray-800"
                          }`}>
                            {wave.status}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="text-center py-12 text-gray-600">Live data unavailable</div>
        )}
      </div>
    </div>
  );
}
