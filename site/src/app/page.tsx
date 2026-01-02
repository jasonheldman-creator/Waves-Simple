"use client";

import React, { useState, useEffect } from "react";

interface WaveSnapshot {
  Wave_ID: string;
  Wave: string;
  Return_1D?: string;
  Return_30D?: string;
  Return_60D?: string;
  Return_365D?: string;
  AsOfUTC?: string;
  DataStatus?: string;
  MissingTickers?: string;
}

interface SnapshotResponse {
  count: number;
  timestamp: string;
  data: WaveSnapshot[];
}

export default function SnapshotConsole() {
  const [snapshot, setSnapshot] = useState<SnapshotResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const fetchSnapshot = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/snapshot");
      const data = await response.json();

      if (response.ok) {
        setSnapshot(data);
        setMessage(null);
      } else {
        setMessage({ type: "error", text: data.message || "Failed to fetch snapshot" });
      }
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "Failed to fetch snapshot",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSnapshot();
  }, []);

  const formatReturn = (val?: string) => {
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
            Live market data snapshot for {snapshot?.count || 28} canonical waves
          </p>
        </div>

        {/* Status Messages */}
        {message && (
          <div
            className={`mb-6 p-4 rounded-lg ${
              message.type === "success"
                ? "bg-green-50 border border-green-200 text-green-800"
                : "bg-red-50 border border-red-200 text-red-800"
            }`}
          >
            <div className="whitespace-pre-wrap">{message.text}</div>
          </div>
        )}

        {/* Controls */}
        <div className="mb-6 flex items-center justify-between bg-white p-4 rounded-lg shadow">
          <div className="text-sm text-gray-600">
            {snapshot && (
              <div>
                <strong>Total Waves:</strong> {snapshot.count} |
                <strong className="ml-2">Last Updated:</strong>{" "}
                {snapshot.data[0]?.AsOfUTC
                  ? new Date(snapshot.data[0].AsOfUTC).toLocaleString()
                  : "—"}
              </div>
            )}
          </div>
          <div className="flex gap-3">
            <button
              onClick={fetchSnapshot}
              disabled={loading}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Loading..." : "Refresh"}
            </button>
            {/* Rebuild functionality disabled - data now sourced from static CSV */}
            {/* <button
              onClick={rebuildSnapshot}
              disabled={rebuilding}
              className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
            >
              {rebuilding ? "Rebuilding..." : "Rebuild Snapshot Now"}
            </button> */}
          </div>
        </div>

        {/* Snapshot Table */}
        {loading && !snapshot ? (
          <div className="text-center py-12">
            <div className="text-gray-600">Loading snapshot data...</div>
          </div>
        ) : snapshot ? (
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
                      60D Return
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      365D Return
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {snapshot.data.map((wave, index) => {
                    const ret1d = wave.Return_1D ? parseFloat(wave.Return_1D) : null;
                    const ret30d = wave.Return_30D ? parseFloat(wave.Return_30D) : null;
                    const ret60d = wave.Return_60D ? parseFloat(wave.Return_60D) : null;
                    const ret365d = wave.Return_365D ? parseFloat(wave.Return_365D) : null;

                    return (
                      <tr
                        key={wave.Wave_ID}
                        className={index % 2 === 0 ? "bg-white" : "bg-gray-50"}
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {wave.Wave}
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
                          {formatReturn(wave.Return_1D)}
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
                          {formatReturn(wave.Return_30D)}
                        </td>
                        <td
                          className={`px-6 py-4 whitespace-nowrap text-sm text-right ${
                            ret60d !== null
                              ? ret60d >= 0
                                ? "text-green-600"
                                : "text-red-600"
                              : "text-gray-400"
                          }`}
                        >
                          {formatReturn(wave.Return_60D)}
                        </td>
                        <td
                          className={`px-6 py-4 whitespace-nowrap text-sm text-right ${
                            ret365d !== null
                              ? ret365d >= 0
                                ? "text-green-600"
                                : "text-red-600"
                              : "text-gray-400"
                          }`}
                        >
                          {formatReturn(wave.Return_365D)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-center">
                          <span
                            className={`px-2 py-1 text-xs font-semibold rounded-full ${
                              wave.DataStatus === "OK"
                                ? "bg-green-100 text-green-800"
                                : "bg-red-100 text-red-800"
                            }`}
                          >
                            {wave.DataStatus || "—"}
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
          <div className="text-center py-12 text-red-600">Failed to load snapshot data</div>
        )}
      </div>
    </div>
  );
}
