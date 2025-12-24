"use client";

import { useEffect, useState } from "react";
import type { DataState } from "@/types/market";

export default function DataStateBadge() {
  const [dataState, setDataState] = useState<DataState>("FALLBACK");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchDataState = async () => {
      try {
        const response = await fetch("/api/market");
        if (response.ok) {
          const data = await response.json();
          setDataState(data.governance.dataState);
        }
      } catch (error) {
        console.error("Failed to fetch data state:", error);
        setDataState("FALLBACK");
      } finally {
        setIsLoading(false);
      }
    };

    fetchDataState();
    // Refresh every 60 seconds
    const interval = setInterval(fetchDataState, 60000);
    return () => clearInterval(interval);
  }, []);

  const getStateColor = (state: DataState) => {
    switch (state) {
      case "LIVE":
        return "bg-green-500/20 text-green-400 border-green-500/50";
      case "SNAPSHOT":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/50";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/50";
    }
  };

  const getStateIcon = (state: DataState) => {
    switch (state) {
      case "LIVE":
        return "●"; // Dot for live
      case "SNAPSHOT":
        return "◐"; // Half-dot for snapshot
      default:
        return "○"; // Empty dot for fallback
    }
  };

  const getStateMessage = (state: DataState) => {
    switch (state) {
      case "LIVE":
        return null; // No message when live
      case "SNAPSHOT":
        return "Using last validated snapshot. Live feed delayed.";
      default:
        return "Using last validated snapshot. Live feed delayed.";
    }
  };

  if (isLoading) {
    return null;
  }

  const message = getStateMessage(dataState);

  return (
    <div className="border-b border-gray-800 bg-black">
      <div className="mx-auto max-w-7xl px-4 py-1.5">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Data State:</span>
            <div
              className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-semibold ${getStateColor(dataState)}`}
            >
              <span className="animate-pulse">{getStateIcon(dataState)}</span>
              <span>{dataState}</span>
            </div>
          </div>
          {message && (
            <p className="text-xs text-gray-500 hidden sm:block">{message}</p>
          )}
        </div>
      </div>
    </div>
  );
}
