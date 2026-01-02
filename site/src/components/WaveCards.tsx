"use client";

import { useEffect, useMemo, useState } from "react";

interface WaveCard {
  id: number;
  name: string;
  description: string;
  performance?: string;
  status?: string;
  performance1d?: string;
  performance30d?: string;
  performanceYtd?: string;
}

interface LiveWaveData {
  wave_id: string;
  wave_name: string;
  status: string;
  performance_1d: string;
  performance_30d: string;
  performance_ytd: string;
  last_updated: string;
}

/**
 * Robust CSV line parser:
 * - Handles quoted fields with commas
 * - Handles escaped quotes inside quotes: "" -> "
 */
function parseCSVLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (ch === '"') {
      // If we are inside quotes and the next char is also a quote, it's an escaped quote
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++; // skip next quote
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }

    cur += ch;
  }

  out.push(cur);
  return out.map((v) => v.trim());
}

function normalizeDash(value: string | undefined | null) {
  const v = (value ?? "").trim();
  return v === "" ? "—" : v;
}

function safeStatus(value: string | undefined | null) {
  const v = (value ?? "").trim();
  return v === "" ? "NO DATA" : v;
}

/**
 * Fallback 28 waves (used ONLY if CSV fetch fails).
 * This guarantees UI still shows 28 cards with "—" values.
 * If you want, we can auto-generate this list once from your CSV
 * and keep it aligned.
 */
const FALLBACK_WAVES: Array<{ wave_id: string; wave_name: string }> = [
  { wave_id: "ai_cloud_megacap_wave", wave_name: "AI & Cloud MegaCap Wave" },
  { wave_id: "clean_transit_infrastructure_wave", wave_name: "Clean Transit-Infrastructure Wave" },
  { wave_id: "crypto_ai_growth_wave", wave_name: "Crypto AI Growth Wave" },
  { wave_id: "crypto_broad_growth_wave", wave_name: "Crypto Broad Growth Wave" },
  { wave_id: "crypto_defi_growth_wave", wave_name: "Crypto DeFi Growth Wave" },
  { wave_id: "crypto_income_wave", wave_name: "Crypto Income Wave" },
  { wave_id: "crypto_l2_growth_wave", wave_name: "Crypto L2 Growth Wave" },
  { wave_id: "demas_fund_wave", wave_name: "Demas Fund Wave" },
  { wave_id: "future_energy_ev_wave", wave_name: "Future Energy & EV Wave" },
  { wave_id: "future_power_energy_wave", wave_name: "Future Power & Energy Wave" },
  { wave_id: "gold_wave", wave_name: "Gold Wave" },
  { wave_id: "income_wave", wave_name: "Income Wave" },
  { wave_id: "russell_3000_wave", wave_name: "Russell 3000 Wave" },
  { wave_id: "sp500_wave", wave_name: "S&P 500 Wave" },
  { wave_id: "small_cap_growth_wave", wave_name: "Small Cap Growth Wave" },
  { wave_id: "small_to_mid_cap_growth_wave", wave_name: "Small to Mid Cap Growth Wave" },
  { wave_id: "smartsafe_tax_free_mm_wave", wave_name: "SmartSafe Tax-Free Money Market Wave" },
  { wave_id: "smartsafe_treasury_cash_wave", wave_name: "SmartSafe Treasury Cash Wave" },
  { wave_id: "us_megacap_core_wave", wave_name: "US MegaCap Core Wave" },
  { wave_id: "us_mid_small_growth_semis_wave", wave_name: "US Mid/Small Growth & Semis Wave" },
  { wave_id: "us_small_cap_disruptors_wave", wave_name: "US Small-Cap Disruptors Wave" },
  { wave_id: "vector_muni_ladder_wave", wave_name: "Vector Muni Ladder Wave" },
  { wave_id: "vector_treasury_ladder_wave", wave_name: "Vector Treasury Ladder Wave" },

  // Add the remaining 5 to reach 28 (keep names exactly as you want displayed)
  { wave_id: "us_growth_wave", wave_name: "US Growth Wave" },
  { wave_id: "quantum_computing_wave", wave_name: "Quantum Computing Wave" },
  { wave_id: "ai_automation_wave", wave_name: "AI & Automation Wave" },
  { wave_id: "smart_safe_wave", wave_name: "SmartSafe Wave" },
  { wave_id: "innovation_wave", wave_name: "Innovation Wave" },
];

export default function WaveCards() {
  const [liveData, setLiveData] = useState<LiveWaveData[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchLiveData = async () => {
      try {
        setIsLoading(true);

        const url = `/data/live_snapshot.csv?ts=${Date.now()}`;
        const response = await fetch(url, { cache: "no-store" });

        if (!response.ok) {
          throw new Error(`CSV fetch failed: ${response.status}`);
        }

        const csvText = await response.text();
        const lines = csvText.split(/\r?\n/).filter((l) => l.trim() !== "");

        if (lines.length < 2) {
          throw new Error("CSV file is empty or missing rows");
        }

        // Header check (soft)
        // wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated
        const rows = lines.slice(1);

        const parsed: LiveWaveData[] = rows
          .map((line) => {
            const values = parseCSVLine(line);

            return {
              wave_id: values[0] ?? "",
              wave_name: values[1] ?? "",
              status: values[2] ?? "",
              performance_1d: values[3] ?? "",
              performance_30d: values[4] ?? "",
              performance_ytd: values[5] ?? "",
              last_updated: values[6] ?? "",
            };
          })
          // remove any accidental blank row
          .filter((r) => r.wave_id.trim() !== "" || r.wave_name.trim() !== "");

        if (parsed.length === 0) {
          throw new Error("CSV parsed to 0 rows");
        }

        // De-dupe by wave_id (just in case)
        const seen = new Set<string>();
        const deduped = parsed.filter((r) => {
          const key = r.wave_id.trim() || r.wave_name.trim();
          if (!key) return false;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });

        if (!cancelled) {
          setLiveData(deduped);
          setErrorMessage(null);
          setIsLoading(false);
        }
      } catch (err) {
        console.error("Failed to fetch live snapshot CSV:", err);
        if (!cancelled) {
          setErrorMessage("Live data unavailable. Showing last known / fallback waves.");
          setLiveData(null); // force fallback
          setIsLoading(false);
        }
      }
    };

    fetchLiveData();
    const intervalId = setInterval(fetchLiveData, 60000);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, []);

  // Always render:
  // - If CSV loaded: render its rows
  // - If not: render fallback 28 waves
  const displayWaves: WaveCard[] = useMemo(() => {
    const source =
      liveData && liveData.length > 0
        ? liveData.map((r) => ({
            wave_id: r.wave_id,
            wave_name: r.wave_name,
            status: safeStatus(r.status),
            performance_1d: normalizeDash(r.performance_1d),
            performance_30d: normalizeDash(r.performance_30d),
            performance_ytd: normalizeDash(r.performance_ytd),
            last_updated: r.last_updated || "",
          }))
        : FALLBACK_WAVES.map((w) => ({
            wave_id: w.wave_id,
            wave_name: w.wave_name,
            status: "NO DATA",
            performance_1d: "—",
            performance_30d: "—",
            performance_ytd: "—",
            last_updated: "",
          }));

    return source.map((row, idx) => ({
      id: idx + 1,
      name: row.wave_name || row.wave_id,
      description: `Wave ID: ${row.wave_id}`,
      performance: row.performance_1d,
      performance1d: row.performance_1d,
      performance30d: row.performance_30d,
      performanceYtd: row.performance_ytd,
      status: row.status,
    }));
  }, [liveData]);

  return (
    <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">
            Investment <span className="text-cyan-400">Waves</span>
          </h2>
          <p className="mt-4 text-lg text-gray-400">
            Explore our portfolio of strategic investment waves
          </p>

          {isLoading && <p className="mt-2 text-sm text-gray-500">Loading wave data...</p>}

          {errorMessage && (
            <div className="mt-4 rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3">
              <p className="text-sm text-red-400">{errorMessage}</p>
            </div>
          )}

          {!isLoading && liveData && (
            <p className="mt-2 text-xs text-gray-500">
              Loaded {liveData.length} wave rows from <code>/data/live_snapshot.csv</code>
            </p>
          )}
        </div>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {displayWaves.map((wave) => (
            <div
              key={wave.id}
              className="group rounded-lg border border-gray-800 bg-gray-900/30 p-6 backdrop-blur-sm transition-all hover:border-cyan-500/50 hover:bg-gray-900/50"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-white group-hover:text-cyan-400">
                    {wave.name}
                  </h3>

                  <span
                    className={`mt-1 inline-block rounded-full px-2 py-1 text-xs font-medium ${
                      wave.status === "DEMO"
                        ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
                        : wave.status === "NO DATA"
                        ? "bg-red-500/20 text-red-400 border border-red-500/30"
                        : "bg-green-500/20 text-green-400"
                    }`}
                  >
                    {wave.status}
                  </span>
                </div>

                {/* Always show the metrics block (even if it's "—") */}
                <div className="text-right">
                  <div
                    className={`text-lg font-bold ${
                      wave.performance && wave.performance !== "—" && wave.performance.startsWith("-")
                        ? "text-red-400"
                        : wave.performance && wave.performance !== "—"
                        ? "text-green-400"
                        : "text-gray-400"
                    }`}
                  >
                    {wave.performance ?? "—"}
                  </div>
                  <div className="text-xs text-gray-500">1D</div>

                  <div
                    className={`text-sm font-semibold mt-1 ${
                      wave.performance30d && wave.performance30d !== "—" && wave.performance30d.startsWith("-")
                        ? "text-red-400"
                        : wave.performance30d && wave.performance30d !== "—"
                        ? "text-green-400"
                        : "text-gray-400"
                    }`}
                  >
                    {wave.performance30d ?? "—"}
                  </div>
                  <div className="text-xs text-gray-500">30D</div>

                  <div
                    className={`text-sm font-semibold mt-1 ${
                      wave.performanceYtd && wave.performanceYtd !== "—" && wave.performanceYtd.startsWith("-")
                        ? "text-red-400"
                        : wave.performanceYtd && wave.performanceYtd !== "—"
                        ? "text-green-400"
                        : "text-gray-400"
                    }`}
                  >
                    {wave.performanceYtd ?? "—"}
                  </div>
                  <div className="text-xs text-gray-500">YTD</div>
                </div>
              </div>

              <p className="mt-4 text-sm text-gray-400">{wave.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}