"use client";

import { useEffect, useState } from "react";

interface WaveCard {
  id: number;
  name: string;
  description: string;
  performance?: string;
  status?: string;
}

interface WaveCardsProps {
  waves?: WaveCard[];
}

interface SnapshotData {
  wave_id: string;
  wave_name: string;
  status: string;
  performance_1d: number;
  performance_30d: number;
  performance_ytd: number;
  last_updated: string;
}

export default function WaveCards({ waves }: WaveCardsProps) {
  const [liveWaves, setLiveWaves] = useState<WaveCard[] | null>(null);
  const [dataSource, setDataSource] = useState<"LIVE" | "DEMO">("DEMO");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchWaveData = async () => {
      try {
        // Try external URL first if defined
        const externalUrl = process.env.NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL;
        let csvData: string | null = null;
        let source: "LIVE" | "DEMO" = "DEMO";

        if (externalUrl) {
          try {
            const response = await fetch(externalUrl);
            if (response.ok) {
              csvData = await response.text();
              source = "LIVE";
            }
          } catch (error) {
            console.warn("External CSV URL failed, falling back to internal:", error);
          }
        }

        // Fallback to internal endpoint
        if (!csvData) {
          const response = await fetch("/api/live_snapshot.csv");
          if (response.ok) {
            csvData = await response.text();
          }
        }

        if (csvData) {
          const parsedWaves = parseCSVData(csvData);
          setLiveWaves(parsedWaves);
          setDataSource(source);
        }
      } catch (error) {
        console.error("Failed to fetch wave data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchWaveData();
  }, []);

  function parseCSVData(csvText: string): WaveCard[] {
    const lines = csvText.trim().split('\n');
    const dataLines = lines.slice(1); // Skip header
    
    return dataLines.map((line, index) => {
      const parts = line.split(',');
      const wave_id = parts[0];
      const wave_name = parts[1];
      const status = parts[2];
      const performance_ytd = parseFloat(parts[5]);
      
      return {
        id: index + 1,
        name: wave_name,
        description: getWaveDescription(wave_id),
        performance: formatPerformance(performance_ytd),
        status: status,
      };
    });
  }

  function formatPerformance(value: number): string {
    const percentage = (value * 100).toFixed(2);
    return `${value >= 0 ? '+' : ''}${percentage}%`;
  }

  function getWaveDescription(wave_id: string): string {
    // Map wave_id to institutional descriptions
    const descriptions: Record<string, string> = {
      'ai_cloud_megacap_wave': 'Large-cap technology and cloud infrastructure positioning. Capital Role: Growth allocation (15-25%). Risk Regime: Innovation-driven expansion.',
      'sp500_wave': 'Broad market beta exposure with systematic risk controls. Capital Role: Core foundation (40-60%). Risk Regime: All-weather market participation.',
      'russell_3000_wave': 'Total market exposure across capitalization spectrum. Capital Role: Diversified core (30-50%). Risk Regime: Comprehensive market coverage.',
      'income_wave': 'Dividend sustainability mandate with quality overlay. Capital Role: Income sleeve (15-25%). Risk Regime: Defensive income generation.',
      'small_cap_growth_wave': 'Small-cap growth with momentum and quality factors. Capital Role: Opportunistic allocation (10-20%). Risk Regime: Growth-oriented expansion.',
    };
    
    return descriptions[wave_id] || 'Institutional portfolio strategy with disciplined risk management and systematic execution framework.';
  }

  // Generate institutional-grade waves with unique strategies if not provided
  const defaultWaves: WaveCard[] = [
    {
      id: 1,
      name: "Wave 1 - Core Equity",
      description: "Large-cap quality core portfolio. Capital Role: Portfolio foundation (40-60% allocation). Risk Regime: All-weather positioning. Allocation Purpose: Beta exposure with quality overlay. Portfolio Sleeve: Core/Satellite foundation.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 2,
      name: "Wave 2 - Growth Alpha",
      description: "Momentum-driven growth strategy. Capital Role: Opportunistic allocation (10-20%). Risk Regime: Risk-on positioning. Allocation Purpose: Alpha capture in expansion cycles. Portfolio Sleeve: Satellite growth engine.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 3,
      name: "Wave 3 - Value Recovery",
      description: "Systematic value with mean reversion. Capital Role: Contrarian allocation (10-15%). Risk Regime: Risk-off/recovery phases. Allocation Purpose: Capture mispricing and market dislocations. Portfolio Sleeve: Defensive/Opportunistic blend.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 4,
      name: "Wave 4 - Income Generation",
      description: "Dividend sustainability mandate. Capital Role: Stable income sleeve (15-25%). Risk Regime: Low volatility preference. Allocation Purpose: Cash flow generation and capital preservation. Portfolio Sleeve: Defensive core.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 5,
      name: "Wave 5 - Multi-Factor",
      description: "Value-momentum-quality-volatility blend. Capital Role: Diversified factor exposure (20-30%). Risk Regime: Regime-adaptive positioning. Allocation Purpose: Balanced factor returns across cycles. Portfolio Sleeve: Core diversification.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 6,
      name: "Wave 6 - Sector Rotation",
      description: "Macro-driven sector allocation. Capital Role: Tactical positioning (5-15%). Risk Regime: Cycle-dependent rotation. Allocation Purpose: Economic cycle alpha capture. Portfolio Sleeve: Tactical overlay.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 7,
      name: "Wave 7 - Global Diversification",
      description: "International equity with currency management. Capital Role: Geographic diversification (15-25%). Risk Regime: Global macro-aware. Allocation Purpose: Non-US growth and diversification. Portfolio Sleeve: Geographic expansion.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 8,
      name: "Wave 8 - Small-Mid Cap",
      description: "Size factor with liquidity constraints. Capital Role: Inefficiency capture (10-20%). Risk Regime: Risk-on with higher volatility. Allocation Purpose: Small-cap premium and market inefficiency. Portfolio Sleeve: Opportunistic alpha.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 9,
      name: "Wave 9 - Innovation Thematic",
      description: "Secular technology and innovation trends. Capital Role: Thematic growth (5-15%). Risk Regime: High conviction, growth-oriented. Allocation Purpose: Long-term secular alpha capture. Portfolio Sleeve: Thematic satellite.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 10,
      name: "Wave 10 - Defensive Positioning",
      description: "Low-beta capital preservation. Capital Role: Risk mitigation (10-20%). Risk Regime: Risk-off and volatility spikes. Allocation Purpose: Downside protection and stability. Portfolio Sleeve: Defensive anchor.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 11,
      name: "Wave 11 - ESG Integration",
      description: "Sustainable investing with ESG factors. Capital Role: Values-aligned allocation (10-30%). Risk Regime: Long-term orientation. Allocation Purpose: ESG alpha and impact alignment. Portfolio Sleeve: Core with values overlay.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 12,
      name: "Wave 12 - Volatility Harvesting",
      description: "Options-based volatility strategies. Capital Role: Alternative return stream (5-10%). Risk Regime: Volatility regime-dependent. Allocation Purpose: Uncorrelated alpha and volatility capture. Portfolio Sleeve: Alternative diversifier.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 13,
      name: "Wave 13 - Macro Allocation",
      description: "Multi-asset global macro positioning. Capital Role: Dynamic beta adjustment (10-20%). Risk Regime: All-weather with tactical shifts. Allocation Purpose: Top-down allocation and currency exposure. Portfolio Sleeve: Strategic macro overlay.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 14,
      name: "Wave 14 - Market Neutral",
      description: "Long-short equity with factor neutrality. Capital Role: Absolute return sleeve (5-15%). Risk Regime: Market-neutral positioning. Allocation Purpose: Pure alpha extraction independent of beta. Portfolio Sleeve: Alternative alpha.",
      performance: "+0.00%",
      status: "Active",
    },
    {
      id: 15,
      name: "Wave 15 - Special Situations",
      description: "Event-driven catalyst strategies. Capital Role: Opportunistic events (5-10%). Risk Regime: Event-specific timing. Allocation Purpose: M&A, restructuring, and corporate action alpha. Portfolio Sleeve: Tactical opportunistic.",
      performance: "+0.00%",
      status: "Active",
    },
  ];

  const displayWaves = waves || liveWaves || defaultWaves;

  return (
    <section className="bg-gradient-to-b from-black to-gray-900 py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <h2 className="text-3xl font-bold text-white sm:text-4xl">
              Investment <span className="text-cyan-400">Waves</span>
            </h2>
            <div
              className={`inline-flex items-center gap-1.5 rounded border px-3 py-1 text-xs font-semibold ${
                dataSource === "LIVE"
                  ? "bg-green-500/20 text-green-400 border-green-500/50"
                  : "bg-yellow-500/20 text-yellow-400 border-yellow-500/50"
              }`}
            >
              <span className="animate-pulse">{dataSource === "LIVE" ? "●" : "◐"}</span>
              <span>{dataSource}</span>
            </div>
          </div>
          <p className="mt-4 text-lg text-gray-400">
            {isLoading
              ? "Loading wave performance data..."
              : "Explore our portfolio of strategic investment waves"}
          </p>
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
                  <span className="mt-1 inline-block rounded-full bg-green-500/20 px-2 py-1 text-xs font-medium text-green-400">
                    {wave.status}
                  </span>
                </div>
                {wave.performance && (
                  <div className="text-right">
                    <div className="text-lg font-bold text-green-400">{wave.performance}</div>
                    <div className="text-xs text-gray-500">Performance</div>
                  </div>
                )}
              </div>
              <p className="mt-4 text-sm text-gray-400">{wave.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
