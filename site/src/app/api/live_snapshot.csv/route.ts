import { NextResponse } from "next/server";

/**
 * GET /api/live_snapshot.csv
 * 
 * Returns deterministic demo CSV data for wave performance.
 * This endpoint always returns DEMO status with fixed placeholder values.
 * No real performance calculation is performed.
 */
export async function GET() {
  // Deterministic demo data - always returns the same values
  const csvContent = `wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated
wave_1,Core Equity,DEMO,1.23%,2.34%,5.67%,2025-12-24T12:00:00Z
wave_2,Growth Alpha,DEMO,0.95%,3.10%,8.45%,2025-12-24T12:00:00Z
wave_3,Value Recovery,DEMO,0.78%,1.85%,4.22%,2025-12-24T12:00:00Z
wave_4,Income Generation,DEMO,0.52%,1.45%,3.18%,2025-12-24T12:00:00Z
wave_5,Multi-Factor,DEMO,1.05%,2.67%,6.33%,2025-12-24T12:00:00Z
wave_6,Sector Rotation,DEMO,1.42%,3.55%,7.89%,2025-12-24T12:00:00Z
wave_7,Global Diversification,DEMO,0.88%,2.12%,5.45%,2025-12-24T12:00:00Z
wave_8,Small-Mid Cap,DEMO,1.67%,4.23%,9.78%,2025-12-24T12:00:00Z
wave_9,Innovation Thematic,DEMO,2.15%,5.67%,12.34%,2025-12-24T12:00:00Z
wave_10,Defensive Positioning,DEMO,0.45%,1.12%,2.88%,2025-12-24T12:00:00Z
wave_11,ESG Integration,DEMO,0.92%,2.45%,5.89%,2025-12-24T12:00:00Z
wave_12,Volatility Harvesting,DEMO,0.67%,1.78%,4.56%,2025-12-24T12:00:00Z
wave_13,Macro Allocation,DEMO,1.11%,2.89%,6.77%,2025-12-24T12:00:00Z
wave_14,Market Neutral,DEMO,0.55%,1.34%,3.45%,2025-12-24T12:00:00Z
wave_15,Special Situations,DEMO,1.88%,4.56%,10.23%,2025-12-24T12:00:00Z`;

  return new NextResponse(csvContent, {
    headers: {
      "Content-Type": "text/csv",
      "Cache-Control": "public, max-age=60, s-maxage=60",
    },
  });
}
