import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/snapshot
 * 
 * Returns the live snapshot CSV from data/live_snapshot.csv
 * Supports optional ?format=csv query parameter to force download
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const format = searchParams.get("format");
    
    // Fetch from GitHub raw URL
    const owner = process.env.GITHUB_REPO?.split("/")[0] || "jasonheldman-creator";
    const repo = process.env.GITHUB_REPO?.split("/")[1] || "Waves-Simple";
    const branch = process.env.GITHUB_BRANCH || "main";
    
    const rawUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/data/live_snapshot.csv`;
    
    const response = await fetch(rawUrl, {
      headers: {
        "Cache-Control": "no-cache",
      },
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch snapshot: ${response.statusText}`);
    }
    
    const csvContent = await response.text();
    
    // Count data rows (excluding header)
    const lines = csvContent.trim().split("\n");
    const dataRowCount = lines.length - 1;
    
    // Validate exactly 28 waves
    if (dataRowCount !== 28) {
      return NextResponse.json(
        { 
          error: "Invalid snapshot", 
          message: `Expected 28 waves but found ${dataRowCount}`,
          details: "The snapshot file is corrupted or incomplete"
        },
        { status: 500 }
      );
    }
    
    // Return CSV content
    const headers: HeadersInit = {
      "Content-Type": format === "csv" ? "text/csv; charset=utf-8" : "application/json",
      "Cache-Control": "no-store",
    };
    
    if (format === "csv") {
      headers["Content-Disposition"] = 'attachment; filename="live_snapshot.csv"';
      return new NextResponse(csvContent, { headers });
    }
    
    // Parse CSV and return as JSON
    const [headerLine, ...dataLines] = lines;
    const headers_arr = headerLine.split(",");
    
    const data = dataLines.map(line => {
      const values = line.split(",");
      const row: Record<string, string> = {};
      headers_arr.forEach((header, i) => {
        row[header] = values[i] || "";
      });
      return row;
    });
    
    return NextResponse.json({
      count: data.length,
      timestamp: new Date().toISOString(),
      data,
    }, { headers: { "Cache-Control": "no-store" } });
    
  } catch (error) {
    console.error("Error fetching snapshot:", error);
    return NextResponse.json(
      { 
        error: "Failed to fetch snapshot",
        message: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}
