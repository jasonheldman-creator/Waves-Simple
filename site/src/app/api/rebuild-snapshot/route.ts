import { NextResponse } from "next/server";
import { Octokit } from "@octokit/rest";

interface WaveWeight {
  wave: string;
  ticker: string;
  weight: number;
}

interface WaveData {
  Wave_ID: string;
  Wave: string;
  Return_1D: number | null;
  Return_30D: number | null;
  Return_60D: number | null;
  Return_365D: number | null;
  AsOfUTC: string;
  DataStatus: string;
  MissingTickers: string;
}

interface TickerPrice {
  date: string;
  price: number;
}

/**
 * Slugify wave name to wave_id
 */
function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

/**
 * Fetch crypto prices from CoinGecko
 */
async function fetchCryptoPrices(ticker: string, days: number = 365): Promise<TickerPrice[]> {
  try {
    // Convert ticker format: BTC-USD -> bitcoin
    const coinId = ticker.replace("-USD", "").toLowerCase();
    const coinMap: Record<string, string> = {
      btc: "bitcoin",
      eth: "ethereum",
      bnb: "binancecoin",
      sol: "solana",
      xrp: "ripple",
      ada: "cardano",
      avax: "avalanche-2",
      dot: "polkadot",
      uni: "uniswap",
      aave: "aave",
      link: "chainlink",
      mkr: "maker",
      crv: "curve-dao-token",
      inj: "injective-protocol",
      snx: "synthetix-network-token",
      comp: "compound-governance-token",
      steth: "staked-ether",
      ldo: "lido-dao",
      cake: "pancakeswap-token",
      matic: "matic-network",
      arb: "arbitrum",
      op: "optimism",
      imx: "immutable-x",
      mnt: "mantle",
      stx: "blockstack",
      near: "near",
      apt: "aptos",
      atom: "cosmos",
      tao: "bittensor",
      render: "render-token",
      fet: "fetch-ai",
      icp: "internet-computer",
      ocean: "ocean-protocol",
      agix: "singularitynet",
    };
    
    const coinGeckoId = coinMap[coinId] || coinId;
    
    const url = `https://api.coingecko.com/api/v3/coins/${coinGeckoId}/market_chart?vs_currency=usd&days=${days}`;
    const response = await fetch(url, {
      headers: {
        "Accept": "application/json",
      },
    });
    
    if (!response.ok) {
      throw new Error(`CoinGecko API error: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    if (!data.prices || data.prices.length === 0) {
      throw new Error("No price data returned");
    }
    
    return data.prices.map(([timestamp, price]: [number, number]) => ({
      date: new Date(timestamp).toISOString().split("T")[0],
      price,
    }));
  } catch (error) {
    console.error(`Failed to fetch crypto prices for ${ticker}:`, error);
    return [];
  }
}

/**
 * Fetch equity prices from Stooq
 */
async function fetchEquityPrices(ticker: string, days: number = 365): Promise<TickerPrice[]> {
  try {
    // Stooq uses format: https://stooq.com/q/d/l/?s=AAPL.US&i=d
    const url = `https://stooq.com/q/d/l/?s=${ticker}.US&i=d`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Stooq API error: ${response.statusText}`);
    }
    
    const csvText = await response.text();
    const lines = csvText.trim().split("\n");
    
    if (lines.length < 2) {
      throw new Error("No data returned from Stooq");
    }
    
    // Parse CSV: Date,Open,High,Low,Close,Volume
    const prices: TickerPrice[] = [];
    for (let i = 1; i < lines.length; i++) {
      const [date, , , , close] = lines[i].split(",");
      if (date && close && !isNaN(parseFloat(close))) {
        prices.push({
          date,
          price: parseFloat(close),
        });
      }
    }
    
    // Return most recent N days
    return prices.slice(0, days).reverse();
  } catch (error) {
    console.error(`Failed to fetch equity prices for ${ticker}:`, error);
    return [];
  }
}

/**
 * Calculate return between two prices
 */
function calculateReturn(oldPrice: number, newPrice: number): number {
  return (newPrice - oldPrice) / oldPrice;
}

/**
 * Compute wave returns for different periods
 */
function computeWaveReturns(
  weights: WaveWeight[],
  pricesCache: Map<string, TickerPrice[]>
): {
  return1D: number | null;
  return30D: number | null;
  return60D: number | null;
  return365D: number | null;
  missingTickers: string[];
} {
  const missingTickers: string[] = [];
  
  // Helper to compute weighted return for a period
  const computePeriodReturn = (daysAgo: number): number | null => {
    let totalWeight = 0;
    let weightedReturn = 0;
    
    for (const { ticker, weight } of weights) {
      const prices = pricesCache.get(ticker);
      if (!prices || prices.length < daysAgo + 1) {
        if (!missingTickers.includes(ticker)) {
          missingTickers.push(ticker);
        }
        continue;
      }
      
      const latestPrice = prices[prices.length - 1].price;
      const oldPrice = prices[prices.length - 1 - daysAgo].price;
      const tickerReturn = calculateReturn(oldPrice, latestPrice);
      
      weightedReturn += tickerReturn * weight;
      totalWeight += weight;
    }
    
    return totalWeight > 0 ? weightedReturn / totalWeight : null;
  };
  
  return {
    return1D: computePeriodReturn(1),
    return30D: computePeriodReturn(30),
    return60D: computePeriodReturn(60),
    return365D: computePeriodReturn(365),
    missingTickers,
  };
}

/**
 * POST /api/rebuild-snapshot
 * 
 * Fetches live market data and rebuilds the snapshot
 * Commits to GitHub only if exactly 28 waves with valid data
 */
export async function POST() {
  try {
    console.log("Starting snapshot rebuild...");
    
    // 1. Load wave_weights.csv from GitHub
    const owner = process.env.GITHUB_REPO?.split("/")[0] || "jasonheldman-creator";
    const repo = process.env.GITHUB_REPO?.split("/")[1] || "Waves-Simple";
    const branch = process.env.GITHUB_BRANCH || "main";
    const token = process.env.GITHUB_TOKEN;
    
    if (!token) {
      throw new Error("GITHUB_TOKEN environment variable not set");
    }
    
    const weightsUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/wave_weights.csv`;
    const weightsResponse = await fetch(weightsUrl);
    
    if (!weightsResponse.ok) {
      throw new Error(`Failed to fetch wave_weights.csv: ${weightsResponse.statusText}`);
    }
    
    const weightsText = await weightsResponse.text();
    const weightsLines = weightsText.trim().split("\n");
    
    // Parse wave weights
    const waveWeights = new Map<string, WaveWeight[]>();
    for (let i = 1; i < weightsLines.length; i++) {
      const [wave, ticker, weightStr] = weightsLines[i].split(",");
      if (!wave || !ticker || !weightStr) continue;
      
      const trimmedWave = wave.trim();
      if (!waveWeights.has(trimmedWave)) {
        waveWeights.set(trimmedWave, []);
      }
      waveWeights.get(trimmedWave)!.push({
        wave: trimmedWave,
        ticker: ticker.trim(),
        weight: parseFloat(weightStr.trim()),
      });
    }
    
    // Get canonical wave list
    const canonicalWaves = Array.from(waveWeights.keys()).sort();
    
    // Validate exactly 28 waves
    if (canonicalWaves.length !== 28) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid wave count",
          message: `Expected exactly 28 waves but found ${canonicalWaves.length} in wave_weights.csv`,
          waves: canonicalWaves,
        },
        { status: 500 }
      );
    }
    
    console.log(`Found ${canonicalWaves.length} canonical waves`);
    
    // 2. Fetch market data for all tickers
    const allTickers = new Set<string>();
    for (const weights of waveWeights.values()) {
      for (const { ticker } of weights) {
        allTickers.add(ticker);
      }
    }
    
    console.log(`Fetching market data for ${allTickers.size} tickers...`);
    
    const pricesCache = new Map<string, TickerPrice[]>();
    const fetchPromises: Promise<void>[] = [];
    
    for (const ticker of allTickers) {
      const promise = (async () => {
        if (ticker.endsWith("-USD")) {
          const prices = await fetchCryptoPrices(ticker);
          if (prices.length > 0) {
            pricesCache.set(ticker, prices);
          }
        } else {
          const prices = await fetchEquityPrices(ticker);
          if (prices.length > 0) {
            pricesCache.set(ticker, prices);
          }
        }
      })();
      fetchPromises.push(promise);
    }
    
    await Promise.all(fetchPromises);
    
    console.log(`Fetched prices for ${pricesCache.size}/${allTickers.size} tickers`);
    
    // 3. Compute wave returns
    const timestamp = new Date().toISOString();
    const waveDataList: WaveData[] = [];
    const failedWaves: string[] = [];
    
    for (const waveName of canonicalWaves) {
      const weights = waveWeights.get(waveName)!;
      const returns = computeWaveReturns(weights, pricesCache);
      
      const waveId = slugify(waveName);
      
      // A wave is considered failed if it has no valid returns
      const hasAnyReturn = returns.return1D !== null || 
                          returns.return30D !== null || 
                          returns.return60D !== null || 
                          returns.return365D !== null;
      
      if (!hasAnyReturn) {
        failedWaves.push(waveName);
      }
      
      waveDataList.push({
        Wave_ID: waveId,
        Wave: waveName,
        Return_1D: returns.return1D,
        Return_30D: returns.return30D,
        Return_60D: returns.return60D,
        Return_365D: returns.return365D,
        AsOfUTC: timestamp,
        DataStatus: hasAnyReturn ? "OK" : "FAILED",
        MissingTickers: returns.missingTickers.join(";"),
      });
    }
    
    // 4. Validate before commit
    const validWaves = waveDataList.filter(w => w.DataStatus === "OK");
    
    if (validWaves.length !== 28) {
      return NextResponse.json(
        {
          success: false,
          error: "Validation failed",
          message: `Only ${validWaves.length}/28 waves have valid data`,
          failedWaves,
          details: waveDataList.map(w => ({
            wave: w.Wave,
            status: w.DataStatus,
            missingTickers: w.MissingTickers,
          })),
        },
        { status: 500 }
      );
    }
    
    // 5. Generate CSV content
    const csvLines = [
      "Wave_ID,Wave,Return_1D,Return_30D,Return_60D,Return_365D,AsOfUTC,DataStatus,MissingTickers",
    ];
    
    for (const wave of waveDataList) {
      const formatReturn = (val: number | null) => val !== null ? val.toFixed(6) : "";
      csvLines.push(
        `${wave.Wave_ID},${wave.Wave},${formatReturn(wave.Return_1D)},${formatReturn(wave.Return_30D)},${formatReturn(wave.Return_60D)},${formatReturn(wave.Return_365D)},${wave.AsOfUTC},${wave.DataStatus},${wave.MissingTickers}`
      );
    }
    
    const csvContent = csvLines.join("\n");
    
    // 6. Commit to GitHub
    const octokit = new Octokit({ auth: token });
    
    // Get current file SHA
    let fileSha: string | undefined;
    try {
      const { data: fileData } = await octokit.repos.getContent({
        owner,
        repo,
        path: "data/live_snapshot.csv",
        ref: branch,
      });
      
      if ("sha" in fileData) {
        fileSha = fileData.sha;
      }
    } catch {
      // File doesn't exist yet, that's ok
      console.log("File doesn't exist yet, will create it");
    }
    
    // Commit the file
    const commitMessage = `Live snapshot update: 28 waves @ ${timestamp}`;
    await octokit.repos.createOrUpdateFileContents({
      owner,
      repo,
      path: "data/live_snapshot.csv",
      message: commitMessage,
      content: Buffer.from(csvContent).toString("base64"),
      branch,
      sha: fileSha,
    });
    
    console.log("Snapshot committed successfully");
    
    return NextResponse.json({
      success: true,
      message: "Snapshot rebuilt and committed successfully",
      timestamp,
      waveCount: 28,
      commit: commitMessage,
      summary: {
        totalWaves: waveDataList.length,
        validWaves: validWaves.length,
        tickersFetched: pricesCache.size,
        totalTickers: allTickers.size,
      },
    });
    
  } catch (error) {
    console.error("Error rebuilding snapshot:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to rebuild snapshot",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
