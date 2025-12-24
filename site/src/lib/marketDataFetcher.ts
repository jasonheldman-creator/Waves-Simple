/**
 * Market Data Fetcher using Stooq CSV endpoints
 * Stooq provides free market data without requiring API keys
 */

import type { MarketSymbol, DataState } from "@/types/market";

interface StooqDataPoint {
  symbol: string;
  date: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
}

/**
 * Fetch data from Stooq CSV endpoint
 */
async function fetchStooqData(symbol: string): Promise<StooqDataPoint[]> {
  const stooqSymbol = mapToStooqSymbol(symbol);
  const url = `https://stooq.com/q/d/l/?s=${stooqSymbol}&i=d`;
  
  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; WAVES-Intelligence/1.0)',
      },
    });

    if (!response.ok) {
      throw new Error(`Stooq API error: ${response.status}`);
    }

    const csvText = await response.text();
    return parseStooqCSV(csvText, symbol);
  } catch (error) {
    console.error(`Failed to fetch Stooq data for ${symbol}:`, error);
    throw error;
  }
}

function mapToStooqSymbol(symbol: string): string {
  const symbolMap: Record<string, string> = {
    'SPY': 'spy.us',
    'QQQ': 'qqq.us',
    'IWM': 'iwm.us',
    'TLT': 'tlt.us',
    'GLD': 'gld.us',
    'AAPL': 'aapl.us',
    'MSFT': 'msft.us',
    'NVDA': 'nvda.us',
    '^VIX': '^vix',
    '^TNX': '^tnx',
  };
  return symbolMap[symbol] || symbol.toLowerCase();
}

function parseStooqCSV(csvText: string, originalSymbol: string): StooqDataPoint[] {
  const lines = csvText.trim().split('\n');
  const dataLines = lines.slice(1);
  
  return dataLines.map(line => {
    const [date, open, high, low, close, volume] = line.split(',');
    return {
      symbol: originalSymbol,
      date,
      close: parseFloat(close),
      open: parseFloat(open),
      high: parseFloat(high),
      low: parseFloat(low),
      volume: parseInt(volume, 10),
    };
  }).filter(point => !isNaN(point.close));
}

function calculateChange(current: number, previous: number): {
  change: number;
  changePercent: number;
} {
  const change = current - previous;
  const changePercent = (change / previous) * 100;
  return { change, changePercent };
}

export async function fetchLiveSymbolData(symbol: string): Promise<MarketSymbol> {
  try {
    const data = await fetchStooqData(symbol);
    
    if (data.length === 0) {
      throw new Error('No data received from Stooq');
    }

    data.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

    const current = data[0];
    const yesterday = data[1] || current;
    const weekAgo = data[5] || current;
    const monthAgo = data[21] || current;

    const daily = calculateChange(current.close, yesterday.close);
    const weekly = calculateChange(current.close, weekAgo.close);
    const monthly = calculateChange(current.close, monthAgo.close);

    return {
      symbol,
      name: getSymbolName(symbol),
      value: current.close,
      dailyChange: daily.change,
      dailyChangePercent: daily.changePercent,
      weeklyChange: weekly.change,
      weeklyChangePercent: weekly.changePercent,
      monthlyChange: monthly.change,
      monthlyChangePercent: monthly.changePercent,
    };
  } catch (error) {
    console.error(`Error fetching live data for ${symbol}:`, error);
    throw error;
  }
}

function getSymbolName(symbol: string): string {
  const names: Record<string, string> = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq-100 ETF',
    'IWM': 'Russell 2000 ETF',
    'TLT': '20+ Yr Treasury ETF',
    'GLD': 'Gold ETF',
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'NVDA': 'NVIDIA Corp.',
    '^VIX': 'VIX Volatility Index',
    '^TNX': '10-Year Treasury Yield',
    'BTC': 'Bitcoin',
  };
  return names[symbol] || symbol;
}

export async function fetchMultipleSymbols(symbols: string[]): Promise<{
  symbols: MarketSymbol[];
  dataState: DataState;
  errors: string[];
}> {
  const results = await Promise.allSettled(
    symbols.map(symbol => fetchLiveSymbolData(symbol))
  );

  const successfulSymbols: MarketSymbol[] = [];
  const errors: string[] = [];

  results.forEach((result, index) => {
    if (result.status === 'fulfilled') {
      successfulSymbols.push(result.value);
    } else {
      errors.push(`${symbols[index]}: ${result.reason.message}`);
    }
  });

  let dataState: DataState = 'LIVE';
  const successRate = successfulSymbols.length / symbols.length;
  
  if (successRate === 0) {
    dataState = 'FALLBACK';
  } else if (successRate < 0.8) {
    dataState = 'SNAPSHOT';
  }

  return {
    symbols: successfulSymbols,
    dataState,
    errors,
  };
}

export async function fetchBitcoinPrice(): Promise<MarketSymbol> {
  try {
    const url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24h_change=true&include_7d_change=true&include_30d_change=true';
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`CoinGecko API error: ${response.status}`);
    }

    const data = await response.json();
    const btcData = data.bitcoin;

    if (!btcData) {
      throw new Error('No Bitcoin data received');
    }

    const price = btcData.usd;
    const dailyChangePercent = btcData.usd_24h_change || 0;
    const weeklyChangePercent = btcData.usd_7d_change || 0;
    const monthlyChangePercent = btcData.usd_30d_change || 0;

    return {
      symbol: 'BTC',
      name: 'Bitcoin',
      value: price,
      dailyChange: (price * dailyChangePercent) / 100,
      dailyChangePercent,
      weeklyChange: (price * weeklyChangePercent) / 100,
      weeklyChangePercent,
      monthlyChange: (price * monthlyChangePercent) / 100,
      monthlyChangePercent,
    };
  } catch (error) {
    console.error('Error fetching Bitcoin price:', error);
    throw error;
  }
}
