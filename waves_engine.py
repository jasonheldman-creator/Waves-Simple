# waves_engine.py — WAVES Intelligence™ Vector Engine (Dynamic Strategy Stack)
#
# Mobile-friendly:
#   • No terminal, no CLI, no CSV requirement for core behavior.
#   • Internal Wave & benchmark definitions.
#
# Core behavior:
#   • Builds synthetic NAV & daily returns for each Wave vs composite benchmark.
#   • Wave returns use a dynamic multi-sleeve strategy:
#       - Momentum tilts (winners overweighted, laggards trimmed)
#       - Volatility targeting (keep Wave vol near a risk budget)
#       - Regime / SmartSafe gating (risk-on vs risk-off based on SPY trend)
#       - Mode-specific risk appetites (Standard, Alpha-Minus-Beta, Private Logic)
#       - Private Logic™ mean-reversion overlay on shocks
#   • Benchmarks stay passive composites (static weights).
#
# Public API used by app.py:
#   • USE_FULL_WAVE_HISTORY
#   • get_all_waves()
#   • get_modes()
#   • compute_history_nav(wave_name, mode, days)
#   • get_benchmark_mix_table()
#   • get_wave_holdings(wave_name)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------

USE_FULL_WAVE_HISTORY: bool = False  # kept for compatibility (logs mode, if added later)

TRADING_DAYS_PER_YEAR = 252

# Mode risk appetites & exposure caps
MODE_BASE_EXPOSURE: Dict[str, float] = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.85,
    "Private Logic": 1.10,
}

MODE_EXPOSURE_CAPS: Dict[str, tuple[float, float]] = {
    "Standard": (0.70, 1.30),
    "Alpha-Minus-Beta": (0.50, 1.00),
    "Private Logic": (0.80, 1.50),
}

# Regime → additional exposure tilt
REGIME_EXPOSURE: Dict[str, float] = {
    "panic": 0.80,
    "downtrend": 0.90,
    "neutral": 1.00,
    "uptrend": 1.10,
}

# Regime & mode → SmartSafe gating fraction (portion in safe asset)
REGIME_GATING: Dict[str, Dict[str, float]] = {
    "Standard": {
        "panic": 0.50,
        "downtrend": 0.30,
        "neutral": 0.10,
        "uptrend": 0.00,
    },
    "Alpha-Minus-Beta": {
        "panic": 0.75,
        "downtrend": 0.50,
        "neutral": 0.25,
        "uptrend": 0.05,
    },
    "Private Logic": {
        "panic": 0.40,
        "downtrend": 0.25,
        "neutral": 0.05,
        "uptrend": 0.00,
    },
}

# Volatility targeting
PORTFOLIO_VOL_TARGET = 0.20  # ~20% annualized


# ------------------------------------------------------------
# Wave & Benchmark definitions
# ------------------------------------------------------------

@dataclass
class Holding:
    ticker: str
    weight: float
    name: str | None = None


# Internal Wave holdings (example config; extend as needed)
WAVE_WEIGHTS: Dict[str, List[Holding]] = {
    "S&P 500 Wave": [
        Holding("AAPL", 0.07, "Apple Inc."),
        Holding("MSFT", 0.07, "Microsoft Corp."),
        Holding("AMZN", 0.05, "Amazon.com Inc."),
        Holding("GOOGL", 0.04, "Alphabet Inc. (Class A)"),
        Holding("META", 0.03, "Meta Platforms Inc."),
        Holding("NVDA", 0.06, "NVIDIA Corp."),
        Holding("BRK-B", 0.04, "Berkshire Hathaway Inc. (B)"),
        Holding("UNH", 0.03, "UnitedHealth Group Inc."),
        Holding("JPM", 0.03, "JPMorgan Chase & Co."),
        Holding("XOM", 0.03, "Exxon Mobil Corp."),
    ],
    "AI Wave": [
        Holding("NVDA", 0.12, "NVIDIA Corp."),
        Holding("MSFT", 0.10, "Microsoft Corp."),
        Holding("GOOGL", 0.08, "Alphabet Inc. (Class A)"),
        Holding("META", 0.07, "Meta Platforms Inc."),
        Holding("AVGO", 0.08, "Broadcom Inc."),
        Holding("ADBE", 0.07, "Adobe Inc."),
        Holding("AMD", 0.08, "Advanced Micro Devices Inc."),
        Holding("CRM", 0.06, "Salesforce Inc."),
        Holding("ORCL", 0.06, "Oracle Corp."),
        Holding("INTC", 0.06, "Intel Corp."),
    ],
    "Quantum Computing Wave": [
        Holding("IBM", 0.10, "International Business Machines Corp."),
        Holding("MSFT", 0.08, "Microsoft Corp."),
        Holding("GOOGL", 0.08, "Alphabet Inc. (Class A)"),
        Holding("NVDA", 0.10, "NVIDIA Corp."),
        Holding("AMZN", 0.08, "Amazon.com Inc."),
        Holding("QCOM", 0.08, "Qualcomm Inc."),
        Holding("INTC", 0.08, "Intel Corp."),
        Holding("TSM", 0.10, "Taiwan Semiconductor Manufacturing"),
        Holding("ADBE", 0.07, "Adobe Inc."),
        Holding("SNOW", 0.07, "Snowflake Inc."),
    ],
    "Future Power & Energy Wave": [
        Holding("XLE", 0.12, "Energy Select Sector SPDR"),
        Holding("ICLN", 0.10, "iShares Global Clean Energy"),
        Holding("ENPH", 0.08, "Enphase Energy Inc."),
        Holding("NEE", 0.10, "NextEra Energy Inc."),
        Holding("FSLR", 0.08, "First Solar Inc."),
        Holding("TSLA", 0.10, "Tesla Inc."),
        Holding("RUN", 0.07, "Sunrun Inc."),
        Holding("BP", 0.08, "BP plc"),
        Holding("CVX", 0.10, "Chevron Corp."),
        Holding("PLUG", 0.07, "Plug Power Inc."),
    ],
    "Clean Transit-Infrastructure Wave": [
        Holding("TSLA", 0.12, "Tesla Inc."),
        Holding("NIO", 0.08, "NIO Inc."),
        Holding("GM", 0.08, "General Motors"),
        Holding("F", 0.08, "Ford Motor Co."),
        Holding("CAT", 0.08, "Caterpillar Inc."),
        Holding("UNP", 0.08, "Union Pacific Corp."),
        Holding("VMC", 0.08, "Vulcan Materials"),
        Holding("MLM", 0.08, "Martin Marietta Materials"),
        Holding("XLI", 0.16, "Industrial Select Sector SPDR"),
        Holding("PAVE", 0.16, "Global X U.S. Infrastructure Development"),
    ],
    "Small Cap Growth Wave": [
        Holding("IWO", 0.30, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.30, "Vanguard Small-Cap Growth ETF"),
        Holding("ARKK", 0.10, "ARK Innovation ETF"),
        Holding("ZS", 0.10, "Zscaler Inc."),
        Holding("DDOG", 0.10, "Datadog Inc."),
        Holding("NET", 0.10, "Cloudflare Inc."),
    ],
    "Small to Mid Cap Growth Wave": [
        Holding("IWP", 0.30, "iShares Russell Mid-Cap Growth ETF"),
        Holding("MDY", 0.30, "SPDR S&P MidCap 400 ETF"),
        Holding("IWO", 0.20, "iShares Russell 2000 Growth ETF"),
        Holding("SMH", 0.20, "VanEck Semiconductor ETF"),
    ],
    "Crypto Income Wave": [
        Holding("BTC-USD", 0.40, "Bitcoin (USD)"),
        Holding("ETH-USD", 0.30, "Ethereum (USD)"),
        Holding("MSTR", 0.10, "MicroStrategy Inc."),
        Holding("COIN", 0.10, "Coinbase Global Inc."),
        Holding("BITO", 0.10, "ProShares Bitcoin Strategy ETF"),
    ],
    "SmartSafe Money Market Wave": [
        Holding("BIL", 0.50, "SPDR Bloomberg 1-3 Month T-Bill ETF"),
        Holding("SGOV", 0.50, "iShares 0-3 Month Treasury Bond ETF"),
    ],
}

# Composite ETF benchmarks (static)
BENCHMARK_WEIGHTS: Dict[str, List[Holding]] = {
    "S&P 500 Wave": [Holding("SPY", 1.0, "SPDR S&P 500 ETF")],
    "AI Wave": [
        Holding("QQQ", 0.60, "Invesco QQQ Trust"),
        Holding("SMH", 0.40, "VanEck Semiconductor ETF"),
    ],
    "Quantum Computing Wave": [
        Holding("QQQ", 0.50, "Invesco QQQ Trust"),
        Holding("SMH", 0.25, "VanEck Semiconductor ETF"),
        Holding("IBM", 0.25, "International Business Machines Corp."),
    ],
    "Future Power & Energy Wave": [
        Holding("XLE", 0.50, "Energy Select Sector SPDR"),
        Holding("ICLN", 0.50, "iShares Global Clean Energy"),
    ],
    "Clean Transit-Infrastructure Wave": [
        Holding("PAVE", 0.60, "Global X U.S. Infrastructure Development"),
        Holding("XLI", 0.40, "Industrial Select Sector SPDR"),
    ],
    "Small Cap Growth Wave": [
        Holding("IWO", 0.50, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.50, "Vanguard Small-Cap Growth ETF"),
    ],
    "Small to Mid Cap Growth Wave": [
        Holding("IWP", 0.50, "iShares Russell Mid-Cap Growth ETF"),
        Holding("IWO", 0.50, "iShares Russell 2000 Growth ETF"),
    ],
    "Crypto Income Wave": [
        Holding("BTC-USD", 0.50, "Bitcoin (USD)"),
        Holding("ETH-USD", 0.50, "Ethereum (USD)"),
    ],
    "SmartSafe Money Market Wave": [
        Holding("BIL", 0.50, "SPDR Bloomberg 1-3 Month T-Bill"),
        Holding("SGOV", 0.50, "iShares 0-3 Month Treasury Bond ETF"),
    ],
}


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_all_waves() -> list[str]:
    return sorted(WAVE_WEIGHTS.keys())


def get_modes() -> list[str]:
    return list(MODE_BASE_EXPOSURE.keys())


def _normalize_weights(holdings: List[Holding]) -> pd.Series:
    """Return normalized weight Series indexed by ticker."""
    if not holdings:
        return pd.Series(dtype=float)

    df = pd.DataFrame(
        [{"ticker": h.ticker, "weight": h.weight} for h in holdings]
    )
    df = df.groupby("ticker", as_index=False)["weight"].sum()
    total = df["weight"].sum()
    if total <= 0:
        return pd.Series(dtype=float)
    df["weight"] = df["weight"] / total
    return df.set_index("ticker")["weight"]


def _download_history(tickers: list[str], days: int) -> pd.DataFrame:
    """
    Download daily adjusted close prices for given tickers.
    """
    if yf is None:
        raise RuntimeError(
            "yfinance is not available in this environment. "
            "Please ensure yfinance is installed."
        )

    # Add generous buffer for rolling windows (momentum / vol)
    lookback_days = days + 260
    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days)

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # Handle multi-index columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            top = data.columns.levels[0][0]
            data = data[top]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index()
    data = data.ffill().bfill()
    return data


def _regime_from_return(ret_60d: float) -> str:
    """Map 60D index return into a simple market regime."""
    if np.isnan(ret_60d):
        return "neutral"
    if ret_60d <= -0.12:
        return "panic"
    if ret_60d <= -0.04:
        return "downtrend"
    if ret_60d < 0.06:
        return "neutral"
    return "uptrend"


def compute_history_nav(
    wave_name: str,
    mode: str = "Standard",
    days: int = 365,
) -> pd.DataFrame:
    """
    Compute Wave & Benchmark NAV and daily returns over a given window.

    Returns DataFrame indexed by Date with columns:
        ['wave_nav', 'bm_nav', 'wave_ret', 'bm_ret']

    Behavior:
        • Benchmark: passive, static weights.
        • Wave: dynamic strategy —
            - Momentum tilts
            - Volatility targeting
            - Regime / SmartSafe gating
            - Mode-specific exposure and caps
            - Private Logic mean-reversion overlay
    """
    if wave_name not in WAVE_WEIGHTS:
        raise ValueError(f"Unknown Wave: {wave_name}")
    if mode not in MODE_BASE_EXPOSURE:
        raise ValueError(f"Unknown mode: {mode}")

    wave_holdings = WAVE_WEIGHTS[wave_name]
    bm_holdings = BENCHMARK_WEIGHTS.get(wave_name, [])

    wave_weights = _normalize_weights(wave_holdings)
    bm_weights = _normalize_weights(bm_holdings)

    tickers_wave = list(wave_weights.index)
    tickers_bm = list(bm_weights.index)

    # Always include SPY (regime index) and SGOV (SmartSafe proxy) for strategy logic.
    base_index_ticker = "SPY"
    safe_candidates = ["SGOV", "BIL", "SHY"]

    all_tickers = set(tickers_wave + tickers_bm)
    all_tickers.add(base_index_ticker)
    all_tickers.update(safe_candidates)

    all_tickers = sorted(all_tickers)

    if not all_tickers:
        return pd.DataFrame(
            columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float
        )

    price_df = _download_history(all_tickers, days=days)
    if price_df.empty:
        return pd.DataFrame(
            columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float
        )

    # Restrict to requested window (last N days) for outputs
    if len(price_df) > days:
        price_df = price_df.iloc[-days:]

    # Daily returns
    ret_df = price_df.pct_change().fillna(0.0)

    # Align weights to price columns
    wave_weights_aligned = wave_weights.reindex(price_df.columns).fillna(0.0)
    bm_weights_aligned = bm_weights.reindex(price_df.columns).fillna(0.0)

    # Benchmark: static, passive
    bm_ret_series = (ret_df * bm_weights_aligned).sum(axis=1)

    # Regime signal: 60D return of base index (SPY or fallback)
    if base_index_ticker in price_df.columns:
        idx_price = price_df[base_index_ticker]
    else:
        # Fallback: use first benchmark ticker, then first Wave ticker
        fallback_ticker = (
            tickers_bm[0]
            if tickers_bm
            else (tickers_wave[0] if tickers_wave else price_df.columns[0])
        )
        idx_price = price_df[fallback_ticker]

    idx_ret_60d = idx_price / idx_price.shift(60) - 1.0

    # Momentum signal: 60D return per asset
    mom_60 = price_df / price_df.shift(60) - 1.0

    # Choose safe asset ticker
    safe_ticker = None
    for t in safe_candidates:
        if t in price_df.columns:
            safe_ticker = t
            break
    if safe_ticker is None:
        # Fallback to SGOV-like behavior via base index (not ideal but safe)
        safe_ticker = base_index_ticker

    safe_ret_series = ret_df[safe_ticker]

    # Dynamic Wave strategy
    mode_base_exposure = MODE_BASE_EXPOSURE[mode]
    exp_min, exp_max = MODE_EXPOSURE_CAPS[mode]

    wave_ret_list: List[float] = []
    dates: List[pd.Timestamp] = []

    for i, dt in enumerate(ret_df.index):
        # Current returns row
        rets = ret_df.loc[dt]

        # Regime now
        regime = _regime_from_return(idx_ret_60d.get(dt, np.nan))
        regime_exposure = REGIME_EXPOSURE[regime]
        gating_fraction = REGIME_GATING[mode][regime]

        # Momentum tilt for this date
        mom_row = mom_60.loc[dt] if dt in mom_60.index else None
        if mom_row is not None:
            mom_series = mom_row.reindex(price_df.columns).fillna(0.0)
            # Clip momentum scores and convert to tilt factors
            mom_clipped = mom_series.clip(lower=-0.30, upper=0.30)
            # Strong winners get up to ~+24% tilt, laggards up to ~-24%
            tilt_factor = 1.0 + 0.8 * mom_clipped
            # Apply tilt only to Wave components
            effective_weights = wave_weights_aligned * tilt_factor
        else:
            effective_weights = wave_weights_aligned.copy()

        # Normalize risk weights (only among risk assets, safe asset handled separately)
        # Ensure no negative weights
        effective_weights = effective_weights.clip(lower=0.0)
        risk_weight_total = effective_weights.sum()
        if risk_weight_total > 0:
            risk_weights = effective_weights / risk_weight_total
        else:
            # Degenerate: fallback to original weights
            risk_weights = wave_weights_aligned.copy()

        # Base risk portfolio return (without SmartSafe / exposure scaling)
        portfolio_risk_ret = float((rets * risk_weights).sum())
        safe_ret = float(safe_ret_series.loc[dt])

        # Rolling vol for volatility targeting
        if len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:])
            recent_vol = recent.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            recent_vol = PORTFOLIO_VOL_TARGET  # neutral early-on

        vol_adjust = 1.0
        if recent_vol > 0:
            vol_adjust = PORTFOLIO_VOL_TARGET / recent_vol
            vol_adjust = float(np.clip(vol_adjust, 0.7, 1.3))

        # Combined exposure factor
        raw_exposure = mode_base_exposure * regime_exposure * vol_adjust
        exposure = float(np.clip(raw_exposure, exp_min, exp_max))

        # Split between safe and risk sleeves
        safe_fraction = gating_fraction
        risk_fraction = 1.0 - safe_fraction

        base_total_ret = safe_fraction * safe_ret + risk_fraction * exposure * portfolio_risk_ret

        # Private Logic™ mean-reversion overlay:
        #   - If a daily move is a big negative shock vs recent vol → lean in slightly.
        #   - If a daily move is a big positive spike → take some profits.
        total_ret = base_total_ret
        if mode == "Private Logic" and len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:])
            daily_vol = recent.std()
            if daily_vol > 0:
                shock_threshold = 2.0 * daily_vol
                if base_total_ret <= -shock_threshold:
                    # Big selloff → lean in
                    total_ret = base_total_ret * 1.30
                elif base_total_ret >= shock_threshold:
                    # Big spike → dampen
                    total_ret = base_total_ret * 0.70

        wave_ret_list.append(total_ret)
        dates.append(dt)

    wave_ret_series = pd.Series(wave_ret_list, index=pd.Index(dates, name="Date"))

    # Align benchmark series to same index
    bm_ret_series = bm_ret_series.reindex(wave_ret_series.index).fillna(0.0)

    # Compute NAV (start at 1.0)
    wave_nav = (1.0 + wave_ret_series).cumprod()
    bm_nav = (1.0 + bm_ret_series).cumprod()

    out = pd.DataFrame(
        {
            "wave_nav": wave_nav,
            "bm_nav": bm_nav,
            "wave_ret": wave_ret_series,
            "bm_ret": bm_ret_series,
        }
    )
    out.index.name = "Date"
    return out


def get_benchmark_mix_table() -> pd.DataFrame:
    """
    Return ETF mix used for each Wave's benchmark:
        ['Wave', 'Ticker', 'Name', 'Weight']
    """
    rows = []
    for wave, holdings in BENCHMARK_WEIGHTS.items():
        if not holdings:
            continue
        weights = _normalize_weights(holdings)
        for h in holdings:
            if h.ticker not in weights.index:
                continue
            rows.append(
                {
                    "Wave": wave,
                    "Ticker": h.ticker,
                    "Name": h.name or "",
                    "Weight": float(weights[h.ticker]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["Wave", "Weight"], ascending=[True, False])
    return df


def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    """
    Return Wave holdings as DataFrame:
        ['Ticker', 'Name', 'Weight']
    """
    holdings = WAVE_WEIGHTS.get(wave_name, [])
    if not holdings:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])

    weights = _normalize_weights(holdings)

    rows = []
    for h in holdings:
        if h.ticker not in weights.index:
            continue
        rows.append(
            {
                "Ticker": h.ticker,
                "Name": h.name or "",
                "Weight": float(weights[h.ticker]),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"])
    df = df.sort_values("Weight", ascending=False)
    return df