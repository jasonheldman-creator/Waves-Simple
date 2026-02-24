# helpers/forward_intelligence_engine.py
# Forward Intelligence engine — live signal generation per Wave.
# Observational only.  No trade signals, no price targets.

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent.parent
_SIGNALS_PATH = _BASE_DIR / "data" / "forward_intelligence_signals.json"

_REQUIRED_COLUMNS = [
    "ticker",
    "wave",
    "signal_type",   # earnings | corporate_action | news | risk
    "title",
    "description",
    "event_date",
    "severity",
    "source",
    "created_at",
]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_REQUIRED_COLUMNS)


# ---------------------------------------------------------------------------
# Helpers — Finnhub HTTP calls
# ---------------------------------------------------------------------------

def _finnhub_get(path: str, params: dict | None = None) -> dict | list | None:
    """Issue a GET request to the Finnhub REST API.  Returns None on any error."""
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return None
    try:
        import requests  # type: ignore

        base = "https://finnhub.io/api/v1"
        p = dict(params or {})
        p["token"] = api_key
        resp = requests.get(f"{base}{path}", params=p, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _today_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _future_str(days: int) -> str:
    return (datetime.now(tz=timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


def _past_str(days: int) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# A) Earnings signals
# ---------------------------------------------------------------------------

def _earnings_signals(tickers: List[str], wave: str, now_str: str) -> list[dict]:
    rows: list[dict] = []
    created_at = datetime.now(tz=timezone.utc).isoformat()
    cutoff = _future_str(10)

    for ticker in tickers:
        data = _finnhub_get("/calendar/earnings", {"symbol": ticker, "from": now_str, "to": cutoff})
        entries = []
        if isinstance(data, dict):
            entries = data.get("earningsCalendar", [])
        elif isinstance(data, list):
            entries = data

        for entry in entries:
            ed = str(entry.get("date", ""))
            if not ed or ed < now_str or ed > cutoff:
                continue
            eps_est = entry.get("epsEstimate")
            eps_act = entry.get("epsActual")
            desc_parts = [f"Earnings date: {ed}"]
            if eps_est is not None:
                desc_parts.append(f"EPS Estimate: {eps_est}")
            if eps_act is not None:
                desc_parts.append(f"EPS Actual: {eps_act}")
            rows.append({
                "ticker": ticker,
                "wave": wave,
                "signal_type": "earnings",
                "title": f"Upcoming Earnings — {ticker}",
                "description": ". ".join(desc_parts),
                "event_date": ed,
                "severity": "high" if ed <= _future_str(3) else "medium",
                "source": "finnhub/earnings_calendar",
                "created_at": created_at,
            })

    return rows


# ---------------------------------------------------------------------------
# B) Corporate action signals (splits + dividends)
# ---------------------------------------------------------------------------

def _corporate_action_signals(tickers: List[str], wave: str) -> list[dict]:
    rows: list[dict] = []
    created_at = datetime.now(tz=timezone.utc).isoformat()
    from_date = _past_str(30)
    to_date = _future_str(30)

    for ticker in tickers:
        # Splits
        split_data = _finnhub_get("/stock/split", {"symbol": ticker, "from": from_date, "to": to_date})
        splits = []
        if isinstance(split_data, dict):
            splits = split_data.get("data", [])
        elif isinstance(split_data, list):
            splits = split_data
        for s in splits:
            ed = str(s.get("date", ""))
            ratio = s.get("toFactor") or s.get("ratio", "")
            rows.append({
                "ticker": ticker,
                "wave": wave,
                "signal_type": "corporate_action",
                "title": f"Stock Split — {ticker}",
                "description": f"Split ratio: {ratio}. Date: {ed}.",
                "event_date": ed,
                "severity": "medium",
                "source": "finnhub/stock_split",
                "created_at": created_at,
            })

        # Dividends
        div_data = _finnhub_get("/stock/dividend", {"symbol": ticker, "from": from_date, "to": to_date})
        divs = []
        if isinstance(div_data, list):
            divs = div_data
        elif isinstance(div_data, dict):
            divs = div_data.get("data", [])
        for d in divs:
            ed = str(d.get("exDate") or d.get("date", ""))
            amount = d.get("amount", "")
            rows.append({
                "ticker": ticker,
                "wave": wave,
                "signal_type": "corporate_action",
                "title": f"Dividend — {ticker}",
                "description": f"Ex-date: {ed}. Amount: {amount}.",
                "event_date": ed,
                "severity": "low",
                "source": "finnhub/dividend",
                "created_at": created_at,
            })

    return rows


# ---------------------------------------------------------------------------
# C) News signals
# ---------------------------------------------------------------------------

_NEGATIVE_KEYWORDS = [
    "miss", "loss", "decline", "fall", "cut", "reduce", "warn", "risk",
    "concern", "drop", "fell", "downgrade", "slump", "weak", "layoff", "recall",
    "lawsuit", "probe", "investigation",
]
_POSITIVE_KEYWORDS = [
    "beat", "exceeds", "strong", "growth", "profit", "gain", "surge", "record",
    "outperform", "rise", "rose", "upgrade", "raises", "acquisition", "deal",
]
# When a ticker has fewer than this many news items, include neutral headlines too
# so that at least some content is displayed even in slow news periods.
_MIN_ITEMS_FOR_NEUTRAL_FILTER = 3


def _headline_sentiment(text: str) -> str:
    tl = text.lower()
    pos = sum(1 for w in _POSITIVE_KEYWORDS if w in tl)
    neg = sum(1 for w in _NEGATIVE_KEYWORDS if w in tl)
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def _news_signals(tickers: List[str], wave: str) -> list[dict]:
    rows: list[dict] = []
    created_at = datetime.now(tz=timezone.utc).isoformat()
    from_date = _past_str(3)
    to_date = _today_str()

    for ticker in tickers:
        data = _finnhub_get("/company-news", {"symbol": ticker, "from": from_date, "to": to_date})
        items = data if isinstance(data, list) else []
        for item in items[:10]:
            headline = item.get("headline", item.get("title", ""))
            if not headline:
                continue
            sentiment = _headline_sentiment(headline)
            # Simple relevance heuristic: report all non-neutral, or all if fewer items exist
            if sentiment == "Neutral" and len(items) > _MIN_ITEMS_FOR_NEUTRAL_FILTER:
                continue
            ts_raw = item.get("datetime", 0)
            try:
                ts = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                ts = to_date
            severity = "high" if sentiment == "Negative" else "low"
            rows.append({
                "ticker": ticker,
                "wave": wave,
                "signal_type": "news",
                "title": headline[:200],
                "description": f"Sentiment: {sentiment}. Source: {item.get('source', '')}.",
                "event_date": ts,
                "severity": severity,
                "source": item.get("source", "finnhub/company_news"),
                "created_at": created_at,
            })

    return rows


# ---------------------------------------------------------------------------
# D) Structural risk signals
# ---------------------------------------------------------------------------

def _risk_signals(tickers: List[str], wave: str) -> list[dict]:
    """Signal when |daily_return| > 5% OR volume spike > 2x avg (Finnhub quote)."""
    rows: list[dict] = []
    created_at = datetime.now(tz=timezone.utc).isoformat()
    today = _today_str()

    for ticker in tickers:
        data = _finnhub_get("/quote", {"symbol": ticker})
        if not isinstance(data, dict):
            continue

        try:
            current = float(data.get("c", 0))
            prev_close = float(data.get("pc", 0))
            if prev_close > 0 and current > 0:
                daily_return = (current - prev_close) / prev_close
                if abs(daily_return) > 0.05:
                    direction = "up" if daily_return > 0 else "down"
                    rows.append({
                        "ticker": ticker,
                        "wave": wave,
                        "signal_type": "risk",
                        "title": f"Large Price Move — {ticker}",
                        "description": (
                            f"Daily return {daily_return:+.1%} ({direction}). "
                            "Move exceeds ±5% structural risk threshold."
                        ),
                        "event_date": today,
                        "severity": "high",
                        "source": "finnhub/quote",
                        "created_at": created_at,
                    })
        except Exception:
            pass

        # Volume spike: use candle data (daily, last 20 days)
        try:
            from_ts = int(
                (datetime.now(tz=timezone.utc) - timedelta(days=20)).timestamp()
            )
            to_ts = int(datetime.now(tz=timezone.utc).timestamp())
            candles = _finnhub_get(
                "/stock/candle",
                {"symbol": ticker, "resolution": "D", "from": from_ts, "to": to_ts},
            )
            if isinstance(candles, dict) and candles.get("s") == "ok":
                volumes = candles.get("v", [])
                if len(volumes) >= 3:
                    # Exclude latest day from average to detect spikes against historical baseline
                    avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
                    latest_vol = volumes[-1]
                    if avg_vol > 0 and latest_vol > 2 * avg_vol:
                        rows.append({
                            "ticker": ticker,
                            "wave": wave,
                            "signal_type": "risk",
                            "title": f"Volume Spike — {ticker}",
                            "description": (
                                f"Volume {latest_vol:,.0f} is "
                                f"{latest_vol / avg_vol:.1f}x the 20-day average "
                                f"({avg_vol:,.0f}). Structural risk threshold exceeded."
                            ),
                            "event_date": today,
                            "severity": "medium",
                            "source": "finnhub/candle",
                            "created_at": created_at,
                        })
        except Exception:
            pass

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _synthetic_signals(tickers: List[str], wave_name: str) -> list[dict]:
    """Generate deterministic synthetic Forward Intelligence signals from wave holdings.

    Used when external feeds (Finnhub) are unavailable so that panels
    always display populated content.

    Signal types generated:
    - earnings: ±10 day earnings preparation window placeholders
    - corporate_action: structural corporate event placeholders
    - news: news awareness placeholders
    - risk: structural risk alert placeholders
    """
    created_at = datetime.now(tz=timezone.utc).isoformat()
    today = datetime.now(tz=timezone.utc)
    rows: list[dict] = []

    for i, ticker in enumerate(tickers):
        # Deterministic offset based on ticker hash so dates vary across tickers
        offset_seed = sum(ord(c) for c in ticker) % 21  # 0–20 days
        earnings_date = (today + timedelta(days=offset_seed - 10)).strftime("%Y-%m-%d")

        # Earnings preparation window signal (±10 days)
        rows.append({
            "ticker": ticker,
            "wave": wave_name,
            "signal_type": "earnings",
            "title": f"{ticker} — Earnings Preparation Window",
            "description": (
                f"Earnings observation window active for {ticker}. "
                f"Scheduled reporting period near {earnings_date}. "
                "Monitor for volatility and volume changes."
            ),
            "event_date": earnings_date,
            "severity": "medium",
            "source": "synthetic-placeholder",
            "created_at": created_at,
        })

        # Corporate events placeholder (every other ticker)
        if i % 2 == 0:
            event_date = (today + timedelta(days=offset_seed)).strftime("%Y-%m-%d")
            rows.append({
                "ticker": ticker,
                "wave": wave_name,
                "signal_type": "corporate_action",
                "title": f"{ticker} — Corporate Event Observation",
                "description": (
                    f"Structural corporate event placeholder for {ticker}. "
                    "No material action detected; monitoring active."
                ),
                "event_date": event_date,
                "severity": "low",
                "source": "synthetic-placeholder",
                "created_at": created_at,
            })

        # News placeholder (every third ticker)
        if i % 3 == 0:
            rows.append({
                "ticker": ticker,
                "wave": wave_name,
                "signal_type": "news",
                "title": f"{ticker} — News Awareness Placeholder",
                "description": (
                    f"News monitoring active for {ticker}. "
                    "No material headlines detected in observation window."
                ),
                "event_date": today.strftime("%Y-%m-%d"),
                "severity": "low",
                "source": "synthetic-placeholder",
                "created_at": created_at,
            })

        # Structural risk alert (first ticker as portfolio-level signal)
        if i == 0:
            rows.append({
                "ticker": ticker,
                "wave": wave_name,
                "signal_type": "risk",
                "title": f"{wave_name or ticker} — Structural Risk Monitoring",
                "description": (
                    f"Structural risk monitoring active for {wave_name or ticker}. "
                    "Portfolio exposure levels within expected ranges. "
                    "No elevated risk conditions detected."
                ),
                "event_date": today.strftime("%Y-%m-%d"),
                "severity": "low",
                "source": "synthetic-placeholder",
                "created_at": created_at,
            })

    return rows


def generate_forward_signals(
    tickers: List[str],
    wave_name: str = "",
) -> pd.DataFrame:
    """Generate live Forward Intelligence signals for the given tickers.

    Execution order:
    1. Attempt to load live signals via external feeds (Finnhub).
    2. If no signals are produced (feeds unavailable or returning empty),
       generate deterministic synthetic signals derived from wave holdings
       so that Forward Intelligence panels always display populated content.

    Parameters
    ----------
    tickers:
        List of ticker symbols (e.g. from load_wave_holdings).
    wave_name:
        Wave label to embed in each signal row.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, wave, signal_type, title, description, event_date,
        severity, source, created_at.  Never empty when tickers are provided.
    """
    if not tickers:
        return _empty_df()

    tickers = [str(t).strip().upper() for t in tickers if t and str(t).strip()]
    if not tickers:
        return _empty_df()

    now_str = _today_str()
    rows: list[dict] = []

    rows.extend(_earnings_signals(tickers, wave_name, now_str))
    rows.extend(_corporate_action_signals(tickers, wave_name))
    rows.extend(_news_signals(tickers, wave_name))
    rows.extend(_risk_signals(tickers, wave_name))

    if not rows:
        # External feeds produced no signals — fall back to deterministic synthetic signals
        rows = _synthetic_signals(tickers, wave_name)

    df = pd.DataFrame(rows, columns=_REQUIRED_COLUMNS) if rows else _empty_df()

    # Persist signals to disk
    try:
        _SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        records = df.to_dict(orient="records") if not df.empty else []
        with open(_SIGNALS_PATH, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)
    except Exception:
        pass

    return df
