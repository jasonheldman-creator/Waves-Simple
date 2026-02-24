# NOTE 077 (Revised) — Forward Intelligence Signals
# Wave-scoped observational awareness layer.
# Non-executing. No governance authority. No trade signals or price targets.

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd


_REQUIRED_COLUMNS = [
    "signal_id",
    "wave",
    "signal_type",
    "signal_title",
    "observation",
    "confidence",
    "horizon",
    "created_at",
]

_EVENTS_COLUMNS = [
    "ticker",
    "company_name",
    "earnings_date",
    "earnings_status",
    "earnings_surprise",
    "split_flag",
    "dividend_flag",
    "buyback_flag",
    "symbol_change_flag",
    "ma_flag",
]

_NEWS_COLUMNS = [
    "ticker",
    "title",
    "source",
    "timestamp",
    "sentiment",
]

_SIGNALS_COLUMNS = [
    "ticker",
    "volatility_status",
    "volume_status",
    "gap_flag",
    "correlation_status",
    "overall_status",
]

_BASE_DIR = Path(__file__).resolve().parent.parent


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_REQUIRED_COLUMNS)


def load_wave_holdings(wave_name: str) -> pd.DataFrame:
    """Load holdings for a specific wave from wave_weights.csv.

    Returns a DataFrame with columns: ticker, weight.
    Returns an empty DataFrame if the wave is not found or data is unavailable.
    """
    weights_path = _BASE_DIR / "data" / "wave_weights.csv"
    fallback_path = _BASE_DIR / "wave_weights.csv"

    path = weights_path if weights_path.exists() else (fallback_path if fallback_path.exists() else None)
    if path is None:
        return pd.DataFrame(columns=["ticker", "weight"])

    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        wave_col = next(
            (c for c in ["wave_id", "wave_name", "wave"] if c in df.columns), None
        )
        if wave_col is None:
            return pd.DataFrame(columns=["ticker", "weight"])

        mask = df[wave_col].astype(str).str.strip() == str(wave_name).strip()
        wave_df = df[mask].copy()
        if wave_df.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        ticker_col = next(
            (c for c in ["ticker", "symbol"] if c in wave_df.columns), None
        )
        weight_col = "weight" if "weight" in wave_df.columns else None

        if ticker_col is None:
            return pd.DataFrame(columns=["ticker", "weight"])

        result = wave_df[[ticker_col]].rename(columns={ticker_col: "ticker"})
        if weight_col:
            result["weight"] = pd.to_numeric(wave_df[weight_col], errors="coerce").fillna(0.0)
        else:
            result["weight"] = 0.0

        result = result[result["ticker"].astype(str).str.strip() != ""]
        if weight_col:
            result = result[result["weight"] > 0]
        return result.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["ticker", "weight"])


def _trading_days_until(target_date: datetime, reference: datetime | None = None) -> int | None:
    """Estimate trading days between reference and target (Mon–Fri, no holidays)."""
    try:
        ref = reference or datetime.now(tz=timezone.utc).replace(tzinfo=None)
        if hasattr(target_date, "tzinfo") and target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)
        if hasattr(ref, "tzinfo") and ref.tzinfo is not None:
            ref = ref.replace(tzinfo=None)
        if target_date < ref:
            return None
        count = 0
        current = ref
        while current < target_date:
            current += timedelta(days=1)
            if current.weekday() < 5:
                count += 1
        return count
    except Exception:
        return None


def _try_finnhub_earnings(ticker: str) -> dict:
    """Attempt to fetch next earnings date from Finnhub API."""
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return {}
    try:
        import requests  # type: ignore
        url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&token={api_key}"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            earnings_list = data.get("earningsCalendar", [])
            now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            future = [e for e in earnings_list if e.get("date", "") >= now_str]
            if future:
                future.sort(key=lambda x: x["date"])
                entry = future[0]
                surprise = None
                hist = [e for e in earnings_list if e.get("date", "") < now_str]
                if hist:
                    hist.sort(key=lambda x: x["date"], reverse=True)
                    last = hist[0]
                    actual = last.get("epsActual")
                    estimate = last.get("epsEstimate")
                    if actual is not None and estimate is not None and estimate != 0:
                        try:
                            surprise = (float(actual) - float(estimate)) / abs(float(estimate))
                        except Exception:
                            pass
                return {
                    "earnings_date": entry["date"],
                    "surprise": surprise,
                    "company_name": entry.get("company", ticker),
                }
    except Exception:
        pass
    return {}


def _try_yfinance_earnings(ticker: str) -> dict:
    """Attempt to fetch next earnings date via yfinance."""
    try:
        import yfinance as yf  # type: ignore
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is not None and not (hasattr(cal, "empty") and cal.empty):
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date", [None])[0] if cal.get("Earnings Date") else None
            else:
                ed = None
                if hasattr(cal, "loc") and "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"].iloc[0] if hasattr(cal.loc["Earnings Date"], "iloc") else cal.loc["Earnings Date"]
            if ed is not None:
                return {"earnings_date": str(ed)[:10], "company_name": ticker}
    except Exception:
        pass
    return {}


def fetch_forward_events(tickers: List[str]) -> pd.DataFrame:
    """Fetch corporate events for a list of tickers.

    Tries Finnhub first (if FINNHUB_API_KEY env var set), falls back to yfinance.
    Returns a DataFrame with columns defined in _EVENTS_COLUMNS.
    Returns an empty DataFrame (with correct columns) if no data available.
    """
    rows = []
    now = datetime.now(tz=timezone.utc)

    for ticker in tickers:
        if not ticker or not str(ticker).strip():
            continue
        ticker = str(ticker).strip().upper()

        row: dict = {
            "ticker": ticker,
            "company_name": ticker,
            "earnings_date": None,
            "earnings_status": "Unknown",
            "earnings_surprise": None,
            "split_flag": False,
            "dividend_flag": False,
            "buyback_flag": False,
            "symbol_change_flag": False,
            "ma_flag": False,
        }

        data = _try_finnhub_earnings(ticker)
        if not data:
            data = _try_yfinance_earnings(ticker)

        if data:
            row["company_name"] = data.get("company_name", ticker)
            ed_str = data.get("earnings_date")
            if ed_str:
                row["earnings_date"] = ed_str
                try:
                    ed_dt = datetime.strptime(str(ed_str)[:10], "%Y-%m-%d")
                    td = _trading_days_until(ed_dt, now.replace(tzinfo=None))
                    if td is not None:
                        if td <= 10:
                            row["earnings_status"] = "Upcoming"
                        else:
                            row["earnings_status"] = "Scheduled"
                    else:
                        row["earnings_status"] = "Recently Reported"
                except Exception:
                    row["earnings_status"] = "Scheduled"
            surprise = data.get("surprise")
            if surprise is not None:
                row["earnings_surprise"] = round(float(surprise) * 100, 2)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_EVENTS_COLUMNS)

    df = pd.DataFrame(rows, columns=_EVENTS_COLUMNS)
    return df


def _try_finnhub_news(ticker: str) -> list:
    """Attempt to fetch recent news headlines from Finnhub."""
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return []
    try:
        import requests  # type: ignore
        from_date = (datetime.now(tz=timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
        to_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
        )
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            news = resp.json()
            if isinstance(news, list):
                return news[:10]
    except Exception:
        pass
    return []


def _try_yfinance_news(ticker: str) -> list:
    """Attempt to fetch recent news via yfinance."""
    try:
        import yfinance as yf  # type: ignore
        t = yf.Ticker(ticker)
        news = t.news
        if news:
            return news[:10]
    except Exception:
        pass
    return []


def _classify_sentiment(text: str) -> str:
    """Simple keyword-based sentiment classification. Observational only."""
    if not text:
        return "Neutral"
    text_lower = text.lower()
    positive_words = [
        "beat", "exceeds", "strong", "growth", "profit", "gain", "surge",
        "record", "outperform", "rise", "rises", "rose", "upgrade", "raises",
    ]
    negative_words = [
        "miss", "loss", "decline", "fall", "cut", "reduce", "warn", "risk",
        "concern", "drop", "drops", "fell", "downgrade", "slump", "weak",
    ]
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    if pos_count > neg_count:
        return "Positive"
    if neg_count > pos_count:
        return "Negative"
    return "Neutral"


def fetch_news_context(tickers: List[str]) -> pd.DataFrame:
    """Fetch recent material headlines for a list of tickers.

    Tries Finnhub first (if FINNHUB_API_KEY env var set), falls back to yfinance.
    Returns a DataFrame with columns defined in _NEWS_COLUMNS.
    Returns an empty DataFrame (with correct columns) if no data available.
    No price predictions or recommendations are generated.
    """
    rows = []

    for ticker in tickers:
        if not ticker or not str(ticker).strip():
            continue
        ticker = str(ticker).strip().upper()

        news_items = _try_finnhub_news(ticker)
        if not news_items:
            news_items = _try_yfinance_news(ticker)

        if not news_items:
            continue

        for item in news_items[:10]:
            if isinstance(item, dict):
                title = item.get("headline", item.get("title", ""))
                source = item.get("source", "")
                ts_raw = item.get("datetime", item.get("providerPublishTime", 0))
                try:
                    if isinstance(ts_raw, (int, float)) and ts_raw > 0:
                        ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    elif isinstance(ts_raw, str) and ts_raw:
                        ts = str(ts_raw)[:16]
                    else:
                        ts = ""
                except Exception:
                    ts = ""
                sentiment = _classify_sentiment(title)
                rows.append({
                    "ticker": ticker,
                    "title": title,
                    "source": source,
                    "timestamp": ts,
                    "sentiment": sentiment,
                })

    if not rows:
        return pd.DataFrame(columns=_NEWS_COLUMNS)

    return pd.DataFrame(rows, columns=_NEWS_COLUMNS)


def compute_forward_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute structural risk signals for holdings.

    Accepts a DataFrame with at minimum a 'ticker' column and optional
    price/volume data. Returns a DataFrame with columns defined in _SIGNALS_COLUMNS.
    Observational only — no trade signals, price targets, or recommendations.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_SIGNALS_COLUMNS)

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    rows = []
    tickers = df["ticker"].dropna().unique().tolist() if "ticker" in df.columns else []

    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker]
        row = {
            "ticker": str(ticker).upper(),
            "volatility_status": "Stable",
            "volume_status": "Stable",
            "gap_flag": False,
            "correlation_status": "Stable",
            "overall_status": "Stable",
        }

        # Volatility check: if return_std column available
        if "return_std" in ticker_df.columns:
            try:
                std_val = float(ticker_df["return_std"].iloc[0])
                if std_val > 0.035:
                    row["volatility_status"] = "Review Awareness"
                elif std_val > 0.020:
                    row["volatility_status"] = "Monitoring"
            except Exception:
                pass

        # Volume check: if volume_ratio column available
        if "volume_ratio" in ticker_df.columns:
            try:
                vol_ratio = float(ticker_df["volume_ratio"].iloc[0])
                if vol_ratio > 2.5:
                    row["volume_status"] = "Review Awareness"
                elif vol_ratio > 1.5:
                    row["volume_status"] = "Monitoring"
            except Exception:
                pass

        # Gap check: if gap_pct column available
        if "gap_pct" in ticker_df.columns:
            try:
                gap = abs(float(ticker_df["gap_pct"].iloc[0]))
                row["gap_flag"] = gap > 0.03
            except Exception:
                pass

        # Correlation check
        if "correlation_vs_benchmark" in ticker_df.columns:
            try:
                corr = float(ticker_df["correlation_vs_benchmark"].iloc[0])
                if corr < 0.2 or corr > 0.98:
                    row["correlation_status"] = "Monitoring"
            except Exception:
                pass

        # Overall status: worst of individual statuses
        statuses = [row["volatility_status"], row["volume_status"], row["correlation_status"]]
        if "Review Awareness" in statuses or row["gap_flag"]:
            row["overall_status"] = "Review Awareness"
        elif "Monitoring" in statuses:
            row["overall_status"] = "Monitoring"

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_SIGNALS_COLUMNS)

    return pd.DataFrame(rows, columns=_SIGNALS_COLUMNS)


def generate_forward_intelligence_signals(
    attrib_df: pd.DataFrame,
    governance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate observational forward intelligence signals.

    This function is read-only and observational. It does not create
    governance decisions, modify allocations, execute logic, or change
    existing workflows.

    Parameters
    ----------
    attrib_df:
        Alpha attribution summary dataframe loaded from
        ``data/alpha_attribution_summary.csv``.
    governance_df:
        Governance decisions dataframe from
        ``helpers.governance_lifecycle.load_governance_decisions_df``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``signal_id``, ``wave``, ``signal_type``,
        ``signal_title``, ``observation``, ``confidence``, ``horizon``,
        ``created_at``.  Returns an empty DataFrame (not None) when no
        signals are detected.
    """
    signals: list[dict] = []
    now = datetime.now(tz=timezone.utc).isoformat()

    if attrib_df is not None and not attrib_df.empty:
        df = attrib_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        has_wave = "wave" in df.columns
        has_horizon = "horizon" in df.columns
        has_vol = "volatility_alpha" in df.columns
        has_total = "total_alpha" in df.columns
        has_momentum = "momentum_alpha" in df.columns

        if has_wave and has_horizon and has_vol and has_total:
            # ----------------------------------------------------------------
            # Rule A — Earnings Preparation Window
            # If volatility_alpha increasing AND 30D alpha deteriorating
            # ----------------------------------------------------------------
            try:
                for wave_name, wave_df in df.groupby("wave"):
                    wave_df = wave_df.sort_values("horizon")
                    horizons = wave_df["horizon"].tolist()

                    if 30 not in horizons:
                        continue

                    row_30 = wave_df[wave_df["horizon"] == 30]
                    total_30 = float(row_30["total_alpha"].iloc[0])
                    vol_30 = float(row_30["volatility_alpha"].iloc[0])

                    # Compare to a longer horizon to detect "increasing" vol alpha
                    longer = [h for h in horizons if h > 30]
                    if not longer:
                        continue
                    row_long = wave_df[wave_df["horizon"] == min(longer)]
                    vol_long = float(row_long["volatility_alpha"].iloc[0])

                    vol_increasing = vol_30 > vol_long
                    alpha_deteriorating = total_30 < 0

                    if vol_increasing and alpha_deteriorating:
                        signals.append({
                            "signal_id": str(uuid.uuid4()),
                            "wave": wave_name,
                            "signal_type": "Earnings Conditioning",
                            "signal_title": "Earnings Preparation Window",
                            "observation": (
                                f"{wave_name}: volatility alpha rising "
                                f"({vol_30:.4f} vs {vol_long:.4f} at longer horizon) "
                                f"with deteriorating 30-day alpha ({total_30:.4f}). "
                                "Earnings conditioning conditions present."
                            ),
                            "confidence": "Moderate",
                            "horizon": "30D",
                            "created_at": now,
                        })
            except Exception:
                pass

        if has_wave and has_horizon and has_total:
            # ----------------------------------------------------------------
            # Rule B — Position Conditioning
            # If long-term alpha positive AND short-term alpha negative
            # ----------------------------------------------------------------
            try:
                for wave_name, wave_df in df.groupby("wave"):
                    wave_df = wave_df.sort_values("horizon")
                    horizons = sorted(wave_df["horizon"].tolist())

                    short_horizons = [h for h in horizons if h <= 30]
                    long_horizons = [h for h in horizons if h >= 60]

                    if not short_horizons or not long_horizons:
                        continue

                    row_short = wave_df[wave_df["horizon"] == min(short_horizons)]
                    row_long = wave_df[wave_df["horizon"] == max(long_horizons)]

                    alpha_short = float(row_short["total_alpha"].iloc[0])
                    alpha_long = float(row_long["total_alpha"].iloc[0])

                    if alpha_long > 0 and alpha_short < 0:
                        signals.append({
                            "signal_id": str(uuid.uuid4()),
                            "wave": wave_name,
                            "signal_type": "Short-Term Dislocation",
                            "signal_title": "Position Conditioning",
                            "observation": (
                                f"{wave_name}: long-term alpha positive "
                                f"({alpha_long:.4f} at {max(long_horizons)}D) "
                                f"diverging from negative short-term alpha "
                                f"({alpha_short:.4f} at {min(short_horizons)}D). "
                                "Short-term dislocation observed."
                            ),
                            "confidence": "High",
                            "horizon": f"{min(short_horizons)}D–{max(long_horizons)}D",
                            "created_at": now,
                        })
            except Exception:
                pass

        if has_wave and has_horizon and has_total:
            # ----------------------------------------------------------------
            # Rule C — Governance Pressure Build
            # If ≥2 waves show same attribution deterioration
            # ----------------------------------------------------------------
            try:
                deteriorating_waves = []
                for wave_name, wave_df in df.groupby("wave"):
                    wave_df = wave_df.sort_values("horizon")
                    horizons = sorted(wave_df["horizon"].tolist())
                    if len(horizons) < 2:
                        continue
                    first_alpha = float(
                        wave_df[wave_df["horizon"] == horizons[0]]["total_alpha"].iloc[0]
                    )
                    last_alpha = float(
                        wave_df[wave_df["horizon"] == horizons[-1]]["total_alpha"].iloc[0]
                    )
                    if first_alpha < last_alpha < 0:
                        deteriorating_waves.append(wave_name)

                if len(deteriorating_waves) >= 2:
                    signals.append({
                        "signal_id": str(uuid.uuid4()),
                        "wave": ", ".join(deteriorating_waves[:3])
                        + ("..." if len(deteriorating_waves) > 3 else ""),
                        "signal_type": "Cluster Formation",
                        "signal_title": "Governance Pressure Build",
                        "observation": (
                            f"{len(deteriorating_waves)} waves show correlated attribution "
                            "deterioration across horizons. Cluster formation observed: "
                            + ", ".join(deteriorating_waves[:3])
                            + ("..." if len(deteriorating_waves) > 3 else "")
                            + "."
                        ),
                        "confidence": "Moderate",
                        "horizon": "Multi-Horizon",
                        "created_at": now,
                    })
            except Exception:
                pass

        if has_wave and has_horizon and has_momentum:
            # ----------------------------------------------------------------
            # Rule D — Emerging Momentum Shift
            # If momentum_alpha flips sign between horizons
            # ----------------------------------------------------------------
            try:
                for wave_name, wave_df in df.groupby("wave"):
                    wave_df = wave_df.sort_values("horizon")
                    horizons = sorted(wave_df["horizon"].tolist())
                    if len(horizons) < 2:
                        continue

                    mom_values = []
                    for h in horizons:
                        row = wave_df[wave_df["horizon"] == h]
                        mom_values.append((h, float(row["momentum_alpha"].iloc[0])))

                    # Check for sign flip across consecutive horizons
                    for i in range(len(mom_values) - 1):
                        h1, m1 = mom_values[i]
                        h2, m2 = mom_values[i + 1]
                        if m1 != 0 and m2 != 0 and (m1 > 0) != (m2 > 0):
                            signals.append({
                                "signal_id": str(uuid.uuid4()),
                                "wave": wave_name,
                                "signal_type": "Momentum Transition",
                                "signal_title": "Emerging Momentum Shift",
                                "observation": (
                                    f"{wave_name}: momentum alpha flips from "
                                    f"{m1:.4f} at {h1}D to {m2:.4f} at {h2}D. "
                                    "Momentum transition signal detected."
                                ),
                                "confidence": "Low",
                                "horizon": f"{h1}D→{h2}D",
                                "created_at": now,
                            })
                            break
            except Exception:
                pass

    if not signals:
        return _empty_df()

    return pd.DataFrame(signals, columns=_REQUIRED_COLUMNS)
