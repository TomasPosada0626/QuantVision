from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from analytics.event_tracker import AnalyticsEvent, EventTracker
from security.input_validation import sanitize_csv_upload, sanitize_ticker
from services.indicators_service import add_indicators
from services.market_data_service import add_return_features, get_ticker_data
from utils import rolling_quantile_anomaly


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary OHLCV frame to expected numeric columns.

    Args:
        df: Raw frame from API/download/csv.

    Returns:
        Normalized frame containing `Open`, `High`, `Low`, `Close`, `Volume`.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [col[0] if isinstance(col, tuple) else col for col in out.columns]
    if "Close" not in out.columns and len(out.columns) > 0:
        out["Close"] = pd.to_numeric(out.iloc[:, 0], errors="coerce")
    out["Open"] = pd.to_numeric(out.get("Open", out["Close"]), errors="coerce")
    out["High"] = pd.to_numeric(out.get("High", out["Close"]), errors="coerce")
    out["Low"] = pd.to_numeric(out.get("Low", out["Close"]), errors="coerce")
    out["Volume"] = pd.to_numeric(out.get("Volume", 0), errors="coerce").fillna(0)
    return out


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_ohlcv(frame)
    normalized = add_return_features(normalized)
    normalized = add_indicators(normalized)
    normalized["Anomaly_rolling_quantile_base"] = rolling_quantile_anomaly(
        normalized["Close"], window=20, quantile=0.95
    )
    return normalized


def load_market_data(
    tickers: list[str],
    start_date: date,
    end_date: date,
    uploaded_file: Any,
    event_tracker: EventTracker,
    username: str,
) -> dict[str, pd.DataFrame]:
    """Load market data from csv and ticker providers with caching.

    Args:
        tickers: Tickers to load from provider.
        start_date: Query start date.
        end_date: Query end date.
        uploaded_file: Optional uploaded csv file object.
        event_tracker: Analytics event tracker.
        username: User associated with events.

    Returns:
        Mapping from ticker key to modeled dataframe.
    """
    results: dict[str, pd.DataFrame] = {}

    @st.cache_data(show_spinner=False, ttl=60 * 30)
    def _cached_prepare_ticker(ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
        frame, _, warning_message = get_ticker_data(
            ticker=ticker,
            start_date=start_str,
            end_date=end_str,
            data_dir="data",
        )
        if warning_message or frame.empty:
            return pd.DataFrame()
        return _prepare_features(frame)

    if uploaded_file is not None:
        try:
            csv_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
            custom_df = sanitize_csv_upload(csv_text)
            if "Date" in custom_df.columns:
                custom_df = custom_df.set_index("Date")
            custom_df.index = pd.to_datetime(custom_df.index)
            results["USER_CSV"] = _prepare_features(custom_df)
            event_tracker.track(
                AnalyticsEvent(
                    username=username,
                    feature="market_data",
                    event_name="load_market_data",
                    metadata="source=csv",
                )
            )
        except Exception as csv_error:
            st.sidebar.error(f"CSV parse error: {csv_error}")

    for ticker in tickers:
        try:
            safe_ticker = sanitize_ticker(ticker)
            normalized = _cached_prepare_ticker(safe_ticker, str(start_date), str(end_date))
            if normalized.empty:
                continue
            results[safe_ticker] = normalized
            event_tracker.track(
                AnalyticsEvent(
                    username=username,
                    feature="market_data",
                    event_name="load_market_data",
                    metadata=f"source=ticker;ticker={safe_ticker}",
                )
            )
        except Exception as data_error:
            st.error(f"Data load failed for {ticker}: {data_error}")

    return results


@st.cache_data(show_spinner=False, ttl=3600)
def load_ticker_lazy(ticker: str, years: int = 5) -> pd.DataFrame:
    """Load only a recent history window for one ticker.

    Args:
        ticker: Ticker symbol.
        years: Years of history to fetch.

    Returns:
        Modeled ticker dataframe.
    """
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365 * years)
    frame, _, warning_message = get_ticker_data(
        ticker=ticker,
        start_date=str(start_date),
        end_date=str(end_date),
        data_dir="data",
    )
    if warning_message or frame.empty:
        return pd.DataFrame()
    return _prepare_features(frame)
