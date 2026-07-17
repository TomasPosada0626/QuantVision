from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from anomaly_methods import detect_anomalies_zscore
from security.input_validation import sanitize_csv_upload, sanitize_ticker
from services.market_data_service import get_ticker_data


def test_anomaly_empty_series() -> None:
    """All methods should return an empty/false-compatible result for empty input."""
    empty = pd.Series(dtype=float)
    result = detect_anomalies_zscore(empty)
    assert result.size == 0


def test_anomaly_all_nan() -> None:
    """All-NaN inputs should not raise and should produce no positive anomalies."""
    all_nan = pd.Series([np.nan, np.nan, np.nan])
    result = detect_anomalies_zscore(all_nan)
    assert np.all(~result.fillna(False).to_numpy())


def test_anomaly_single_value() -> None:
    """Single-value series should preserve shape and avoid exceptions."""
    single = pd.Series([1.0])
    result = detect_anomalies_zscore(single)
    assert len(result) == 1


def test_invalid_csv_upload() -> None:
    """Suspicious csv schema should be rejected gracefully."""
    bad_csv = "script,Close\nalert(1),100\n"
    with pytest.raises(ValueError):
        sanitize_csv_upload(bad_csv)


def test_invalid_ticker() -> None:
    """Invalid ticker symbols should be rejected by validation boundary."""
    with pytest.raises(ValueError):
        sanitize_ticker("XXXINVALID$$")


def test_market_data_download_failure_returns_empty(tmp_path) -> None:
    """Download failures should surface as an empty frame with warning flag."""

    def _empty_download(*args, **kwargs):
        return pd.DataFrame()

    frame, warning, _ = get_ticker_data(
        "AAPL",
        date(2025, 1, 1),
        date(2025, 1, 10),
        data_dir=str(tmp_path / "market_data"),
        download_fn=_empty_download,
    )
    assert frame.empty
    assert warning is True


def test_negative_threshold() -> None:
    """Negative threshold should raise a validation error."""
    with pytest.raises(ValueError):
        detect_anomalies_zscore(pd.Series([1, 2, 3]), threshold=-1)


def test_large_dataset_performance() -> None:
    """A 10K-row Z-Score pass should complete quickly for interactive use."""
    large_returns = pd.Series(np.random.randn(10_000))
    start = time.time()
    _ = detect_anomalies_zscore(large_returns)
    elapsed = time.time() - start
    assert elapsed < 1.0
