import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from services.market_data_service import add_return_features, get_ticker_data


def test_get_ticker_data_uses_cache_when_range_covered(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "AAPL_10y.csv"

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            ("Close", "AAPL"): [10, 11, 12, 13, 14],
            ("Open", "AAPL"): [9, 10, 11, 12, 13],
        },
        index=idx,
    )
    df.to_csv(csv_path)

    downloaded_df, downloaded, warning = get_ticker_data(
        ticker="AAPL",
        start_date=idx.min(),
        end_date=idx.max(),
        data_dir=str(data_dir),
        download_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("should not download")
        ),
    )

    assert downloaded is False
    assert warning is None
    assert "Close" in downloaded_df.columns


def test_get_ticker_data_downloads_when_cache_missing(tmp_path) -> None:
    data_dir = tmp_path / "data"
    idx = pd.date_range("2024-01-01", periods=5, freq="D")

    def fake_download(*args, **kwargs):
        return pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=idx)

    downloaded_df, downloaded, warning = get_ticker_data(
        ticker="MSFT",
        start_date=idx.min(),
        end_date=idx.max(),
        data_dir=str(data_dir),
        download_fn=fake_download,
    )

    assert downloaded is True
    assert warning is None
    assert not downloaded_df.empty
    assert (data_dir / "MSFT_10y.csv").exists()


def test_add_return_features_adds_numeric_close_and_return() -> None:
    df = pd.DataFrame({"Close": [100, 102, 101]})
    result = add_return_features(df)
    assert "Return" in result.columns
    assert result["Close"].dtype.kind in {"i", "u", "f"}
    assert result["Return"].isna().sum() == 1
