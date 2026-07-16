import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ui.charts import build_anomaly_chart, build_price_chart


def test_build_price_chart_returns_figure_and_series() -> None:
    df = pd.DataFrame(
        {"Close": [10, 11, 12]}, index=pd.date_range("2025-01-01", periods=3, freq="D")
    )
    fig, y_data = build_price_chart(df, "AAPL")
    assert len(fig.data) >= 1
    assert len(y_data) == 3


def test_build_anomaly_chart_contains_points_and_line() -> None:
    df = pd.DataFrame(
        {"Close": [10, 11, 12, 13]}, index=pd.date_range("2025-01-01", periods=4, freq="D")
    )
    pts = df.iloc[[1, 3]].copy()
    pts["Method"] = ["Z-Score", "I-Forest"]
    y_data = df["Close"]

    fig = build_anomaly_chart(df, pts, y_data)
    assert len(fig.data) >= 2
