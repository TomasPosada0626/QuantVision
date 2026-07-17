import pandas as pd


def rolling_quantile_anomaly(
    series: pd.Series, window: int = 20, quantile: float = 0.99
) -> pd.Series:
    """Detect anomalies using an adaptive rolling quantile threshold.

    Args:
        series: Input time series to evaluate.
        window: Rolling lookback window size.
        quantile: Quantile threshold in (0, 1]; values above this dynamic
            threshold are flagged as anomalies.

    Returns:
        Boolean Series where True indicates an anomaly.

    Raises:
        ValueError: If window <= 0 or quantile is outside (0, 1].
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    if quantile <= 0 or quantile > 1:
        raise ValueError("quantile must be in (0, 1]")

    threshold = series.rolling(window=window, min_periods=1).quantile(quantile)
    return series > threshold
