from typing import Optional

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def _coerce_returns(returns: Series | None) -> Series:
    """Normalize optional return series into a non-null pandas Series.

    Args:
        returns: Optional return series.

    Returns:
        A pandas Series; empty when input is None.
    """
    if returns is None:
        return pd.Series(dtype=float)
    return returns


def detect_anomalies_zscore(returns: Series | None, threshold: float = 3.0) -> Series:
    """Detect anomalies using Z-Score.

    Z-Score estimates how many standard deviations each value is from the
    sample mean. Values beyond the configured threshold are flagged.

    Args:
        returns: Series of returns (or any numeric observations).
        threshold: Positive z-score cutoff used to mark anomalies.

    Returns:
        Boolean pandas Series where True means anomaly detected.

    Raises:
        ValueError: If threshold is not strictly positive.
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    returns_series = _coerce_returns(returns)
    result = pd.Series(False, index=returns_series.index)
    mask = returns_series.notna()
    clean = returns_series[mask]
    if clean.empty:
        return result

    mean = clean.mean()
    std = clean.std()
    if std == 0 or np.isnan(std):
        return result

    # An anomaly is any value more than 'threshold' std deviations from the mean.
    result[mask] = np.abs(clean - mean) > threshold * std
    return result


def detect_anomalies_iforest(
    returns: Series | None, contamination: float = 0.01, random_state: Optional[int] = 42
) -> Series:
    """Detect anomalies using Isolation Forest.

    Args:
        returns: Series of returns (or numeric observations).
        contamination: Expected anomaly ratio in (0, 0.5).
        random_state: Random seed for reproducibility.

    Returns:
        Boolean pandas Series where True means anomaly detected.

    Raises:
        ValueError: If contamination is outside (0, 0.5).
    """
    if contamination <= 0 or contamination >= 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    returns_series = _coerce_returns(returns)
    mask = returns_series.notna()
    result = pd.Series(False, index=returns_series.index)
    clean = returns_series[mask]
    if clean.empty:
        return result

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(clean.values.reshape(-1, 1))
    # Mark as anomaly where prediction is -1
    result[mask] = preds == -1
    return result


def detect_anomalies_lof(
    returns: Series | None, contamination: float = 0.01, n_neighbors: int = 20
) -> Series:
    """Detect anomalies using Local Outlier Factor.

    Args:
        returns: Series of returns (or numeric observations).
        contamination: Expected anomaly ratio in (0, 0.5).
        n_neighbors: Number of neighbors used by LOF.

    Returns:
        Boolean pandas Series where True means anomaly detected.

    Raises:
        ValueError: If contamination or n_neighbors are invalid.
    """
    if contamination <= 0 or contamination >= 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be >= 2")
    returns_series = _coerce_returns(returns)
    mask = returns_series.notna()
    clean = returns_series[mask]
    result = pd.Series(False, index=returns_series.index)
    if clean.empty or len(clean) < 3:
        return result
    # LOF requires n_neighbors < n_samples.
    safe_neighbors = min(max(2, n_neighbors), len(clean) - 1)
    lof = LocalOutlierFactor(n_neighbors=safe_neighbors, contamination=contamination)
    preds = lof.fit_predict(clean.values.reshape(-1, 1))
    result[mask] = preds == -1
    return result


def detect_anomalies_one_class_svm(
    returns: Series | None, nu: float = 0.05, gamma: str = "scale"
) -> Series:
    """Detect anomalies using One-Class SVM.

    Args:
        returns: Series of returns (or numeric observations).
        nu: Upper bound on anomaly fraction in (0, 1].
        gamma: Kernel coefficient strategy/value.

    Returns:
        Boolean pandas Series where True means anomaly detected.

    Raises:
        ValueError: If nu is outside (0, 1].
    """
    if nu <= 0 or nu > 1:
        raise ValueError("nu must be in (0, 1]")
    returns_series = _coerce_returns(returns)
    mask = returns_series.notna()
    clean = returns_series[mask]
    result = pd.Series(False, index=returns_series.index)
    if clean.empty or len(clean) < 3:
        return result
    model = OneClassSVM(nu=nu, gamma=gamma)
    preds = model.fit_predict(clean.values.reshape(-1, 1))
    result[mask] = preds == -1
    return result
