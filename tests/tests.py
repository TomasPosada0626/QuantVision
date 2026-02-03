
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pandas as pd
from utils import rolling_quantile_anomaly
from anomaly_methods import detect_anomalies_zscore, detect_anomalies_iforest

def test_zscore_anomalies():
    returns = pd.Series([0.01, 0.02, 0.03, 0.5, -0.6, 0.02, 0.01])
    anomalies = detect_anomalies_zscore(returns, threshold=1.5)
    # Should detect the large outliers
    assert anomalies.sum() == 2

def test_iforest_anomalies():
    returns = pd.Series([0.01, 0.02, 0.03, 0.5, -0.6, 0.02, 0.01])
    anomalies = detect_anomalies_iforest(returns, contamination=0.2)
    # Should detect at least the two large outliers
    assert anomalies.sum() >= 2

def test_zscore_no_anomalies():
    """Test Z-Score method with no anomalies present."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.02, 0.01])
    anomalies = detect_anomalies_zscore(returns, threshold=3)
    assert anomalies.sum() == 0

def test_iforest_all_normal():
    """Test Isolation Forest with all normal data."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.02, 0.01])
    anomalies = detect_anomalies_iforest(returns, contamination=0.2)
    assert anomalies.sum() <= 1  # At most 1 outlier due to contamination

def test_rolling_quantile_anomaly():
    """Test rolling quantile anomaly detection."""
    s = pd.Series([1, 2, 3, 100, 5, 6, 7, 8, 9, 10])
    anomalies = rolling_quantile_anomaly(s, window=3, quantile=0.95)
    assert anomalies.sum() >= 1


