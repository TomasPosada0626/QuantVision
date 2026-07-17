import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from services.multifactor_optimization_service import (
    MultifactorConfig,
    MultifactorOptimizationService,
)


def test_multifactor_optimizer_builds_weights_and_selection() -> None:
    svc = MultifactorOptimizationService()

    factor_frame = pd.DataFrame(
        {
            "momentum": {"AAPL": 0.8, "MSFT": 0.6, "NVDA": 0.9, "JPM": 0.2},
            "quality": {"AAPL": 0.7, "MSFT": 0.9, "NVDA": 0.5, "JPM": 0.6},
            "value": {"AAPL": 0.4, "MSFT": 0.3, "NVDA": 0.1, "JPM": 0.9},
        }
    )

    idx = pd.date_range("2025-01-01", periods=8, freq="D")
    price_frame = pd.DataFrame(
        {
            "AAPL": [100, 101, 103, 102, 104, 105, 107, 108],
            "MSFT": [90, 91, 92, 93, 94, 95, 96, 97],
            "NVDA": [200, 205, 210, 220, 215, 225, 230, 235],
            "JPM": [70, 69, 70, 71, 72, 73, 73, 74],
        },
        index=idx,
    )

    config = MultifactorConfig(
        factor_weights={"momentum": 0.5, "quality": 0.3, "value": -0.2},
        top_n=3,
        risk_aversion=3.0,
        long_only=True,
    )

    result = svc.build_optimized_portfolio(
        price_frame=price_frame,
        factor_frame=factor_frame,
        config=config,
    )

    assert len(result["selected_assets"]) == 3
    weights = result["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(weight >= 0 for weight in weights.values())


def test_multifactor_optimizer_validates_missing_factors() -> None:
    svc = MultifactorOptimizationService()
    factor_frame = pd.DataFrame({"momentum": {"AAPL": 1.0}})

    error = None
    try:
        svc.compute_composite_scores(factor_frame, {"momentum": 1.0, "quality": 1.0})
    except ValueError as exc:
        error = str(exc)

    assert error is not None
    assert "missing factor columns" in error
