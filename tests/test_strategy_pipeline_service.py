import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from services.strategy_pipeline_service import StrategyPipelineService


def test_promote_and_run_notebook_strategy(tmp_path) -> None:
    service = StrategyPipelineService(output_dir=str(tmp_path / "strategies"))

    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    signals = pd.DataFrame(
        {
            "Close": [100, 98, 102, 105, 101, 108],
            "buy_signal": [False, True, False, False, False, False],
            "sell_signal": [False, False, False, True, False, False],
        },
        index=idx,
    )

    manifest = service.promote_notebook_strategy(
        strategy_name="Mean Reversion Notebook",
        notebook_path="notebooks/mean_reversion.ipynb",
        signals_df=signals,
    )

    assert manifest.exists()
    payload = service.load_artifact(manifest)
    assert payload["strategy_name"] == "Mean Reversion Notebook"

    result = service.run_promoted_strategy(manifest)
    assert "Return %" in result
    assert "Trades" in result


def test_promote_notebook_strategy_validates_columns(tmp_path) -> None:
    service = StrategyPipelineService(output_dir=str(tmp_path / "strategies"))
    bad_df = pd.DataFrame({"Close": [100, 101]})

    error = None
    try:
        service.promote_notebook_strategy(
            strategy_name="Bad",
            notebook_path="notebooks/bad.ipynb",
            signals_df=bad_df,
        )
    except ValueError as exc:
        error = str(exc)

    assert error is not None
    assert "buy/sell signal columns" in error
