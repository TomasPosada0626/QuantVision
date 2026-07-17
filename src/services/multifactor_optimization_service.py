from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultifactorConfig:
    factor_weights: dict[str, float]
    top_n: int = 5
    risk_aversion: float = 3.0
    long_only: bool = True


class MultifactorOptimizationService:
    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        std = float(series.std(ddof=0))
        if std == 0:
            return pd.Series(0.0, index=series.index)
        return (series - float(series.mean())) / std

    def compute_composite_scores(
        self,
        factor_frame: pd.DataFrame,
        factor_weights: dict[str, float],
    ) -> pd.Series:
        if factor_frame.empty:
            raise ValueError("factor_frame cannot be empty")

        missing = [factor for factor in factor_weights if factor not in factor_frame.columns]
        if missing:
            raise ValueError(f"missing factor columns: {missing}")

        score = pd.Series(0.0, index=factor_frame.index, dtype=float)
        total_weight = 0.0
        for factor, weight in factor_weights.items():
            score = score + self._zscore(factor_frame[factor].astype(float)) * float(weight)
            total_weight += abs(float(weight))

        if total_weight > 0:
            score = score / total_weight
        return score.sort_values(ascending=False)

    @staticmethod
    def optimize_weights(
        returns_frame: pd.DataFrame,
        risk_aversion: float = 3.0,
        long_only: bool = True,
    ) -> pd.Series:
        if returns_frame.empty:
            raise ValueError("returns_frame cannot be empty")

        returns = returns_frame.dropna(axis=0, how="any")
        if returns.empty:
            raise ValueError("returns_frame has no valid rows after dropping NaNs")

        mu = returns.mean(axis=0).to_numpy(dtype=float)
        cov = returns.cov().to_numpy(dtype=float)
        n_assets = cov.shape[0]

        cov_reg = cov + np.eye(n_assets) * 1e-6
        try:
            inv_cov = np.linalg.pinv(cov_reg)
            raw = inv_cov @ mu
        except np.linalg.LinAlgError:
            raw = np.ones(n_assets, dtype=float)

        raw = raw / max(float(risk_aversion), 1e-6)

        if long_only:
            raw = np.clip(raw, 0.0, None)

        if float(raw.sum()) <= 0:
            raw = np.ones(n_assets, dtype=float)

        weights = raw / float(raw.sum())
        return pd.Series(weights, index=returns.columns)

    def build_optimized_portfolio(
        self,
        price_frame: pd.DataFrame,
        factor_frame: pd.DataFrame,
        config: MultifactorConfig,
    ) -> dict[str, object]:
        scores = self.compute_composite_scores(
            factor_frame=factor_frame, factor_weights=config.factor_weights
        )

        top_n = max(1, int(config.top_n))
        selected_assets = scores.head(top_n).index.tolist()

        missing_prices = [asset for asset in selected_assets if asset not in price_frame.columns]
        if missing_prices:
            raise ValueError(f"missing asset prices for selected assets: {missing_prices}")

        selected_prices = price_frame[selected_assets].copy()
        returns = selected_prices.pct_change().dropna(axis=0, how="any")
        weights = self.optimize_weights(
            returns_frame=returns,
            risk_aversion=config.risk_aversion,
            long_only=config.long_only,
        )

        return {
            "selected_assets": selected_assets,
            "scores": scores.loc[selected_assets].to_dict(),
            "weights": weights.to_dict(),
        }
