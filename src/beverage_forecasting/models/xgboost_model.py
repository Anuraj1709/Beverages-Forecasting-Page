from __future__ import annotations

import pandas as pd

from ..config import ForecastConfig
from ..exceptions import ModelUnavailableError
from ..features import FeatureMetadata, make_next_feature_row, make_supervised_frame
from .base import ForecastModel


class XGBoostLagModel(ForecastModel):
    name = "xgboost"

    def __init__(self, config: ForecastConfig) -> None:
        self.config = config
        self._model = None
        self._history: pd.Series | None = None
        self._metadata: FeatureMetadata | None = None

    def fit(self, series: pd.Series, state: str) -> "XGBoostLagModel":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ModelUnavailableError("xgboost is required for XGBoost forecasts.") from exc

        supervised, metadata = make_supervised_frame(series, self.config)
        self._metadata = metadata
        self._history = series.astype(float).copy()
        self._model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        self._model.fit(supervised[metadata.columns], supervised["y"])
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._model is None or self._history is None or self._metadata is None:
            raise RuntimeError("Model must be fitted before prediction.")

        history = self._history.copy()
        predictions = []
        for forecast_date in pd.date_range(
            history.index[-1] + pd.tseries.frequencies.to_offset(self.config.frequency),
            periods=horizon,
            freq=self.config.frequency,
        ):
            features = make_next_feature_row(history, forecast_date, self.config)
            yhat = float(self._model.predict(features[self._metadata.columns])[0])
            yhat = max(yhat, 0.0)
            predictions.append((forecast_date, yhat))
            history.loc[forecast_date] = yhat

        return pd.Series({date: value for date, value in predictions}, name=self.name)
