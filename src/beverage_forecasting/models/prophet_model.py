from __future__ import annotations

import pandas as pd

from ..config import ForecastConfig
from ..exceptions import ModelUnavailableError
from .base import ForecastModel


class ProphetForecastModel(ForecastModel):
    name = "prophet"

    def __init__(self, config: ForecastConfig) -> None:
        self.config = config
        self._model = None
        self._last_index: pd.Timestamp | None = None

    def fit(self, series: pd.Series, state: str) -> "ProphetForecastModel":
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ModelUnavailableError("prophet is required for Prophet forecasts.") from exc

        frame = series.reset_index()
        frame.columns = ["ds", "y"]
        self._model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        self._model.add_country_holidays(country_name=self.config.country_holidays)
        self._model.fit(frame)
        self._last_index = series.index[-1]
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._model is None or self._last_index is None:
            raise RuntimeError("Model must be fitted before prediction.")
        index = pd.date_range(
            self._last_index + pd.tseries.frequencies.to_offset(self.config.frequency),
            periods=horizon,
            freq=self.config.frequency,
        )
        future = pd.DataFrame({"ds": index})
        forecast = self._model.predict(future)
        return pd.Series(forecast["yhat"].to_numpy(), index=index, name=self.name).clip(lower=0)
