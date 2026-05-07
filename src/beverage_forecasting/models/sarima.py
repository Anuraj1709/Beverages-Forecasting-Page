from __future__ import annotations

import warnings

import pandas as pd

from ..config import ForecastConfig
from ..exceptions import ModelUnavailableError
from .base import ForecastModel


class SarimaModel(ForecastModel):
    name = "sarima"

    def __init__(
        self,
        config: ForecastConfig,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 52),
    ) -> None:
        self.config = config
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None
        self._last_index: pd.Timestamp | None = None

    def fit(self, series: pd.Series, state: str) -> "SarimaModel":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ModelUnavailableError("statsmodels is required for SARIMA.") from exc

        self._last_index = series.index[-1]
        seasonal_order = self.seasonal_order
        if len(series) < seasonal_order[-1] * 2:
            seasonal_order = (1, 0, 1, 12)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series.astype(float),
                order=self.order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = model.fit(disp=False)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None or self._last_index is None:
            raise RuntimeError("Model must be fitted before prediction.")
        index = pd.date_range(
            self._last_index + pd.tseries.frequencies.to_offset(self.config.frequency),
            periods=horizon,
            freq=self.config.frequency,
        )
        values = self._result.forecast(steps=horizon)
        return pd.Series(values.to_numpy(), index=index, name=self.name).clip(lower=0)
