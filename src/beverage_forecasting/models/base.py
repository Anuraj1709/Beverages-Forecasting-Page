from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class FittedForecast:
    model_name: str
    state: str
    validation_metrics: dict[str, float]
    validation_predictions: pd.Series
    model: "ForecastModel"


class ForecastModel(ABC):
    name: str

    @abstractmethod
    def fit(self, series: pd.Series, state: str) -> "ForecastModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        raise NotImplementedError
