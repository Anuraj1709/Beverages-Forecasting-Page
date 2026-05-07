from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ForecastConfig
from ..exceptions import ModelUnavailableError
from .base import ForecastModel


class LSTMForecastModel(ForecastModel):
    name = "lstm"

    def __init__(self, config: ForecastConfig, lookback: int = 30, epochs: int = 30) -> None:
        self.config = config
        self.lookback = lookback
        self.epochs = epochs
        self._model = None
        self._history: pd.Series | None = None
        self._mean = 0.0
        self._std = 1.0

    def fit(self, series: pd.Series, state: str) -> "LSTMForecastModel":
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.models import Sequential
        except ImportError as exc:
            raise ModelUnavailableError("tensorflow is required for LSTM forecasts.") from exc

        values = series.astype(float).to_numpy()
        self._mean = float(values.mean())
        self._std = float(values.std() or 1.0)
        scaled = (values - self._mean) / self._std
        x_train, y_train = self._windows(scaled)
        self._history = series.astype(float).copy()

        model = Sequential(
            [
                LSTM(32, input_shape=(self.lookback, 1)),
                Dropout(0.1),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
        model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=16,
            verbose=0,
            callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
        )
        self._model = model
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._model is None or self._history is None:
            raise RuntimeError("Model must be fitted before prediction.")
        values = self._history.astype(float).to_numpy().tolist()
        predictions = []
        index = pd.date_range(
            self._history.index[-1] + pd.tseries.frequencies.to_offset(self.config.frequency),
            periods=horizon,
            freq=self.config.frequency,
        )
        for forecast_date in index:
            window = np.array(values[-self.lookback:], dtype=float)
            scaled = ((window - self._mean) / self._std).reshape(1, self.lookback, 1)
            yhat = float(self._model.predict(scaled, verbose=0)[0][0] * self._std + self._mean)
            yhat = max(yhat, 0.0)
            predictions.append(yhat)
            values.append(yhat)
        return pd.Series(predictions, index=index, name=self.name)

    def _windows(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_rows, y_rows = [], []
        for idx in range(self.lookback, len(values)):
            x_rows.append(values[idx - self.lookback : idx])
            y_rows.append(values[idx])
        x = np.array(x_rows).reshape(-1, self.lookback, 1)
        y = np.array(y_rows)
        return x, y
