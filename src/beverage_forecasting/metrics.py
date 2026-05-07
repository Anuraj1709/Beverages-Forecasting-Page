from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = actual.astype(float).to_numpy()
    predicted_values = predicted.astype(float).to_numpy()
    errors = actual_values - predicted_values
    non_zero = np.where(actual_values == 0, np.nan, actual_values)
    mape = np.nanmean(np.abs(errors / non_zero)) * 100
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.nan_to_num(mape, nan=np.inf)),
    }
