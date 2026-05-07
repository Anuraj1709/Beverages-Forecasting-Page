from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ForecastConfig


@dataclass
class FeatureMetadata:
    columns: list[str]
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]


def make_supervised_frame(series: pd.Series, config: ForecastConfig) -> tuple[pd.DataFrame, FeatureMetadata]:
    frame = pd.DataFrame({"y": series.astype(float)})
    frame = add_calendar_features(frame, config)

    for lag in config.lags:
        frame[f"lag_t_{lag}"] = frame["y"].shift(lag)

    for window in config.rolling_windows:
        shifted = frame["y"].shift(1)
        frame[f"rolling_mean_{window}"] = shifted.rolling(window=window, min_periods=2).mean()
        frame[f"rolling_std_{window}"] = shifted.rolling(window=window, min_periods=2).std()

    frame = frame.dropna()
    columns = [column for column in frame.columns if column != "y"]
    return frame, FeatureMetadata(columns, config.lags, config.rolling_windows)


def add_calendar_features(frame: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    enriched = frame.copy()
    index = pd.DatetimeIndex(enriched.index)
    enriched["day_of_week"] = index.dayofweek
    enriched["month"] = index.month
    enriched["quarter"] = index.quarter
    enriched["week_of_year"] = index.isocalendar().week.astype(int).to_numpy()
    enriched["is_month_start"] = index.is_month_start.astype(int)
    enriched["is_month_end"] = index.is_month_end.astype(int)
    enriched["holiday_flag"] = holiday_flags(index, config).astype(int)
    return enriched


def holiday_flags(index: pd.DatetimeIndex, config: ForecastConfig) -> np.ndarray:
    try:
        import holidays

        calendar = holidays.country_holidays(config.country_holidays, years=range(index.min().year, index.max().year + 1))
        return np.array([_week_has_holiday(date, calendar) for date in index], dtype=bool)
    except Exception:
        fixed_mm_dd = {"01-01", "07-04", "11-11", "12-25"}
        return np.array([date.strftime("%m-%d") in fixed_mm_dd for date in index], dtype=bool)


def _week_has_holiday(date: pd.Timestamp, calendar: object) -> bool:
    week_start = date - pd.Timedelta(days=6)
    return any(day.date() in calendar for day in pd.date_range(week_start, date, freq="D"))


def make_next_feature_row(history: pd.Series, forecast_date: pd.Timestamp, config: ForecastConfig) -> pd.DataFrame:
    values = history.astype(float).copy()
    row = pd.DataFrame(index=[forecast_date])
    row = add_calendar_features(row, config)
    for lag in config.lags:
        row[f"lag_t_{lag}"] = values.iloc[-lag] if len(values) >= lag else values.iloc[-1]
    for window in config.rolling_windows:
        tail = values.iloc[-window:]
        row[f"rolling_mean_{window}"] = tail.mean()
        row[f"rolling_std_{window}"] = tail.std(ddof=1) if len(tail) > 1 else 0.0
    return row.fillna(0.0)
