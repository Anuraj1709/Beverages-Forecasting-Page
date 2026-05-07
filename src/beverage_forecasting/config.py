from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ForecastConfig:
    date_col: str = "Date"
    state_col: str = "State"
    target_col: str = "Total"
    category_col: str = "Category"
    frequency: str = "W-SUN"
    horizon: int = 8
    validation_size: int = 8
    lags: tuple[int, ...] = (1, 7, 30)
    rolling_windows: tuple[int, ...] = (4, 8, 12)
    min_train_points: int = 40
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    country_holidays: str = "US"


DEFAULT_CONFIG = ForecastConfig()
