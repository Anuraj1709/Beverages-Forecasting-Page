from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ForecastConfig
from .data import list_states, load_sales_data, prepare_state_series
from .model_selection import StateSelection, evaluate_models, validate_model_names
from .registry import write_training_artifacts


@dataclass
class TrainingRun:
    manifest: dict
    selections: list[StateSelection]


def train_forecasting_system(
    data_path: str | Path,
    config: ForecastConfig,
    states: list[str] | None = None,
    model_names: list[str] | None = None,
) -> TrainingRun:
    frame = load_sales_data(data_path, config)
    selected_states = states or list_states(frame, config)
    selected_models = validate_model_names(model_names)

    selections: list[StateSelection] = []
    forecasts = {}
    for state in selected_states:
        series = prepare_state_series(frame, state, config)
        selection = evaluate_models(series, state, config, selected_models)
        selections.append(selection)
        forecasts[state] = selection.best_model.predict(config.horizon)

    manifest = write_training_artifacts(selections, forecasts, config)
    return TrainingRun(manifest=manifest, selections=selections)
