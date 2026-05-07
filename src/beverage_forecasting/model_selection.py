from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import ForecastConfig
from .exceptions import ForecastingError, ModelUnavailableError
from .metrics import regression_metrics
from .models import LSTMForecastModel, ProphetForecastModel, SarimaModel, XGBoostLagModel
from .models.base import ForecastModel


MODEL_FACTORIES = {
    "sarima": SarimaModel,
    "prophet": ProphetForecastModel,
    "xgboost": XGBoostLagModel,
    "lstm": LSTMForecastModel,
}


@dataclass
class CandidateResult:
    name: str
    status: str
    metrics: dict[str, float] | None = None
    error: str | None = None


@dataclass
class StateSelection:
    state: str
    best_model_name: str
    best_model: ForecastModel
    validation_predictions: pd.Series
    metrics: dict[str, float]
    candidates: list[CandidateResult] = field(default_factory=list)


def evaluate_models(
    series: pd.Series,
    state: str,
    config: ForecastConfig,
    model_names: list[str],
) -> StateSelection:
    horizon = min(config.validation_size, max(1, len(series) // 5))
    train = series.iloc[:-horizon]
    validation = series.iloc[-horizon:]

    candidates: list[CandidateResult] = []
    fitted_candidates: list[tuple[ForecastModel, pd.Series, dict[str, float]]] = []

    for model_name in model_names:
        factory = MODEL_FACTORIES[model_name]
        try:
            model = factory(config).fit(train, state)
            predictions = model.predict(horizon)
            predictions.index = validation.index
            metrics = regression_metrics(validation, predictions)
            fitted_candidates.append((model, predictions, metrics))
            candidates.append(CandidateResult(model_name, "trained", metrics=metrics))
        except ModelUnavailableError as exc:
            candidates.append(CandidateResult(model_name, "skipped", error=str(exc)))
        except Exception as exc:
            candidates.append(CandidateResult(model_name, "failed", error=str(exc)))

    if not fitted_candidates:
        errors = "; ".join(f"{candidate.name}: {candidate.error}" for candidate in candidates)
        raise ForecastingError(f"No model could be trained for {state}. {errors}")

    best_validation_model, best_predictions, best_metrics = min(
        fitted_candidates,
        key=lambda item: (item[2]["mape"], item[2]["rmse"], item[2]["mae"]),
    )
    best_name = best_validation_model.name
    best_full_model = MODEL_FACTORIES[best_name](config).fit(series, state)

    return StateSelection(
        state=state,
        best_model_name=best_name,
        best_model=best_full_model,
        validation_predictions=best_predictions,
        metrics=best_metrics,
        candidates=candidates,
    )


def validate_model_names(model_names: list[str] | None) -> list[str]:
    selected = model_names or list(MODEL_FACTORIES)
    unknown = set(selected) - set(MODEL_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return selected
