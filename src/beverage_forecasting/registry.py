from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ForecastConfig

if TYPE_CHECKING:
    from .model_selection import StateSelection


def write_training_artifacts(
    selections: list[StateSelection],
    forecasts: dict[str, pd.Series],
    config: ForecastConfig,
) -> dict[str, Any]:
    artifacts_dir = Path(config.artifacts_dir)
    models_dir = artifacts_dir / "models"
    forecasts_dir = artifacts_dir / "forecasts"
    models_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "frequency": config.frequency,
        "horizon": config.horizon,
        "states": {},
    }

    for selection in selections:
        model_path = _save_model_best_effort(selection, models_dir)
        manifest["states"][selection.state] = {
            "best_model": selection.best_model_name,
            "metrics": selection.metrics,
            "model_artifact": str(model_path) if model_path else None,
            "candidates": [
                {
                    "name": candidate.name,
                    "status": candidate.status,
                    "metrics": candidate.metrics,
                    "error": candidate.error,
                }
                for candidate in selection.candidates
            ],
        }

    latest_forecasts = {
        state: [
            {"date": date.date().isoformat(), "prediction": float(value)}
            for date, value in forecast.items()
        ]
        for state, forecast in forecasts.items()
    }

    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (forecasts_dir / "latest_forecasts.json").write_text(json.dumps(latest_forecasts, indent=2), encoding="utf-8")
    return manifest


def load_manifest(config: ForecastConfig) -> dict[str, Any]:
    return json.loads((Path(config.artifacts_dir) / "manifest.json").read_text(encoding="utf-8"))


def load_latest_forecasts(config: ForecastConfig) -> dict[str, Any]:
    return json.loads((Path(config.artifacts_dir) / "forecasts" / "latest_forecasts.json").read_text(encoding="utf-8"))


def _save_model_best_effort(selection: StateSelection, models_dir: Path) -> Path | None:
    state_dir = models_dir / _slug(selection.state)
    state_dir.mkdir(parents=True, exist_ok=True)

    keras_model = getattr(selection.best_model, "_model", None)
    if selection.best_model_name == "lstm" and keras_model is not None:
        model_path = state_dir / "lstm.keras"
        keras_model.save(model_path)
        return model_path

    try:
        import joblib

        model_path = state_dir / f"{selection.best_model_name}.joblib"
        joblib.dump(selection.best_model, model_path)
        return model_path
    except Exception:
        return None


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")
