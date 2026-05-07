from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from .config import DEFAULT_CONFIG
from .registry import load_latest_forecasts, load_manifest

app = FastAPI(
    title="Beverage Sales Forecasting API",
    version="0.1.0",
    description="Serves selected 8-week sales forecasts by state.",
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_manifest(DEFAULT_CONFIG)
        return {"status": "ok"}
    except FileNotFoundError:
        return {"status": "no_artifacts"}


@app.get("/states")
def states() -> dict[str, list[str]]:
    manifest = _manifest_or_404()
    return {"states": sorted(manifest["states"].keys())}


@app.get("/models")
def models() -> dict:
    return _manifest_or_404()["states"]


@app.get("/forecast/{state}")
def forecast_state(state: str, horizon: int = Query(default=8, ge=1, le=52)) -> dict:
    forecasts = _forecasts_or_404()
    matched = next((name for name in forecasts if name.lower() == state.lower()), None)
    if matched is None:
        raise HTTPException(status_code=404, detail=f"No forecast found for state: {state}")
    return {"state": matched, "horizon": horizon, "forecast": forecasts[matched][:horizon]}


@app.get("/forecast")
def forecast_all(horizon: int = Query(default=8, ge=1, le=52)) -> dict:
    forecasts = _forecasts_or_404()
    return {
        "horizon": horizon,
        "forecasts": {state: values[:horizon] for state, values in forecasts.items()},
    }


def _manifest_or_404() -> dict:
    try:
        return load_manifest(DEFAULT_CONFIG)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Train models before starting the API.") from exc


def _forecasts_or_404() -> dict:
    try:
        return load_latest_forecasts(DEFAULT_CONFIG)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Train models before requesting forecasts.") from exc
