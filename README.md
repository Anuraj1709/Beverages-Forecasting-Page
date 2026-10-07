# Beverage Sales Forecasting Service

A production-style forecasting project that trains and serves weekly beverage sales predictions by state using FastAPI and a collection of time-series models.

## Overview

This project loads the [Forecasting Case- Study.xlsx](./Forecasting%20Case-%20Study.xlsx) dataset, cleans and engineers state-level weekly sales features, compares multiple forecasting models, and exposes the latest results through a REST API.

The system currently supports:

- SARIMA
- Prophet
- XGBoost
- LSTM

It selects the best-performing model per state using a time-series validation workflow and saves generated artifacts for later use in the API layer.

## Features

- Data loading and cleaning for state-level beverage sales
- Weekly resampling and feature engineering with calendar, lag, and rolling features
- Model comparison and best-model selection
- Artifact persistence for trained models and forecast outputs
- FastAPI-based API with Swagger docs
- Saved forecast results served through the API layer

## Project Structure

```text
.
├── app.py                     # FastAPI app entrypoint for deployment
├── src/
│   └── beverage_forecasting/
│       ├── api.py            # API routes and schema endpoints
│       ├── cli.py            # Training CLI
│       ├── config.py         # Default configuration
│       ├── data.py           # Data loading and preprocessing
│       ├── features.py       # Feature engineering
│       ├── pipeline.py       # Training pipeline
│       ├── registry.py       # Artifact loading and lookup
│       ├── models/           # Forecasting model implementations
│       └── ...
├── artifacts/                 # Training output and forecast JSON files
├── Forecasting Case- Study.xlsx
├── requirements.txt
├── requirements-full.txt
├── pyproject.toml
├── Dockerfile
├── README.md
└── docs_architecture.md
```

## Dataset

The project expects the sales workbook at the project root:

```text
Forecasting Case- Study.xlsx
```

Required columns include:

- `State`
- `Date`
- `Total`
- `Category`

The loader accepts both Excel serial-date values and date strings such as `13-02-2022`.

## Local Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-full.txt
python -m pip install -e .
```

On Windows, if the system Python complains about missing `Scripts/*.exe` entries, install packages to user site-packages instead:

```bash
python -m pip install --user --no-warn-script-location -r requirements-full.txt
python -m pip install --user --no-warn-script-location -e .
```

Note: `tensorflow` and `prophet` can be heavy. If needed, install the base stack first, then add the heavier packages before running full training.

## Train Models

Run the training pipeline with the default dataset and forecast horizon:

```bash
python -m beverage_forecasting.cli train --data "Forecasting Case- Study.xlsx" --horizon 8
```

For a quicker validation run on a subset of states and models:

```bash
python -m beverage_forecasting.cli train --data "Forecasting Case- Study.xlsx" --states Alabama Arizona --models sarima xgboost --horizon 8
```

Artifacts are written to the `artifacts/` directory, including:

- `manifest.json`: selected model and metrics per state
- `models/<state>/<model>.joblib` or `.keras`: trained model files
- `forecasts/latest_forecasts.json`: latest forecast output

## Run the FastAPI App

Start the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or, if running from the package directly:

```bash
uvicorn beverage_forecasting.api:app --host 0.0.0.0 --port 8000
```

Then open the docs page:

```text
http://localhost:8000/docs
```

The app also redirects `/` to `/docs`.

## API Endpoints

The API exposes the following endpoints:

- `GET /health` — checks whether artifacts are available
- `GET /states` — returns the available states
- `GET /models` — returns model metadata from the manifest
- `GET /forecast/{state}?horizon=8` — returns a forecast for one state
- `GET /forecast?horizon=8` — returns forecasts for all states

Example requests:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/states
curl "http://localhost:8000/forecast/Alabama?horizon=8"
```

## Deployment

Deploy the FastAPI app using the platform of your choice. The public app should serve saved forecast artifacts instead of retraining models at request time.

### Recommended flow

1. Train locally.
2. Commit the updated `artifacts/` JSON output.
3. Deploy the project with `app.py` as the FastAPI entrypoint.
4. Access the dashboard at `/` or the API docs at `/docs`.

## Time-Series Workflow

The forecasting pipeline:

- sorts data by date per state
- resamples to weekly frequency
- interpolates and fills missing sales values
- creates lag and rolling features
- trains several candidate models
- selects the best model for each state using holdout validation

This ensures the model is evaluated against a realistic validation window without leaking future information.

## License

This project does not currently define a specific license file. If you intend to publish or redistribute it, add an appropriate open-source license before doing so.
