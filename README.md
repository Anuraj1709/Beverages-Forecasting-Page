# Beverage Sales Forecasting Service

Production-style time series forecasting system for weekly beverage sales by state.

## What It Does

- Loads the Excel case-study dataset.
- Cleans missing dates and missing sales values per state.
- Builds calendar, holiday, lag, and rolling features.
- Trains and compares SARIMA, Prophet, XGBoost, and LSTM models.
- Selects the best model per state using time-series validation.
- Saves artifacts and serves 8-week forecasts through a REST API.

## Dataset

The workbook is expected at:

```text
Forecasting Case- Study.xlsx
```

Required columns:

- `State`
- `Date`
- `Total`
- `Category`

The loader accepts both Excel serial dates and text dates such as `13-02-2022`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-full.txt
python -m pip install -e .
```

`tensorflow` and `prophet` can be heavy on Windows. If installation is slow, install the core stack first, then add those packages in the same environment before running full training.

If you use the system Python on Windows and see a `Scripts/*.exe` install error, install to your user site-packages:

```bash
python -m pip install --user --no-warn-script-location -r requirements-full.txt
python -m pip install --user --no-warn-script-location -e .
```

## Train Models

```bash
python -m beverage_forecasting.cli train --data "Forecasting Case- Study.xlsx" --horizon 8
```

Useful faster smoke run:

```bash
python -m beverage_forecasting.cli train --data "Forecasting Case- Study.xlsx" --states Alabama Arizona --models sarima xgboost --horizon 8
```

Artifacts are written to `artifacts/`:

- `manifest.json`: selected model and metrics per state
- `models/<state>/<model>.joblib|keras`: trained model artifacts
- `forecasts/latest_forecasts.json`: latest 8-week predictions

## Run API

```bash
uvicorn beverage_forecasting.api:app --host 0.0.0.0 --port 8000
```

Example requests:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/states
curl "http://localhost:8000/forecast/Alabama?horizon=8"
```

## Deploy On Vercel

The Vercel deployment serves saved forecast results from `artifacts/manifest.json` and `artifacts/forecasts/latest_forecasts.json`. It does not train models on Vercel because TensorFlow, Prophet, XGBoost, and Statsmodels are too heavy for a serverless API bundle.

1. Train locally first:

```bash
python -m beverage_forecasting.cli train --data "Forecasting Case- Study.xlsx" --horizon 8
```

2. Commit and push the updated forecast JSON files.

3. Import the GitHub repository into Vercel.

4. Keep the default settings. Vercel will detect `app.py` as the FastAPI entrypoint and install the lightweight `requirements.txt`.

After deploy, open:

```text
https://your-vercel-app.vercel.app/docs
https://your-vercel-app.vercel.app/forecast/Alabama?horizon=8
```

## API Endpoints

- `GET /health`
- `GET /states`
- `GET /models`
- `GET /forecast/{state}?horizon=8`
- `GET /forecast?horizon=8`

## Time-Series Logic

For each state, rows are sorted by date, resampled to weekly frequency, and missing sales values are interpolated then forward/back filled. Validation uses the last `horizon` periods only, so future values are not leaked into training features. The requested `t-1`, `t-7`, and `t-30` lag features are implemented as period lags after weekly resampling.
