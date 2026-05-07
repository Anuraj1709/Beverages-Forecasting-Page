# Forecasting Service Architecture

## Components

- `data.py`: schema validation, mixed Excel/text date parsing, state-level weekly resampling, missing date/value handling.
- `features.py`: lag features, rolling mean/std, calendar fields, and US holiday flags.
- `models/`: one adapter per algorithm: SARIMA, Prophet, XGBoost, LSTM.
- `model_selection.py`: time-series train/validation split and best-model selection by MAPE, RMSE, MAE.
- `pipeline.py`: end-to-end training orchestration by state.
- `registry.py`: artifact and manifest persistence.
- `api.py`: REST API for health checks, model metadata, and forecasts.

## Model Selection

Each state is trained independently. The last `validation_size` weekly periods are held out, every candidate model forecasts those periods, and the model with the lowest MAPE wins. RMSE and MAE are used as tie-breakers. The winning algorithm is then refit on full history and used to create the next 8 weekly forecasts.

## Missing Data Strategy

Historical data is aggregated by `State` and `Date`, converted to weekly `W-SUN` frequency, and reindexed. Missing dates become explicit weekly rows. Missing values are time-interpolated, then forward/back filled for edge cases.

## Feature Strategy

The mandatory `t-1`, `t-7`, and `t-30` lag features are implemented after weekly normalization, so they represent prior weekly forecast periods. Rolling statistics are shifted by one period to prevent leakage.

## API Contract

The API serves from durable artifacts instead of retraining on request. This keeps inference fast and predictable:

- Train offline with `python -m beverage_forecasting.cli train`.
- Start the API with `uvicorn beverage_forecasting.api:app`.
- Query forecasts through `GET /forecast/{state}`.
