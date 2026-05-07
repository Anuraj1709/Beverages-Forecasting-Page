FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -e .

COPY "Forecasting Case- Study.xlsx" ./

EXPOSE 8000

CMD ["uvicorn", "beverage_forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]
