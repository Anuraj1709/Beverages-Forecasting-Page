from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .config import DEFAULT_CONFIG
from .registry import load_latest_forecasts, load_manifest

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Beverage Forecasting Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9e1ec;
      --paper: #ffffff;
      --wash: #f4f7fb;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --good: #14845f;
      --warn: #b45309;
      --shadow: 0 16px 50px rgba(22, 32, 42, 0.09);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-width: 320px;
      background: var(--wash);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }

    a {
      color: inherit;
    }

    .app-shell {
      min-height: 100vh;
    }

    .topbar {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }

    .topbar-inner {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-mark {
      width: 40px;
      height: 40px;
      border-radius: 8px;
      background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
      display: grid;
      place-items: center;
      color: white;
      font-weight: 800;
      flex: 0 0 auto;
    }

    .brand h1 {
      margin: 0;
      font-size: clamp(1.1rem, 3vw, 1.35rem);
      letter-spacing: 0;
    }

    .brand p {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 0.9rem;
    }

    .docs-link {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 12px;
      background: #ffffff;
      text-decoration: none;
      font-weight: 700;
      font-size: 0.9rem;
      white-space: nowrap;
    }

    main {
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 40px;
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 180px 140px;
      gap: 12px;
      align-items: end;
      margin-bottom: 18px;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
    }

    select,
    input {
      width: 100%;
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      color: var(--ink);
      padding: 0 12px;
      font: inherit;
      font-weight: 650;
    }

    button {
      height: 42px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }

    button:hover {
      background: #0b665f;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .metric,
    .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .metric {
      padding: 16px;
      min-height: 112px;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
    }

    .metric strong {
      display: block;
      margin-top: 10px;
      font-size: clamp(1.35rem, 4vw, 1.9rem);
      line-height: 1.05;
    }

    .metric small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-weight: 650;
    }

    .content-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.85fr);
      gap: 18px;
      align-items: start;
    }

    .panel {
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }

    .panel-header h2 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0;
    }

    .panel-header span {
      color: var(--muted);
      font-size: 0.85rem;
      font-weight: 700;
    }

    .chart-wrap {
      padding: 18px;
    }

    svg {
      width: 100%;
      height: 330px;
      display: block;
      overflow: visible;
    }

    .axis {
      stroke: #c8d2df;
      stroke-width: 1;
    }

    .forecast-line {
      fill: none;
      stroke: var(--accent-2);
      stroke-width: 4;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .forecast-area {
      fill: rgba(37, 99, 235, 0.11);
    }

    .point {
      fill: #ffffff;
      stroke: var(--accent-2);
      stroke-width: 3;
    }

    .tick-label {
      fill: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .data-table {
      width: 100%;
      border-collapse: collapse;
    }

    th,
    td {
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 0.92rem;
    }

    th {
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0;
      background: #f8fafc;
    }

    td:last-child,
    th:last-child {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .model-stack {
      display: grid;
      gap: 12px;
      padding: 18px;
    }

    .model-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 12px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }

    .model-row strong {
      text-transform: capitalize;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 26px;
      border-radius: 999px;
      padding: 3px 9px;
      background: #e7f7f2;
      color: var(--good);
      font-size: 0.75rem;
      font-weight: 850;
    }

    .subtle {
      color: var(--muted);
      font-size: 0.85rem;
      font-weight: 650;
    }

    .state-list {
      max-height: 306px;
      overflow: auto;
    }

    .state-row {
      width: 100%;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: white;
      color: var(--ink);
      text-align: left;
      border-radius: 0;
    }

    .state-row:hover,
    .state-row.active {
      background: #eef7ff;
    }

    .state-row span {
      color: var(--muted);
      font-size: 0.85rem;
      font-weight: 750;
    }

    .empty {
      padding: 24px;
      color: var(--warn);
      font-weight: 700;
    }

    @media (max-width: 900px) {
      .toolbar,
      .content-grid,
      .summary-grid {
        grid-template-columns: 1fr;
      }

      .topbar-inner {
        align-items: flex-start;
        flex-direction: column;
        padding: 16px 0;
      }

      .docs-link {
        width: 100%;
        text-align: center;
      }
    }

    @media (max-width: 560px) {
      main,
      .topbar-inner {
        width: min(100% - 20px, 1180px);
      }

      th,
      td {
        padding: 10px 12px;
      }

      .panel-header {
        align-items: flex-start;
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <div class="brand-mark" aria-hidden="true">BF</div>
          <div>
            <h1>Beverage Forecasting</h1>
            <p>Weekly sales outlook by state</p>
          </div>
        </div>
        <a class="docs-link" href="/docs">API Docs</a>
      </div>
    </header>

    <main>
      <section class="toolbar" aria-label="Forecast controls">
        <label>
          State
          <select id="stateSelect"></select>
        </label>
        <label>
          Horizon
          <input id="horizonInput" type="number" min="1" max="52" value="8">
        </label>
        <button id="refreshButton" type="button">Refresh</button>
      </section>

      <section class="summary-grid" aria-label="Forecast summary">
        <article class="metric">
          <span>Selected State</span>
          <strong id="selectedState">Loading</strong>
          <small id="selectedModel">Model pending</small>
        </article>
        <article class="metric">
          <span>Next Week</span>
          <strong id="nextWeek">--</strong>
          <small id="nextWeekDate">--</small>
        </article>
        <article class="metric">
          <span>Forecast Total</span>
          <strong id="forecastTotal">--</strong>
          <small id="forecastWindow">--</small>
        </article>
        <article class="metric">
          <span>Validation MAPE</span>
          <strong id="stateMape">--</strong>
          <small>Lower is better</small>
        </article>
      </section>

      <section class="content-grid">
        <article class="panel">
          <div class="panel-header">
            <h2 id="chartTitle">Forecast Trend</h2>
            <span id="chartRange">--</span>
          </div>
          <div class="chart-wrap">
            <svg id="forecastChart" role="img" aria-labelledby="chartTitle"></svg>
          </div>
          <table class="data-table">
            <thead>
              <tr>
                <th>Week</th>
                <th>Prediction</th>
              </tr>
            </thead>
            <tbody id="forecastRows"></tbody>
          </table>
        </article>

        <aside>
          <article class="panel">
            <div class="panel-header">
              <h2>Model Metrics</h2>
              <span id="trainedAt">--</span>
            </div>
            <div class="model-stack" id="modelMetrics"></div>
          </article>

          <article class="panel" style="margin-top: 18px;">
            <div class="panel-header">
              <h2>States</h2>
              <span id="stateCount">--</span>
            </div>
            <div class="state-list" id="stateList"></div>
          </article>
        </aside>
      </section>
    </main>
  </div>

  <script>
    const stateSelect = document.querySelector("#stateSelect");
    const horizonInput = document.querySelector("#horizonInput");
    const refreshButton = document.querySelector("#refreshButton");
    const selectedState = document.querySelector("#selectedState");
    const selectedModel = document.querySelector("#selectedModel");
    const nextWeek = document.querySelector("#nextWeek");
    const nextWeekDate = document.querySelector("#nextWeekDate");
    const forecastTotal = document.querySelector("#forecastTotal");
    const forecastWindow = document.querySelector("#forecastWindow");
    const stateMape = document.querySelector("#stateMape");
    const chart = document.querySelector("#forecastChart");
    const chartRange = document.querySelector("#chartRange");
    const forecastRows = document.querySelector("#forecastRows");
    const modelMetrics = document.querySelector("#modelMetrics");
    const trainedAt = document.querySelector("#trainedAt");
    const stateCount = document.querySelector("#stateCount");
    const stateList = document.querySelector("#stateList");

    let models = {};
    let states = [];

    const money = new Intl.NumberFormat("en-US", {
      notation: "compact",
      maximumFractionDigits: 1
    });

    const fullMoney = new Intl.NumberFormat("en-US", {
      maximumFractionDigits: 0
    });

    function formatDate(value) {
      return new Date(value + "T00:00:00").toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric"
      });
    }

    function formatModelName(value) {
      return String(value || "--").toUpperCase();
    }

    function getHorizon() {
      const parsed = Number.parseInt(horizonInput.value, 10);
      return Math.min(52, Math.max(1, Number.isFinite(parsed) ? parsed : 8));
    }

    async function getJson(url) {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return response.json();
    }

    function drawChart(forecast) {
      chart.replaceChildren();
      if (!forecast.length) {
        chart.innerHTML = '<text x="20" y="40" class="tick-label">No forecast data available</text>';
        return;
      }

      const width = 760;
      const height = 330;
      const pad = { top: 20, right: 28, bottom: 48, left: 72 };
      const values = forecast.map((item) => item.prediction);
      const min = Math.min(...values) * 0.96;
      const max = Math.max(...values) * 1.04;
      const span = max - min || 1;

      chart.setAttribute("viewBox", `0 0 ${width} ${height}`);

      const x = (index) => {
        const usable = width - pad.left - pad.right;
        return pad.left + (forecast.length === 1 ? usable / 2 : (index / (forecast.length - 1)) * usable);
      };
      const y = (value) => pad.top + (1 - ((value - min) / span)) * (height - pad.top - pad.bottom);

      const points = forecast.map((item, index) => `${x(index)},${y(item.prediction)}`).join(" ");
      const area = `${pad.left},${height - pad.bottom} ${points} ${width - pad.right},${height - pad.bottom}`;
      chart.insertAdjacentHTML("beforeend", `<polyline class="forecast-area" points="${area}"></polyline>`);
      chart.insertAdjacentHTML("beforeend", `<line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>`);
      chart.insertAdjacentHTML("beforeend", `<line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"></line>`);
      chart.insertAdjacentHTML("beforeend", `<polyline class="forecast-line" points="${points}"></polyline>`);

      forecast.forEach((item, index) => {
        chart.insertAdjacentHTML("beforeend", `<circle class="point" cx="${x(index)}" cy="${y(item.prediction)}" r="5"></circle>`);
      });

      const labels = [
        { value: max, y: pad.top + 4 },
        { value: (max + min) / 2, y: y((max + min) / 2) },
        { value: min, y: height - pad.bottom }
      ];

      labels.forEach((label) => {
        chart.insertAdjacentHTML("beforeend", `<text class="tick-label" x="8" y="${label.y}" dominant-baseline="middle">${money.format(label.value)}</text>`);
      });

      forecast.forEach((item, index) => {
        if (index === 0 || index === forecast.length - 1) {
          chart.insertAdjacentHTML("beforeend", `<text class="tick-label" x="${x(index)}" y="${height - 16}" text-anchor="middle">${formatDate(item.date).replace(", 202", "")}</text>`);
        }
      });
    }

    function renderModels(state) {
      const model = models[state];
      if (!model) {
        modelMetrics.innerHTML = '<div class="empty">No model metadata found.</div>';
        return;
      }

      const candidates = model.candidates || [];
      modelMetrics.replaceChildren(...candidates.map((candidate) => {
        const row = document.createElement("div");
        row.className = "model-row";
        const metrics = candidate.metrics || {};
        row.innerHTML = `
          <strong>${candidate.name}</strong>
          ${candidate.name === model.best_model ? '<span class="badge">Best</span>' : '<span class="subtle">Trained</span>'}
          <span class="subtle">MAE ${money.format(metrics.mae || 0)}</span>
          <span class="subtle">MAPE ${Number(metrics.mape || 0).toFixed(1)}%</span>
        `;
        return row;
      }));
    }

    function renderStateList(activeState) {
      stateList.replaceChildren(...states.map((state) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `state-row${state === activeState ? " active" : ""}`;
        const model = models[state] || {};
        const mape = model.metrics ? `${Number(model.metrics.mape).toFixed(1)}%` : "--";
        button.innerHTML = `<strong>${state}</strong><span>${mape}</span>`;
        button.addEventListener("click", () => {
          stateSelect.value = state;
          loadForecast();
        });
        return button;
      }));
    }

    async function loadForecast() {
      const state = stateSelect.value || states[0];
      const horizon = getHorizon();
      horizonInput.value = horizon;

      const data = await getJson(`/forecast/${encodeURIComponent(state)}?horizon=${horizon}`);
      const forecast = data.forecast || [];
      const model = models[state] || {};
      const first = forecast[0];
      const last = forecast[forecast.length - 1];
      const total = forecast.reduce((sum, item) => sum + item.prediction, 0);

      selectedState.textContent = state;
      selectedModel.textContent = `${formatModelName(model.best_model)} selected model`;
      nextWeek.textContent = first ? money.format(first.prediction) : "--";
      nextWeekDate.textContent = first ? formatDate(first.date) : "--";
      forecastTotal.textContent = forecast.length ? money.format(total) : "--";
      forecastWindow.textContent = forecast.length && last ? `${forecast.length} weeks through ${formatDate(last.date)}` : "--";
      stateMape.textContent = model.metrics ? `${Number(model.metrics.mape).toFixed(1)}%` : "--";
      chartRange.textContent = first && last ? `${formatDate(first.date)} to ${formatDate(last.date)}` : "--";

      forecastRows.replaceChildren(...forecast.map((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${formatDate(item.date)}</td><td>${fullMoney.format(item.prediction)}</td>`;
        return row;
      }));

      drawChart(forecast);
      renderModels(state);
      renderStateList(state);
    }

    async function init() {
      try {
        const [statesData, modelsData] = await Promise.all([
          getJson("/states"),
          getJson("/models")
        ]);

        states = statesData.states || [];
        models = modelsData || {};
        stateCount.textContent = `${states.length} states`;
        trainedAt.textContent = "Latest artifacts";

        stateSelect.replaceChildren(...states.map((state) => {
          const option = document.createElement("option");
          option.value = state;
          option.textContent = state;
          return option;
        }));

        if (states.includes("California")) {
          stateSelect.value = "California";
        }

        await loadForecast();
      } catch (error) {
        document.querySelector("main").innerHTML = `<section class="panel"><div class="empty">Unable to load forecasts. ${error.message}</div></section>`;
      }
    }

    stateSelect.addEventListener("change", loadForecast);
    refreshButton.addEventListener("click", loadForecast);
    horizonInput.addEventListener("change", loadForecast);

    init();
  </script>
</body>
</html>
"""

app = FastAPI(
    title="Beverage Sales Forecasting API",
    version="0.1.0",
    description="Serves selected 8-week sales forecasts by state.",
)


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


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
