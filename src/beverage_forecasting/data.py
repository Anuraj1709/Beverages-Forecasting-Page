from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd

from .config import ForecastConfig
from .exceptions import NotEnoughHistoryError


def load_sales_data(path: str | Path, config: ForecastConfig) -> pd.DataFrame:
    """Load and validate the sales workbook."""
    path = Path(path)
    try:
        frame = pd.read_excel(path)
    except ImportError:
        frame = _read_xlsx_without_openpyxl(path)

    missing = {
        config.state_col,
        config.date_col,
        config.target_col,
    } - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame[config.date_col] = frame[config.date_col].map(_parse_mixed_date)
    frame[config.target_col] = pd.to_numeric(frame[config.target_col], errors="coerce")
    frame[config.state_col] = frame[config.state_col].astype(str).str.strip()
    if config.category_col in frame.columns:
        frame[config.category_col] = frame[config.category_col].astype(str).str.strip()

    frame = frame.dropna(subset=[config.state_col, config.date_col])
    frame = frame.sort_values([config.state_col, config.date_col]).reset_index(drop=True)
    return frame


def prepare_state_series(frame: pd.DataFrame, state: str, config: ForecastConfig) -> pd.Series:
    """Return a clean weekly series for one state."""
    state_frame = frame.loc[frame[config.state_col] == state, [config.date_col, config.target_col]]
    if state_frame.empty:
        raise ValueError(f"Unknown state: {state}")

    series = (
        state_frame.groupby(config.date_col, as_index=True)[config.target_col]
        .sum()
        .sort_index()
        .resample(config.frequency)
        .sum(min_count=1)
    )
    series = series.interpolate(method="time").ffill().bfill()

    if len(series) < config.min_train_points:
        raise NotEnoughHistoryError(
            f"{state} has {len(series)} weekly rows; need at least {config.min_train_points}."
        )
    series.name = state
    return series


def list_states(frame: pd.DataFrame, config: ForecastConfig) -> list[str]:
    return sorted(frame[config.state_col].dropna().astype(str).unique().tolist())


def _parse_mixed_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, float)):
        return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=float(value))).normalize()

    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=float(text))).normalize()
    for day_first in (True, False):
        parsed = pd.to_datetime(text, dayfirst=day_first, errors="coerce")
        if not pd.isna(parsed):
            return parsed.normalize()
    return pd.NaT


def _read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    """Minimal XLSX reader for this assignment workbook when openpyxl is absent."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings = _shared_strings(archive, ns)
        sheet_xml = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[dict[str, object]] = []
        for row in sheet_xml.findall(".//a:sheetData/a:row", ns):
            parsed: dict[str, object] = {}
            for cell in row.findall("a:c", ns):
                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value != "":
                    value = shared_strings[int(value)]
                parsed[_cell_column(cell.attrib["r"])] = value
            rows.append(parsed)

    headers = {column: value for column, value in rows[0].items()}
    records = []
    for row in rows[1:]:
        records.append({headers[column]: value for column, value in row.items() if column in headers})
    return pd.DataFrame(records)


def _shared_strings(archive: ZipFile, ns: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", ns):
        strings.append("".join(node.text or "" for node in item.findall(".//a:t", ns)))
    return strings


def _cell_column(reference: str) -> str:
    return re.match(r"[A-Z]+", reference).group(0)
