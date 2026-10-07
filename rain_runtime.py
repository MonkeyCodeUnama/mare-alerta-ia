from pathlib import Path
import json
import math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import joblib
import sklearn

SCHEMA_VERSION = "rain-hourly-v1"
VARIABLES = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
    "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "cape",
    "total_column_integrated_water_vapour", "relative_humidity_850hPa", "relative_humidity_700hPa",
    "relative_humidity_500hPa", "temperature_850hPa", "temperature_700hPa",
    "geopotential_height_500hPa", "rain", "precipitation"]
EXPECTED_UNITS = {}
for name in VARIABLES:
    if "humidity" in name or "cloud_cover" in name:
        unit = "%"
    elif "temperature" in name or name == "dew_point_2m":
        unit = "°C"
    elif "pressure" in name:
        unit = "hPa"
    elif name in ("wind_speed_10m", "wind_gusts_10m"):
        unit = "km/h"
    elif name == "wind_direction_10m":
        unit = "°"
    elif name == "cape":
        unit = "J/kg"
    elif name == "total_column_integrated_water_vapour":
        unit = "kg/m²"
    elif name == "geopotential_height_500hPa":
        unit = "m"
    else:
        unit = "mm"
    EXPECTED_UNITS[name] = unit


def numeric_or_null(value, name):
    if value is None:
        return np.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: esperado número finito ou null")
    return float(value)


def clean_frame(frame):
    frame = frame.copy()
    invalid = {}
    nonnegative = {"precipitation", "rain", "cape", "wind_speed_10m", "wind_gusts_10m",
                   "total_column_integrated_water_vapour"}
    for name in frame.columns:
        bad = pd.Series(False, index=frame.index)
        if "humidity" in name or "cloud_cover" in name:
            bad = (frame[name] < 0) | (frame[name] > 100)
        elif name == "wind_direction_10m":
            bad = (frame[name] < 0) | (frame[name] > 360)
        elif name in nonnegative:
            bad = frame[name] < 0
        elif name in ("pressure_msl", "surface_pressure"):
            bad = frame[name] <= 0
        invalid[name] = int(bad.sum())
        frame.loc[bad, name] = np.nan
    return frame, invalid


def validate_location(latitude, longitude):
    latitude = numeric_or_null(latitude, "latitude")
    longitude = numeric_or_null(longitude, "longitude")
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ValueError("Coordenadas obrigatórias")
    if abs(latitude - (-1.4558)) > 0.01 or abs(longitude - (-48.5044)) > 0.01:
        raise ValueError("Modelo restrito ao ponto de referência de Belém")


def load_dataset(path):
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict) or not isinstance(content.get("params"), dict) or not isinstance(content.get("response"), dict):
        raise ValueError("Esperado envelope com params e response")
    params, payload = content["params"], content["response"]
    if payload.get("error"):
        raise ValueError(payload.get("reason", "Erro do provedor"))
    if params.get("models") != "gfs_global":
        raise ValueError("Fonte incompatível: esperado gfs_global")
    if params.get("timezone") not in ("UTC", "GMT") or payload.get("utc_offset_seconds") != 0:
        raise ValueError("Dados precisam estar em UTC/GMT")
    for key, expected in {"wind_speed_unit": "kmh", "precipitation_unit": "mm", "temperature_unit": "celsius"}.items():
        if params.get(key) != expected:
            raise ValueError(f"Parâmetro incompatível: {key}")
    validate_location(params.get("latitude"), params.get("longitude"))
    for key in ("latitude", "longitude"):
        returned = numeric_or_null(payload.get(key), key)
        if not np.isfinite(returned) or abs(returned - params[key]) > 0.25:
            raise ValueError(f"Grade retornada incompatível: {key}")
    hourly, units = payload.get("hourly"), payload.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise ValueError("hourly e hourly_units obrigatórios")
    times = hourly.get("time")
    if not isinstance(times, list) or len(times) < 2:
        raise ValueError("Série horária insuficiente")
    columns = {}
    for name in VARIABLES:
        if units.get(name) != EXPECTED_UNITS[name]:
            raise ValueError(f"Unidade incompatível para {name}: {units.get(name)!r}; esperado {EXPECTED_UNITS[name]!r}")
        values = hourly.get(name)
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError(f"Array ausente ou desalinhado: {name}")
        columns[name] = [numeric_or_null(v, name) for v in values]
    if any(not isinstance(t, str) for t in times):
        raise ValueError("Timestamps precisam ser strings ISO-8601")
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True, errors="raise"))
    if index.hasnans or index.duplicated().any() or not index.floor("h").equals(index):
        raise ValueError("Timestamps ausentes, duplicados ou fora da grade horária")
    start = pd.Timestamp(params["start_date"], tz="UTC")
    end = pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(hours=23)
    if end < start or (end - start).days > 36525:
        raise ValueError("Intervalo inválido ou excessivo")
    grid = pd.date_range(start, end, freq="h")
    if len(index.difference(grid)):
        raise ValueError("Timestamps fora do intervalo solicitado")
    raw = pd.DataFrame(columns, index=index).sort_index()
    raw.index.name = "time"
    frame, invalid = clean_frame(raw.reindex(grid))
    frame.index.name = "time"
    report = {"raw_rows": len(raw), "grid_rows": len(grid), "missing_timestamps": len(grid.difference(index)),
              "invalid_values": invalid, "missing_counts": frame.isna().sum().astype(int).to_dict(),
              "missing_fraction": frame.isna().mean().to_dict(), "params": params,
              "returned_location": {k: payload[k] for k in ("latitude", "longitude")}, "units": units}
    return frame, report


def make_features(frame, config):
    if frame.index.tz is None:
        raise ValueError("Índice sem timezone")
    raw_names = [v for v in VARIABLES if v not in config["excluded_features"]]
    result = frame[raw_names].copy()
    direction = np.deg2rad(result.pop("wind_direction_10m"))
    result["wind_direction_sin"] = np.sin(direction)
    result["wind_direction_cos"] = np.cos(direction)
    local = result.index.tz_convert(config["timezone"])
    result["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    year_length = np.where(local.is_leap_year, 366, 365)
    year_phase = ((local.dayofyear - 1) + local.hour / 24) / year_length
    result["year_sin"] = np.sin(2 * np.pi * year_phase)
    result["year_cos"] = np.cos(2 * np.pi * year_phase)
    return result


class RainPredictor:
    """Usa o mesmo código de limpeza/features do notebook; não consulta API externa."""
    def __init__(self, artifact_dir):
        root = Path(artifact_dir)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata["versions"]["scikit-learn"] != sklearn.__version__:
            raise RuntimeError("Instale a versão de scikit-learn indicada em requirements.txt")
        self.bundle = joblib.load(root / "model_bundle.joblib")
        if self.bundle["model_version"] != metadata["model_version"] or self.bundle["schema_version"] != SCHEMA_VERSION:
            raise RuntimeError("Artefato e metadados incompatíveis")
        self.metadata = metadata

    def predict(self, request):
        config = self.bundle["config"]
        if request.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("schema_version incompatível")
        if request.get("horizon_hours") != config["horizon_hours"]:
            raise ValueError("Horizonte incompatível")
        validate_location(request.get("latitude"), request.get("longitude"))
        time = pd.Timestamp(request["feature_time"])
        if time.tzinfo is None or time != time.floor("h"):
            raise ValueError("feature_time precisa indicar timezone e uma hora exata")
        time = time.tz_convert("UTC")
        values, units = request.get("features"), request.get("units")
        if not isinstance(values, dict) or not isinstance(units, dict):
            raise ValueError("features e units obrigatórios")
        names = self.bundle["required_raw_features"]
        row = {}
        for name in names:
            if name not in values:
                raise ValueError(f"Feature obrigatória ausente: {name}; use null para ausência conhecida")
            if units.get(name) != EXPECTED_UNITS[name]:
                raise ValueError(f"Unidade incompatível: {name}")
            row[name] = numeric_or_null(values[name], name)
        frame, invalid = clean_frame(pd.DataFrame([row], index=pd.DatetimeIndex([time])))
        # As colunas excluídas não entram nas features nem precisam estar na requisição.
        missing = float(frame[names].isna().mean(axis=1).iloc[0])
        if missing > config["max_row_missing_fraction"]:
            raise ValueError("Dados insuficientes para inferência")
        features = make_features(frame, config)[self.bundle["feature_names"]]
        transformed = self.bundle["preprocessor"].transform(features)
        probability = float(self.bundle["model"].predict_proba(transformed)[0, 1])
        return {"location": "belem-pa", "issued_at": datetime.now(timezone.utc).isoformat(),
                "feature_time": time.isoformat(), "window_start": time.isoformat(),
                "window_end": (time + pd.Timedelta(hours=config["horizon_hours"])).isoformat(),
                "horizon_hours": config["horizon_hours"], "rain_probability": probability,
                "rain_expected": probability >= self.bundle["decision_threshold"],
                "rain_event_threshold_mm": config["rain_threshold_mm"],
                "decision_threshold": self.bundle["decision_threshold"],
                "model_version": self.bundle["model_version"], "data_version": self.bundle["data_version"],
                "target_source": "gfs_modelled_precipitation",
                "quality": {"missing_feature_fraction": missing,
                            "invalid_fields": [k for k, v in invalid.items() if v]}}
