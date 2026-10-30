from pathlib import Path
import os
from contextlib import asynccontextmanager
from typing import Literal
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from rain_runtime import RainPredictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("IA_API_KEY é obrigatória")
    base = Path(__file__).resolve().parent
    model_dir = Path(os.environ.get("MODEL_DIR", str(base / "artifacts")))
    required = ["model_bundle.joblib", "metadata.json"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Artefatos ausentes em {model_dir}: {', '.join(missing)}. "
            "Copie os arquivos exportados pelo notebook para artifacts/."
        )
    app.state.predictor = RainPredictor(model_dir)
    yield
    app.state.predictor = None

app = FastAPI(title="Previsão de chuva — Belém", version="1.0.0", lifespan=lifespan)
API_KEY = os.environ.get("IA_API_KEY", "")

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["rain-hourly-v1"]
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    feature_time: datetime
    horizon_hours: Literal[1]
    features: dict[str, StrictFloat | StrictInt | None]
    units: dict[str, str]

@app.get("/health")
def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="modelo não carregado")
    return {"status": "ok", "model_version": predictor.metadata["model_version"]}

@app.post("/v1/predict")
def predict(body: PredictionRequest, request: Request, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="não autorizado")
    try:
        return request.app.state.predictor.predict(body.model_dump(mode="json"))
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
