from __future__ import annotations

from time import perf_counter
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from fraud_detection.api.schemas import (
    BatchPredictionRequest,
    DriftResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResponse,
    TransactionRequest,
)
from fraud_detection.api.service import (
    ModelLoadError,
    health_status,
    latest_drift_report,
    model_metadata,
    predict_records,
)

REQUEST_COUNTER = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API request latency",
    ["method", "path"],
)


app = FastAPI(title="Fraud Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    start = perf_counter()
    response = await call_next(request)
    elapsed = perf_counter() - start
    REQUEST_COUNTER.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(elapsed)
    return response


@app.get("/api/v1/health", response_model=HealthResponse)
def get_health() -> dict:
    return health_status()


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> dict:
    try:
        return predict_records([request.model_dump()])[0]
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/predict/batch", response_model=list[PredictionResponse])
def predict_batch(requests: BatchPredictionRequest) -> list[dict]:
    try:
        return predict_records([request.model_dump() for request in requests])
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/model", response_model=ModelMetadataResponse)
def get_model_metadata() -> dict:
    try:
        return model_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/drift", response_model=DriftResponse)
def get_drift() -> dict:
    return latest_drift_report()


@app.get("/api/v1/metrics")
def get_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
