from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=1)
    type: str
    amount: float = Field(..., ge=0)
    nameOrig: str
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    nameDest: str
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_prediction: int
    threshold: float
    model_version: str
    selected_model: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_path: Optional[str] = None


class ModelMetadataResponse(BaseModel):
    version: str
    selected_model: str
    threshold: float
    trained_at: Optional[str] = None
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)


class DriftResponse(BaseModel):
    summary: dict[str, Any]
    numeric: dict[str, Any]
    categorical: dict[str, Any]


BatchPredictionRequest = list[TransactionRequest]
