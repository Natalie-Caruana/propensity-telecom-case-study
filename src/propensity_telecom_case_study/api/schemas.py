"""Pydantic schemas for the inference API request and response."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input features for a single customer record."""

    # Numeric features
    age: float
    tenure_months: float
    monthly_charges: float
    data_usage_gb: float
    call_minutes: float
    num_products: int
    num_complaints: int
    customer_service_calls: int

    # Categorical features
    region: str
    contract_type: str
    internet_service: str

    # Binary features (0 or 1)
    has_streaming: int = Field(..., ge=0, le=1)
    has_device_protection: int = Field(..., ge=0, le=1)


class PredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1)


class CustomerPrediction(BaseModel):
    propensity_score: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predictions: list[CustomerPrediction]
    model_name: str
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
