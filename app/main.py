import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    HealthResponse,
    PredictionResponse,
)
from src.config import MODEL_PATH
from src.predict import load_package, predict_dataframe


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("CHURN_MODEL_PATH", str(MODEL_PATH)))
    app.state.package = load_package(model_path)
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


def to_response(row: pd.Series, threshold: float) -> PredictionResponse:
    customer_id = row["customerID"] if "customerID" in row.index else None
    if customer_id is not None and pd.isna(customer_id):
        customer_id = None
    return PredictionResponse(
        customerID=None if customer_id is None else str(customer_id),
        churn_probability=float(row["churn_probability"]),
        predicted_churn=int(row["predicted_churn"]),
        threshold=float(threshold),
    )


def run_prediction(customers, package):
    frame = pd.DataFrame([customer.model_dump() for customer in customers])
    try:
        return predict_dataframe(frame, package)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    package = request.app.state.package
    return HealthResponse(
        status="ok",
        model_loaded=package is not None,
        threshold=float(package["threshold"]),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, request: Request):
    package = request.app.state.package
    result = run_prediction([customer], package)
    return to_response(result.iloc[0], package["threshold"])


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, request: Request):
    package = request.app.state.package
    result = run_prediction(payload.customers, package)
    predictions = [to_response(row, package["threshold"]) for _, row in result.iterrows()]
    return BatchPredictionResponse(predictions=predictions)
