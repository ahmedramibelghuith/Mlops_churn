"""Churn prediction API with legacy model compatibility."""
from contextlib import asynccontextmanager

import logging
import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
if __package__ in (None, ""):
    from src.clean_data_csv import clean_data
    from src.features import FEATURES, select_features
    from src.tracking import configure_tracking, best_run
    from examplejson.churn_info import ChurnPredictionModel
else:
    from .src.clean_data_csv import clean_data
    from .src.features import FEATURES, select_features
    from .src.tracking import configure_tracking, best_run
    from .examplejson.churn_info import ChurnPredictionModel

logger = logging.getLogger(__name__)
model = None
model_uri = None


@asynccontextmanager
async def lifespan(app):
    global model, model_uri
    model = None
    model_uri = None
    try:
        configure_tracking()
        model_uri = os.getenv("MODEL_URI")
        if not model_uri:
            run = best_run()
            model_uri = f"runs:/{run['run_id']}/ML_models"
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception:
        logger.exception("Model loading failed; prediction endpoints are unavailable")
    yield
    model = None


app = FastAPI(title="Customer Churn Prediction API", version="2.1.0", lifespan=lifespan)
origins = [value.strip() for value in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials="*" not in origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return {"status": "ok", "model_loaded": True}


@app.get("/features")
def features():
    return {"features": FEATURES}


@app.get("/model-info")
def model_info():
    return {"model_uri": model_uri, "model_loaded": model is not None}


def predict_frame(frame):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    try:
        frame = select_features(frame)
        metadata = getattr(model.metadata, "metadata", None) or {}
        # New models contain their own processors. Existing classifiers keep
        # using the original bundled scaler and encoders.
        if metadata.get("input_format") != "raw-v1":
            frame = clean_data(frame)
        return {"predictions": model.predict(frame).tolist()}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        frame = pd.read_csv(file.file)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid CSV file") from exc
    return predict_frame(frame)


@app.post("/predict")
def predict(data: ChurnPredictionModel):
    return predict_frame(pd.DataFrame([data.model_dump(by_alias=True)]))
