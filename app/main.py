from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib
import pandas as pd
import numpy as np
import os
from typing import List

from app.models import LoanApplication, PredictionResponse, HealthCheckResponse
from app.utils import preprocess_input

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default risk using XGBoost model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgboost_loan_default_model_fixed.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "standard_scaler.pkl")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        raise RuntimeError("Failed to load model. Please check if model files exist and are valid.")

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(loan_application: LoanApplication):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([loan_application.dict()])
        
        # Preprocess the input
        processed_input = preprocess_input(input_df, scaler)
        
        # Make prediction
        default_probability = float(model.predict_proba(processed_input)[:, 1][0])
        default_prediction = bool(model.predict(processed_input)[0])
        
        # Determine risk category
        risk_category = "Low"
        if default_probability >= 0.75:
            risk_category = "Very High"
        elif default_probability >= 0.5:
            risk_category = "High"
        elif default_probability >= 0.25:
            risk_category = "Medium"
            
        return {
            "default_probability": default_probability,
            "default_prediction": default_prediction,
            "risk_category": risk_category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(loan_applications: List[LoanApplication]):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([loan.dict() for loan in loan_applications])
        
        # Preprocess the input
        processed_input = preprocess_input(input_df, scaler)
        
        # Make predictions
        default_probabilities = model.predict_proba(processed_input)[:, 1]
        default_predictions = model.predict(processed_input)
        
        # Prepare response
        results = []
        for i in range(len(loan_applications)):
            prob = float(default_probabilities[i])
            prediction = bool(default_predictions[i])
            
            # Determine risk category
            risk_category = "Low"
            if prob >= 0.75:
                risk_category = "Very High"
            elif prob >= 0.5:
                risk_category = "High"
            elif prob >= 0.25:
                risk_category = "Medium"
                
            results.append({
                "default_probability": prob,
                "default_prediction": prediction,
                "risk_category": risk_category
            })
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)