
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
app = FastAPI()

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Input schema
class CustomerData(BaseModel):
    account_length: int
    international_plan: int
    voice_mail_plan: int
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    number_customer_service_calls: int


@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}


@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert input to dictionary
        input_dict = data.dict()

        # Convert to DataFrame
        df = pd.DataFrame([input_dict])

        # Load training columns
        cols = joblib.load("columns.pkl")

        # Apply same encoding
        df = pd.get_dummies(df)

        # Add missing columns
        for col in cols:
            if col not in df.columns:
                df[col] = 0

        # Ensure same order
        df = df[cols]

        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }

    except Exception as e:
        return {"error": str(e)}
