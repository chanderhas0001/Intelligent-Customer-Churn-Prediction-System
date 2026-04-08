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

# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import pandas as pd
# import joblib

# app = FastAPI()

# # Load files
# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")
# cols = joblib.load("columns.pkl")

# # Templates
# templates = Jinja2Templates(directory="templates")

# # Input schema
# class CustomerData(BaseModel):
#     account_length: int
#     international_plan: int
#     voice_mail_plan: int
#     number_vmail_messages: int
#     total_day_minutes: float
#     total_day_calls: int
#     total_day_charge: float
#     total_eve_minutes: float
#     total_eve_calls: int
#     total_eve_charge: float
#     total_night_minutes: float
#     total_night_calls: int
#     total_night_charge: float
#     total_intl_minutes: float
#     total_intl_calls: int
#     total_intl_charge: float
#     number_customer_service_calls: int

# # Home route → HTML page
# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # Prediction route
# @app.post("/predict")
# def predict(data: CustomerData):
#     try:
#         df = pd.DataFrame([data.dict()])

#         # Encoding
#         df = pd.get_dummies(df)

#         # Match columns
#         for col in cols:
#             if col not in df.columns:
#                 df[col] = 0

#         df = df[cols]

#         # Scaling
#         df = scaler.transform(df)

#         # Prediction
#         prediction = model.predict(df)[0]
#         probability = model.predict_proba(df)[0][1]

#         return {
#             "prediction": int(prediction),
#             "probability": float(probability)
#         }

#     except Exception as e:
#         return {"error": str(e)}






# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import pandas as pd
# import joblib
# import os

# app = FastAPI()

# # -------------------------------
# # SAFE LOADING (prevents crash)
# # -------------------------------
# model = None
# scaler = None
# cols = None

# try:
#     if os.path.exists("model.pkl"):
#         model = joblib.load("model.pkl")
#     else:
#         print("model.pkl not found")

#     if os.path.exists("scaler.pkl"):
#         scaler = joblib.load("scaler.pkl")
#     else:
#         print("scaler.pkl not found")

#     if os.path.exists("columns.pkl"):
#         cols = joblib.load("columns.pkl")
#     else:
#         print("columns.pkl not found")

# except Exception as e:
#     print("Error loading files:", e)

# # -------------------------------
# # Templates
# # -------------------------------
# templates = Jinja2Templates(directory="templates")

# # -------------------------------
# # Input schema
# # -------------------------------
# class CustomerData(BaseModel):
#     account_length: int
#     international_plan: int
#     voice_mail_plan: int
#     number_vmail_messages: int
#     total_day_minutes: float
#     total_day_calls: int
#     total_day_charge: float
#     total_eve_minutes: float
#     total_eve_calls: int
#     total_eve_charge: float
#     total_night_minutes: float
#     total_night_calls: int
#     total_night_charge: float
#     total_intl_minutes: float
#     total_intl_calls: int
#     total_intl_charge: float
#     number_customer_service_calls: int

# # -------------------------------
# # Home route
# # -------------------------------
# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # -------------------------------
# # Prediction route
# # -------------------------------
# @app.post("/predict")
# def predict(data: CustomerData):
#     try:
#         # Check if model loaded
#         if model is None or scaler is None or cols is None:
#             return {"error": "Model files not loaded properly"}

#         df = pd.DataFrame([data.dict()])

#         # Encoding
#         df = pd.get_dummies(df)

#         # Match columns
#         for col in cols:
#             if col not in df.columns:
#                 df[col] = 0

#         df = df[cols]

#         # Scaling
#         df = scaler.transform(df)

#         # Prediction
#         prediction = model.predict(df)[0]
#         probability = model.predict_proba(df)[0][1]

#         return {
#             "prediction": int(prediction),
#             "probability": float(probability)
#         }

#     except Exception as e:
#         return {"error": str(e)}