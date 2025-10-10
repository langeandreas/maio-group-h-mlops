# main.py
"""
This script implements a FastAPI application for predicting diabetes progression 
based on input features. It defines an API with two endpoints:
1. `/health` (GET): A health check endpoint that returns the status of the API 
    and the version of the model being used.
2. `/predict` (POST): An endpoint that accepts input features for diabetes 
    progression prediction, scales the input using a pre-trained scaler, and 
    uses a pre-trained machine learning model to make predictions.
Classes:
    DiabetesInput (BaseModel): A Pydantic model representing the input features 
    required for the prediction.
Global Variables:
    app (FastAPI): The FastAPI application instance.
    MODEL_VERSION (str): The version of the machine learning model.
    model: The pre-trained machine learning model loaded from a pickle file.
    scaler: The pre-trained scaler loaded from a pickle file.
Functions:
    health(): A health check endpoint that returns the API status and model version.
    predict(data: DiabetesInput) -> dict[str, float] | dict[str, str]: 
        An endpoint that predicts diabetes progression based on input features."""
import pickle

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class DiabetesInput(BaseModel):
    """
    FastAPI input data class representing the input features for predicting diabetes progression.
    Attributes:
        age (float): Age of the patient.
        sex (float): Gender of the patient, encoded as a numerical value.
        bmi (float): Body Mass Index (BMI) of the patient.
        bp (float): Average blood pressure of the patient.
        s1 (float): T-Cells (a type of white blood cell) count.
        s2 (float): Low-Density Lipoproteins (LDL) level.
        s3 (float): High-Density Lipoproteins (HDL) level.
        s4 (float): Total cholesterol level.
        s5 (float): Serum triglycerides level.
        s6 (float): Blood sugar level.
    """
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

app = FastAPI()
MODEL_VERSION = "v0.1"
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.get("/health")
def health() -> dict[str, str]:
    """
    Provides the health status of the application.
    Returns:
        dict[str, str]: A dictionary containing the health status and the model version.
    """
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
async def predict(data: DiabetesInput) -> dict[str, float] | dict[str, str]:
    """
    Predict the diabetes progression based on input features sent to the API endpoint.

    

    Args:
        data (DiabetesInput): An object containing the following attributes:
            - age (float): Age of the patient.
            - sex (float): Sex of the patient.
            - bmi (float): Body mass index.
            - bp (float): Average blood pressure.
            - s1 (float): TCH (total cholesterol).
            - s2 (float): LDL (low-density lipoproteins).
            - s3 (float): HDL (high-density lipoproteins).
            - s4 (float): TCH/HDL ratio.
            - s5 (float): Log of serum triglycerides level.
            - s6 (float): Blood sugar level.

    Returns:
        dict[str, float] | dict[str, str]: A dictionary containing either:
            - "prediction" (float): The predicted diabetes progression value.
            - "error" (str): An error message if an exception occurs.
    """

    try:
        features = np.array([[
            data.age, data.sex, data.bmi, data.bp,
            data.s1, data.s2, data.s3, data.s4, data.s5, data.s6
            ]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return {"prediction": float(prediction)}
    except (ValueError, TypeError, RuntimeError) as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
