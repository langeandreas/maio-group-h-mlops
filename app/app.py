# main.py
import uvicorn
import pickle

import numpy as np


from fastapi import FastAPI
from pydantic import BaseModel



# FastAPI input data class
class DiabetesInput(BaseModel):
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
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
async def predict(input: DiabetesInput):
    try:
        features = np.array([[
            input.age, input.sex, input.bmi, input.bp, input.s1, input.s2, input.s3, input.s4, input.s5, input.s6
            ]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)