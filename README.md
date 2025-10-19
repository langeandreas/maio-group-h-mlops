# Assignment 3: **Virtual Diabetes Clinic Triage**
> **Group H** Louis Arbey, Luca Fynn Eckelmann, Andres Lange & Shalinda Silva

# Quickstart (Local)

1. Clone the repository

```bash
git clone https://github.com/langeandreas/maio-group-h-mlops.git
```

2. Install
```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
```

With Poetry: ```poetry install```

3. Train (v0.1)
```bash
   python modelling.py
```
Outputs: models/model.pkl, models/training_metrics.txt

4. Run API
```bash
   uvicorn app.app:app --host 0.0.0.0 --port 8000
```

# API Endpoints
### /health
Returns API status and model version

### /predict
Returns a prediction for the input in the POST request

#### POST Input (format):
```json
{"age": [float], "sex": [float], "bmi": [float], "bp": [float], "s1": [float], "s2": [float], "s3": [float], "s4": [float], "s5": [float], "s6": [float]}
```

#### Response shape
```json
{"prediction: float"}
```

# Docker
Build & run:
```bash
docker build -t ghcr.io/langeandreas/maio-group-h-mlops:<version> .
docker run -p 8000:8000 ghcr.io/langeandreas/maio-group-h-mlops:<version>
```
Version should be in this format: ```v0.x``` (e.g. ```v0.1```)

## Example of CURL requests
Health:
```bash
curl localhost:8000/health
```
Predict:
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "content-type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

# Testing & Linting
To run the tests
```bash
pytest -q
```
To run the linting
```bash
pylint app modelling.py
```

# Changelog
All releases are accessible here: [RELEASES](https://github.com/langeandreas/maio-group-h-mlops/releases)

## v0.1: Initial release
Linear Regression using a Standard Scaler, the exact model metrics can be found in models/training_metrics.txt


## v0.2: Improved model and preprocessing
Ridge Regression using polynomial features, model training metrics can be found in models/training_metrics.txt
