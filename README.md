# maio-group-h-mlops
Repository for the MAIO assignment in MLOps by Group H

## API

### /health
Returns API status and model version

### /predict
Returns a prediction for the input in the POST-request

#### POST Input (json):
{"age": [float], "sex": [float], "bmi": [float], "bp": [float], "s1": [float], "s2": [float], "s3": [float], "s4": [float], "s5": [float], "s6": [float]}

#### Response shape
{"prediction: float.2f"}, 2 decimals

## Model
v0.1: Linear Regression using a Standard Scaler, the exact model metrics can be found in models/training_metrics.txt


### Reproduction instructions:
1. Clone the repository: 
```git clone https://github.com/langeandreas/maio-group-h-mlops.git```
2. Install the environment dependencies 
- either using poetry ```poetry install```
- or creating a virtual environment and installing the dependencies
```
python -m venv venv_folder_name && ./venv/Scripts/activate
pip install -r requirements.txt
```
3. Run ./modelling.py
The resulting model will be saved to models/ alongside metrics and a dataset description
4. Host the API locally by running app/app.py
