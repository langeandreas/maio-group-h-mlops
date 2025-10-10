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
{"prediction: float.2f}, 2 decimals
