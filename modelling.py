# pylint: disable=invalid-name, too-many-locals
"""
This script trains a linear regression model on the Diabetes dataset, evaluates its performance,
and saves the trained model, scaler, and training metrics.

Modules:
    - pickle: For saving the scaler and model objects to disk.
    - sklearn.datasets: To load the Diabetes dataset.
    - sklearn.model_selection: For splitting the dataset into training and testing sets.
    - sklearn.preprocessing: For scaling the features using StandardScaler.
    - sklearn.linear_model: To use the LinearRegression model.
    - sklearn.metrics: For calculating the root mean squared error (RMSE).

Functions:
    - None

Workflow:
    1. Load the Diabetes dataset.
    2. Split the dataset into training and testing sets.
    3. Scale the features using StandardScaler.
    4. Train a LinearRegression model on the scaled training data.
    5. Evaluate the model using RMSE on the test data.
    6. Save the scaler, model, and training metrics to disk.

Outputs:
    - Scaler object saved as "models/scaler.pkl".
    - Trained model saved as "models/model.pkl".
    - Training metrics (including RMSE, random state, model parameters, and dataset description)
      saved as "models/training_metrics.txt".

Usage:
    Run this script directly to train the model and save the outputs. The RMSE will also be printed
    to the console for updating the README file.

Note:
    - The random state for train-test splitting is set to 134893 for reproducibility.
    - The Diabetes dataset's "target" column is used as the dependent variable.
"""

import json
import pickle
import time
import hashlib
from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 134893


def main(version: str = "v0.1"):
    """Main function to train the model and save the artifacts."""
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"]) # pylint: disable=no-member
    y = Xy.frame["target"] # pylint: disable=no-member

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    rmse = root_mean_squared_error(y_test, model.predict(X_test_scaled))

    Path("models").mkdir(exist_ok=True)

    # save artifacts
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # write JSON metrics (for CI/tests/artifacts)
    split_hash = hashlib.sha256(("".join(map(str, X_train.index))).encode()).hexdigest()[:12]
    metrics = {
        "version": version,
        "seed": SEED,
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_hash": split_hash,
        "model_params": model.get_params(),
        "ts": int(time.time()),
    }
    Path("models/training_metrics.txt").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # console print for README/Logs
    print(json.dumps({"RMSE (v0.1)": round(float(rmse), 2)}, indent=2))

if __name__ == "__main__":
    main()
