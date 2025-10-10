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
import pickle

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Xy = load_diabetes(as_frame=True)

X = Xy.frame.drop(columns=["target"]) # pylint: disable=no-member

y = Xy.frame["target"] # pylint: disable=no-member

dataset_description = Xy.DESCR # pylint: disable=no-member

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=134893)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

rmse = root_mean_squared_error(y_test, model.predict(X_test_scaled))
doc = f"""RMSE: {rmse:.2f}\n
#{"#"*60}\n
Random_State: 134893\n
#{"#"*60}\n
model_params: {model.get_params()}\n
#{"#"*60}\n
Dataset: {dataset_description}"""
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/training_metrics.txt", "w", encoding="utf-8") as f:
    f.write(doc)


if __name__ == "__main__":
    print(f"RMSE (v0.1): {rmse:.2f}") # for updating the readme
