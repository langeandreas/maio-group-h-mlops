import pickle

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

Xy = load_diabetes(as_frame=True)

X = Xy.frame.drop(columns=["target"])

y = Xy.frame["target"] # acts as a "progression index" (higher = worse)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=134893)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

rmse = root_mean_squared_error(y_test, model.predict(X_test_scaled))

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)


if __name__ == "__main__":
    print(f"RMSE (v0.1): {rmse:.2f}") # for updating the readme