# pylint: disable=invalid-name, too-many-locals
"""
v0.2 training script:
- Tries Ridge and RandomForestRegressor
- Uses StandardScaler on features
- Picks the best model by RMSE
- Calibrates a simple "high-risk" flag using a threshold on the progression index
- Saves artifacts (scaler.pkl, model.pkl) and JSON metrics

Outputs:
  - models/scaler.pkl
  - models/model.pkl
  - models/training_metrics.txt

Usage:
  python modelling.py
"""

import json
import pickle
import time
import hashlib
from pathlib import Path

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

SEED = 8


def rmse_of(y_true, y_pred):
    """Compute RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def find_best_threshold(y_true_bin, y_score):
    """Pick threshold that maximizes F1 on training scores."""
    thresholds = np.linspace(float(np.min(y_score)), float(np.max(y_score)), 301)
    best_f1, best_t = -1.0, thresholds[0]
    for t in thresholds:
        y_hat = (y_score >= t).astype(int)
        f1 = f1_score(y_true_bin, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def main(version: str = "v0.2"):
    """Train, evaluate, calibrate threshold, save artifacts + metrics."""
    # data
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])  # pylint: disable=no-member
    y = Xy.frame["target"]  # pylint: disable=no-member

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True
    )

    # polynomial features, more features for a complex model to perform well
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Define parameter grids for each model
    ridge_params = {"alpha": [0.05, 0.1, 0.2, 0.3]}
    rf_params = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}


    # Initialize models
    ridge = Ridge(random_state=SEED)
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    # Initialize Support Vector Regressor

    linear_reg = LinearRegression()
    linear_reg.fit(X_train_poly, y_train)

    ridge_grid = GridSearchCV(
        ridge,
        ridge_params,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1)
    ridge_grid.fit(X_train_poly, y_train)

    rf_grid = GridSearchCV(rf, rf_params, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    rf_grid.fit(X_train_poly, y_train)


    # Use the best estimators from GridSearchCV as candidates
    candidates = {
        "LinearRegression": linear_reg,
        "Ridge": ridge_grid.best_estimator_,
        "RandomForestRegressor": rf_grid.best_estimator_,
    }

    # train + select
    scores = {}
    trained = {}
    for name, model in candidates.items():
        model.fit(X_train_poly, y_train)
        rmse = rmse_of(y_test, model.predict(X_test_poly))
        scores[name] = rmse
        trained[name] = model
        print(f"{name} RMSE: {rmse:.2f}")

    best_name = min(scores, key=scores.get)
    best_model = trained[best_name]
    best_rmse = scores[best_name]

    # simple calibration to a "high-risk" flag
    top_pct = 0.90
    q = float(y_train.quantile(top_pct))
    y_train_true_bin = (y_train.values >= q).astype(int)

    # choose threshold on predicted scores to maximize F1 on train
    train_scores = best_model.predict(X_train_poly)
    thr, f1_at_thr = find_best_threshold(y_train_true_bin, train_scores)

    # eval calibrated threshold on test
    test_scores = best_model.predict(X_test_poly)
    y_test_true_bin = (y_test.values >= q).astype(int)
    y_test_pred_bin = (test_scores >= thr).astype(int)
    prec = float(precision_score(y_test_true_bin, y_test_pred_bin, zero_division=0))
    rec = float(recall_score(y_test_true_bin, y_test_pred_bin, zero_division=0))

    # out dir
    Path("models").mkdir(exist_ok=True)

    # save artifacts
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/poly_features.pkl", "wb") as f:
        pickle.dump(poly, f)

    split_hash = hashlib.sha256(("".join(map(str, X_train.index))).encode()).hexdigest()[:12]
    metrics = {
        "version": version,
        "seed": SEED,
        "model_name": best_name,
        "rmse": float(best_rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_hash": split_hash,
        "model_params": best_model.get_params(),
        "calibration": {
            "risk_top_quantile": top_pct,
            "train_threshold": thr,
            "train_f1_at_threshold": f1_at_thr,
            "test_precision_at_threshold": prec,
            "test_recall_at_threshold": rec,
            "true_risk_cutoff_value": q,
        },
        "ts": int(time.time()),
    }
    Path("models/training_metrics.txt").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps({"RMSE (v0.2)": round(float(best_rmse), 2), "model": best_name}, indent=2))


if __name__ == "__main__":
    main()
