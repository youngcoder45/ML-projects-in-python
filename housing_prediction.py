# housing_prediction.py
"""
Project 5: Boston (or California fallback) Housing Price Prediction
- Loads Boston housing if available; otherwise uses California housing.
- Preprocesses features, uses pipelines, compares Ridge (linear + regularization)
  and RandomForestRegressor (nonlinear).
- Uses GridSearchCV for hyperparameter tuning, prints RMSE and R2, and saves best model.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_openml, fetch_california_housing
import joblib

def load_dataset():
    """
    Try to load Boston from OpenML (if available). If not available,
    fall back to California housing.
    Returns X (DataFrame), y (Series), dataset_name (str)
    """
    try:
        # Try Boston via openml (version may vary)
        boston = fetch_openml(name="Boston", version=1, as_frame=True)  # may fail in newer sklearn
        X = boston.data
        y = boston.target.astype(float)
        return X, y, "Boston"
    except Exception:
        # Fallback
        cal = fetch_california_housing(as_frame=True)
        X = cal.data
        y = cal.target.astype(float)
        return X, y, "California"

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def main():
    X, y, name = load_dataset()
    print(f"Using dataset: {name} (features: {X.shape[1]}, samples: {X.shape[0]})\n")

    # Quick EDA prints (simple)
    print("First 5 rows of features:")
    print(X.head(), "\n")
    print("Target sample values (first 10):")
    print(y.head(10).to_list(), "\n")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- Model 1: Ridge (linear with L2 regularization) ----------
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    ridge_params = {
        "ridge__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
    }

    ridge_search = GridSearchCV(
        ridge_pipeline, ridge_params,
        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0
    )
    print("Tuning Ridge (this is fast)...")
    ridge_search.fit(X_train, y_train)
    print("Best Ridge params:", ridge_search.best_params_)

    ridge_best = ridge_search.best_estimator_
    ridge_cv_scores = -cross_val_score(ridge_best, X_train, y_train,
                                      scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    print(f"Ridge CV RMSE (mean ± std): {ridge_cv_scores.mean():.3f} ± {ridge_cv_scores.std():.3f}")

    # Evaluate on test
    y_pred_ridge = ridge_best.predict(X_test)
    print(f"Ridge Test RMSE: {rmse(y_test, y_pred_ridge):.3f}")
    print(f"Ridge Test R²: {r2_score(y_test, y_pred_ridge):.3f}\n")

    # ---------- Model 2: Random Forest Regressor ----------
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),  # scaling isn't required for tree-based, but kept for pipeline uniformity
        ("rf", RandomForestRegressor(random_state=42))
    ])

    rf_params = {
        "rf__n_estimators": [50, 100],            # try small values first; increase if you have time/GPU
        "rf__max_depth": [None, 8, 12],
        "rf__min_samples_split": [2, 5]
    }

    rf_search = GridSearchCV(
        rf_pipeline, rf_params,
        cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0
    )
    print("Tuning RandomForest (may take longer)...")
    rf_search.fit(X_train, y_train)
    print("Best RF params:", rf_search.best_params_)

    rf_best = rf_search.best_estimator_
    rf_cv_scores = -cross_val_score(rf_best, X_train, y_train,
                                    scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
    print(f"RF CV RMSE (mean ± std): {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")

    # Evaluate on test
    y_pred_rf = rf_best.predict(X_test)
    print(f"RF Test RMSE: {rmse(y_test, y_pred_rf):.3f}")
    print(f"RF Test R²: {r2_score(y_test, y_pred_rf):.3f}\n")

    # ---------- Choose best model ----------
    if rmse(y_test, y_pred_rf) < rmse(y_test, y_pred_ridge):
        best_model = rf_best
        best_name = "RandomForest"
        best_rmse = rmse(y_test, y_pred_rf)
    else:
        best_model = ridge_best
        best_name = "Ridge"
        best_rmse = rmse(y_test, y_pred_ridge)

    print(f"Selected best model: {best_name} with Test RMSE = {best_rmse:.3f}")

    # Save model and basic metadata
    model_filename = f"{name.lower()}_{best_name.lower()}_model.pkl"
    joblib.dump({
        "model": best_model,
        "dataset": name,
        "feature_names": list(X.columns),
        "target_mean": float(y.mean())
    }, model_filename)
    print(f"Saved best model plus metadata to: {model_filename}")

    # Example: show predictions vs actual for first 10 test samples
    preds = best_model.predict(X_test)[:10]
    actuals = y_test.values[:10]
    print("\nExample predictions (first 10 test samples):")
    for i, (a, p) in enumerate(zip(actuals, preds), 1):
        print(f"{i:2d}. Actual: {a:.3f}  Pred: {p:.3f}")

if __name__ == "__main__":
    main()
