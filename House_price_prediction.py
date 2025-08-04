# house_price_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("train.csv")

# 2. Basic EDA & cleaning
#    - Drop columns with too many missing values
#    - Fill small missing with median
missing = df.isnull().sum().sort_values(ascending=False)
drop_cols = missing[missing > len(df)*0.5].index
df = df.drop(columns=drop_cols)

for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# 3. Feature selection
#    - For simplicity, take numeric features only
num_feats = df.select_dtypes(include=[np.number]).columns.drop("SalePrice")
X = df[num_feats]
y = df["SalePrice"]

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Modeling
models = {
    "LinearRegression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1)
}

for name, model in models.items():
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train,
                             scoring="neg_root_mean_squared_error", cv=5)
    rmse_cv = -scores.mean()
    
    # Fit & evaluate on test set
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    
    print(f"=== {name} ===")
    print(f"CV RMSE: {rmse_cv:.2f}")
    print(f"Test RMSE: {rmse_test:.2f}")
    print(f"Test R²: {r2:.3f}\n")

# 6. Save the best model (example: Ridge)
import joblib
best_model = Ridge(alpha=1.0)
best_model.fit(X, y)  # retrain on all data
joblib.dump(best_model, "house_price_model.pkl")
print("Saved best_model as house_price_model.pkl")
