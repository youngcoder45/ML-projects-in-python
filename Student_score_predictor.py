# student_score_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Create or load data
# If you don’t have a file, here’s a small example dataset:
data = {
    "Hours_Studied": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    "Score":         [10,  12,   15,   17,   19,   20,   22,   24,   25,   27]
}
df = pd.DataFrame(data)

# If you have a CSV, uncomment this:
# df = pd.read_csv("student_scores.csv")  # with columns "Hours_Studied" and "Score"

# 2. Features and target
X = df[["Hours_Studied"]]
y = df["Score"]

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 4. Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.2f}")

# 7. Visualization
plt.scatter(X, y, label="Actual Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Model Prediction")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Student Score Predictor")
plt.legend()
plt.show()

# 8. Save model
import joblib
joblib.dump(model, "student_score_model.pkl")
print("Model saved to student_score_model.pkl")
