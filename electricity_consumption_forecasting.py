# electricity_consumption_forecasting.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
df = pd.read_csv("electricity_consumption.csv")  
# Expected columns: Date, Consumption

# 2. Convert date & sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 3. Feature engineering: Convert date to numeric (days since start)
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# 4. Features & Target
X = df[['Days']]
y = df['Consumption']

# 5. Train/Test Split
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.3f}")

# 9. Visualization
plt.figure(figsize=(10,5))
plt.plot(df['Date'], y, label="Actual Consumption", color="blue")
plt.plot(df['Date'][split_index:], y_pred, label="Predicted Consumption", color="red")
plt.xlabel("Date")
plt.ylabel("Electricity Consumption (kWh)")
plt.title("Electricity Consumption Forecasting")
plt.legend()
plt.show()

# 10. Forecast future (example: next 30 days)
future_days = np.array(range(df['Days'].max()+1, df['Days'].max()+31)).reshape(-1, 1)
future_pred = model.predict(future_days)

future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=30)
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Consumption': future_pred})

print("\nNext 30 days forecast:")
print(forecast_df)

# 11. Save model
import joblib
joblib.dump(model, "electricity_forecast_model.pkl")
print("Model saved to electricity_forecast_model.pkl")
