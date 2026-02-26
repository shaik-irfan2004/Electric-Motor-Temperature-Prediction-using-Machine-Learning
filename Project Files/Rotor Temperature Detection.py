import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("C:\\Users\\abhim\\vs code\\irfan\\Electric_Motor_Temperature_Prediction\\pmsm_temperature_data.csv")

# Selecting features
X = df[['ambient_temp','coolant_temp','motor_speed','torque','current','voltage']]
y = df['motor_temp']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Performance
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Save model & scaler
joblib.dump(model, "model.save")
joblib.dump(scaler, "transform.save")

print("Model saved as model.save")
print("Scaler saved as transform.save")
