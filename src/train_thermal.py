import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Load Data
df = pd.read_csv('data/raw/device_thermal_dataset.csv')

# 2. Encode Categoricals (Essential for ML to understand text like 'iPhone')
le_model = LabelEncoder()
df['device_model_enc'] = le_model.fit_transform(df['device_model'])

le_cpu = LabelEncoder()
df['cpu_load_enc'] = le_cpu.fit_transform(df['cpu_load'])

# 3. Train ONLY on NORMAL data
# We want the model to learn: "How hot does a Pixel 8 get normally?"
normal_df = df[df['context'] != 'thermal_anomaly'].copy()

X = normal_df[['device_model_enc', 'cpu_load_enc', 'battery_pct', 'is_charging']]
y = normal_df['battery_temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Baseline Regressor
# We use max_depth=10 to keep it lightweight for mobile deployment
thermal_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
thermal_model.fit(X_train, y_train)

# 5. Evaluate
y_pred = thermal_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Calibration Error: {mae:.4f}°C")
print("Interpretation: Any heat spike > 1.0°C is likely external energy waste.")

# 6. Save everything
joblib.dump(thermal_model, 'models/thermal_baseline_model.pkl')
joblib.dump(le_model, 'models/le_device_model.pkl')
joblib.dump(le_cpu, 'models/le_cpu_load.pkl')

print("Thermal Intelligence files saved to /models.")