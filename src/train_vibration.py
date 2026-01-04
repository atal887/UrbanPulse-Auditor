import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from features import get_fft_features # Ensure your features.py is in the same folder
import joblib

# 1. Load Data
# Ensure your dataset is in the data/raw/ folder as per our structure
df = pd.read_csv('data/raw/vibration_dataset_combined.csv').dropna(subset=['accel_x'])

# 2. Extract FFT Features
print("Extracting FFT features (Frequency Domain)...")
X, y = get_fft_features(df)

# 3. Model with Regularization (To prevent Overfitting)
# max_depth=10: Prevents trees from becoming too complex
# min_samples_leaf=5: Ensures the model doesn't create rules for single data points
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_leaf=5, 
    random_state=42,
    class_weight='balanced' # Handles the imbalance between traffic and other labels
)

# 4. Cross-Validation (The "Truth" Test)
# This splits the data 5 different ways and tests 5 times
print("Running 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv)

print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# 5. Final Train/Test Split for Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)

# 6. Detailed Metrics
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
y_pred = model.predict(X_test)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the robust model
joblib.dump(model, 'models/vibration_classifier_robust.pkl')
print("\nRobust model saved to models/vibration_classifier_robust.pkl")