import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("fraud_dataset.csv")

# Features and label
X = df[["total_calls", "avg_duration", "night_calls", "unique_contacts",
        "imei_changes", "scam_keyword_flag", "short_call_ratio"]]
y = df["is_fraud"]

# Add small Gaussian noise for better generalization
X += np.random.normal(0, 0.05, X.shape)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# Random Forest model with mild regularization to reduce overconfidence
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42
)

# Train base model
rf.fit(X_train, y_train)

# Use sigmoid calibration (better for borderline predictions)
calibrated_rf = CalibratedClassifierCV(rf, method="sigmoid", cv=5)
calibrated_rf.fit(X_train, y_train)

# Save scaler and model
joblib.dump(scaler, "scaler.pkl")
joblib.dump(calibrated_rf, "fraud_model.pkl")

# Evaluate
y_pred = calibrated_rf.predict(X_test)
y_prob = calibrated_rf.predict_proba(X_test)[:, 1]
print("Model trained successfully.")
print(classification_report(y_test, y_pred, digits=3))

# Display confidence smoothing
print("\nSample predictions:")
for i in range(10):
    confidence = y_prob[i] * 100
    # Smooth extreme probabilities (avoid 99% spikes)
    if confidence > 90:
        confidence = 80 + (confidence - 90) * 0.3
    elif confidence < 10:
        confidence = 20 - (10 - confidence) * 0.3
    print(f"Case {i+1}: Fraud Probability = {confidence:.1f}%, Actual = {y_test.iloc[i]}")


