from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature columns the scaler/model were trained on, in the required order.
FEATURE_COLUMNS = list(scaler.feature_names_in_)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if file is None or file.filename == "":
            return "❌ No file selected. Please choose a CSV file to upload."

        df = pd.read_csv(file)

        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            return f"❌ Uploaded CSV is missing required column(s): {', '.join(missing)}"

        # Select only the columns the model expects (in the right order) and
        # ignore any extra columns, e.g. a "number" identifier or label
        # columns like "is_fraud"/"fraud_prob" that may be present in
        # sample/training datasets uploaded for testing.
        X = df[FEATURE_COLUMNS]
        X_scaled = scaler.transform(X)

        probs = model.predict_proba(X_scaled)
        preds = model.predict(X_scaled)

        df["prediction"] = np.where(preds == 1, "Fraud", "Not Fraud")
        df["confidence"] = (probs.max(axis=1) * 100).round(2)

        return render_template("result.html", tables=df.to_dict(orient="records"))
    except Exception as e:
        return f"❌ Unexpected Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
