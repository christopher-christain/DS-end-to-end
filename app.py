import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load artifacts
model = joblib.load("artifacts/best_model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Required numeric validation
        for col in ["reading score", "writing score"]:
            if col not in data:
                raise ValueError(f"{col} is required")

            value = float(data[col])
            if value < 0 or value > 100:
                raise ValueError(f"{col} must be between 0 and 100")
            data[col] = value

        df = pd.DataFrame([data])

        X_transformed = preprocessor.transform(df)
        prediction = model.predict(X_transformed)[0]

        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", error=str(e))
