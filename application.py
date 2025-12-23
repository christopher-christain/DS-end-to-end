from flask import Flask, request, jsonify, render_template
import joblib
import os

application = Flask(__name__)

MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        model_path = os.path.join("artifacts", "best_model.pkl")
        MODEL = joblib.load(model_path)
    return MODEL


@application.route("/")
def home():
    return "Student Score Predictor is running"


@application.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model = load_model()

    features = [
        data["reading_score"],
        data["writing_score"]
    ]

    prediction = model.predict([features])[0]
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)
