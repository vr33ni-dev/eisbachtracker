from flask import Flask, request, jsonify, make_response
import joblib
import numpy as np
import pandas as pd
import os
import threading
import time

app = Flask(__name__)

# ---- readiness state ----
MODEL = None
READY = False
LOAD_ERROR = None

def load_model_bg():
    global MODEL, READY, LOAD_ERROR
    try:
        # If your model load is slow, we keep serving /health quickly while it loads
        MODEL = joblib.load("surfer_prediction_model.pkl")
        READY = True
    except Exception as e:
        LOAD_ERROR = str(e)
        READY = False

# Kick off background load at import time (works in gunicorn too)
threading.Thread(target=load_model_bg, daemon=True).start()

# ---- helpers ----
def not_ready_response(use429=False, retry_seconds=5):
    """Return a consistent JSON + Retry-After while 'warming up'."""
    status = 429 if use429 else 503
    body = {
        "error": {
            "code": status,
            "message": "Prediction service is waking up (cold start). Please retry shortly.",
        },
        "ready": READY,
        "hint": "This often happens on Render free tier right after spin-up.",
        "retry_after_seconds": retry_seconds,
    }
    resp = make_response(jsonify(body), status)
    resp.headers["Retry-After"] = str(retry_seconds)
    return resp

def require_fields(data, fields):
    missing = [f for f in fields if f not in data]
    if missing:
        return jsonify({"error": {"code": 400, "message": f"Missing fields: {', '.join(missing)}"}}), 400

# ---- routes ----
@app.get("/")
def home():
    return jsonify({
        "service": "Surfer prediction API",
        "ready": READY,
        "endpoints": {
            "GET /health": "readiness + load error if any",
            "POST /predict": "make a prediction (JSON body)"
        }
    })

@app.get("/health")
def health():
    return jsonify({
        "ready": READY,
        "error": LOAD_ERROR,
    })

@app.route("/predict", methods=["POST"])
def predict():
    # If model still loading (or failed), report clearly
    if not READY or MODEL is None:
        # choose 429 to mirror Render, or 503 (more semantically correct)
        use429 = os.environ.get("USE_429_WHEN_NOT_READY", "false").lower() == "true"
        return not_ready_response(use429=use429, retry_seconds=5)

    data = request.get_json(silent=True) or {}
    missing_resp = require_fields(data, ["hour", "water_temp", "air_temp", "water_level"])
    if missing_resp:
        return missing_resp

    feature_dict = {
        "hour": data["hour"],
        "water_temp": data["water_temp"],
        "air_temp": data["air_temp"],
        "water_level": data["water_level"],
        "weather_condition": data.get("weather_condition"),
    }
    features = pd.DataFrame([feature_dict])

    surfer_count = 0
    explanation = {k: 0.0 for k in ["hour", "water_temp", "air_temp", "water_level", "weather_condition"]}

    # Your rule: no surfers below threshold
    if feature_dict["water_level"] >= 130:
        try:
            prediction = MODEL.predict(features)
            surfer_count = max(0, int(prediction[0]))

            # Only compute contributions if the model exposes .coef_
            if hasattr(MODEL, "coef_"):
                coefficients = MODEL.coef_
                explanation = {
                    feature: float(coef) * float(features.iloc[0][feature])
                    for feature, coef in zip(features.columns, coefficients)
                }
        except Exception as e:
            # Return a clean 500 with detail so callers can surface it nicely
            return jsonify({
                "error": {"code": 500, "message": "Could not compute prediction", "detail": str(e)}
            }), 500

    return jsonify({"surfer_count": surfer_count, "explanation": explanation})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
