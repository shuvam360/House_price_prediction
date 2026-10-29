import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "house_price_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes so frontend (GitHub Pages, Vercel, localhost) can query it
CORS(app, resources={r"/*": {"origins": "*"}})

# Global model and metrics variables
model = None
metrics = {}

def load_model():
    global model, metrics
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

# Load model upon startup
load_model()

FEATURE_MAPPING = {
    "OverallQual": ["OverallQual", "overall_qual", "overallQual", "quality"],
    "GrLivArea": ["GrLivArea", "gr_liv_area", "grLivArea", "living_area", "livingArea"],
    "TotalBsmtSF": ["TotalBsmtSF", "total_bsmt_sf", "totalBsmtSF", "basement_area", "basementArea"],
    "GarageCars": ["GarageCars", "garage_cars", "garageCars", "garage"],
    "FullBath": ["FullBath", "full_bath", "fullBath", "bathrooms"],
    "YearBuilt": ["YearBuilt", "year_built", "yearBuilt", "year"]
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "House Price Prediction API",
        "status": "online",
        "model_loaded": model is not None,
        "endpoints": {
            "GET /": "Service overview",
            "GET /health": "Health check",
            "GET /api/metrics": "Model metrics and weights",
            "POST /api/predict": "Inference endpoint for estimating house price"
        }
    }), 200

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200

@app.route("/metrics", methods=["GET"])
@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    global model
    if model is None:
        load_model()
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded. Please train the model first."
            }), 503

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Invalid JSON payload provided."
        }), 400

    # Extract features using flexible key matching
    feature_values = {}
    missing_fields = []

    for standard_feature, aliases in FEATURE_MAPPING.items():
        found = False
        for alias in aliases:
            if alias in data and data[alias] is not None:
                try:
                    feature_values[standard_feature] = float(data[alias])
                    found = True
                    break
                except (ValueError, TypeError):
                    return jsonify({
                        "status": "error",
                        "message": f"Field '{alias}' must be a valid number."
                    }), 422

        if not found:
            # Fallback to training median if available
            default_val = metrics.get("medians", {}).get(standard_feature)
            if default_val is not None:
                feature_values[standard_feature] = float(default_val)
            else:
                missing_fields.append(standard_feature)

    if missing_fields:
        return jsonify({
            "status": "error",
            "message": f"Missing required fields: {', '.join(missing_fields)}"
        }), 422

    try:
        # Create single-row DataFrame matching the model training columns
        input_df = pd.DataFrame([[
            feature_values["OverallQual"],
            feature_values["GrLivArea"],
            feature_values["TotalBsmtSF"],
            feature_values["GarageCars"],
            feature_values["FullBath"],
            feature_values["YearBuilt"]
        ]], columns=["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath", "YearBuilt"])

        raw_price = float(model.predict(input_df)[0])
        predicted_price = max(0, round(raw_price))
        formatted_price = f"${predicted_price:,.0f}"

        return jsonify({
            "status": "success",
            "predicted_price": predicted_price,
            "formatted_price": formatted_price,
            "currency": "USD",
            "input_received": feature_values,
            "model": metrics.get("model_name", "LinearRegression"),
            "r2_score": metrics.get("metrics", {}).get("r2_score", 0.7944)
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
