from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("model/insulin_dose_ann.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/home")
def home():
    return render_template("welcome.html")  

@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html")  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  
        input_features = np.array(data["features"]).reshape(1, -1)
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]
        return jsonify({"insulin_dose": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
