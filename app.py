from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("final_model.keras")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]

    x = np.array(data).reshape(1, -1)

    x = scaler.transform(x)

     # CNN-LSTM input format
    x = x.reshape(1, x.shape[1], 1)

    prob = float(model.predict(x, verbose=0)[0][0])

    prediction = 1 if prob >= 0.5 else 0

    return jsonify({
        "fraud_probability": prob,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)