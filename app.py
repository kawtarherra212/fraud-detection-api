import gradio as gr
import numpy as np
import tensorflow as tf
import joblib

# Load model + scaler
model = tf.keras.models.load_model("final_model.keras")
scaler = joblib.load("scaler.pkl")

def predict(*features):
    x = np.array(features).reshape(1, -1)
    x = scaler.transform(x)

    # CNN-LSTM input format
    x = x.reshape(1, x.shape[1], 1)

    prob = float(model.predict(x, verbose=0)[0][0])

    prediction = "Fraud" if prob >= 0.5 else "Normal"

    return prediction, prob

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number() for _ in range(29)],
    outputs=["text", "number"],
    title="Fraud Detection Model"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
