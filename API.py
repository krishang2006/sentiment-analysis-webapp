from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load model
model = load_model("lstm_word2vec.keras")

# Load tokenizer (not the Word2Vec object!)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set max length (must match what you used during training)
MAX_LEN = 100

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Convert to padded sequence
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # Predict with LSTM
    pred = model.predict(padded)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"

    return jsonify({
        "input": text,
        "prediction": sentiment,
        "confidence": round(float(pred), 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
