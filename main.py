from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the LSTM model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open("tokenizor.pkl", 'rb') as file:
    tokenizor = pickle.load(file)

# Prediction function
def predict_next_word(model, tokenizor, text, max_sequence_len):
    token_list = tokenizor.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Match length with max_sequence_len

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    predicted = model.predict(token_list, verbose=0)
    predict_word_index = np.argmax(predicted, axis=1)

    for index, word in tokenizor.index_word.items():
        if index == predict_word_index:
            return word
    return "Not Found"

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    next_word = None
    if request.method == "POST":
        input_text = request.form["input_text"]
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizor, input_text, max_sequence_len)
    return render_template("index.html", next_word=next_word , input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
