from flask import Flask, request, jsonify
import torch
from model import SentimentModel, preprocess
from collections import Counter
import string

app = Flask(__name__)

# Load the pre-trained sentiment analysis model
word2idx = {}  # You should load this from the model training process
model = SentimentModel(vocab_size=5000)  # Adjust vocab_size accordingly
model.load_state_dict(torch.load('sentiment_model.pth'))
model.eval()

# Tokenizer function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    # Preprocess the input text and convert to tensor
    tokenized_input = preprocess_text(text)
    input_indices = [word2idx.get(word, 0) for word in tokenized_input]
    input_indices = input_indices[:10] + [0] * (10 - len(input_indices))  # Pad to max_len=10
    input_tensor = torch.tensor([input_indices])
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        sentiment = 'positive' if torch.argmax(output) == 0 else 'negative'
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
