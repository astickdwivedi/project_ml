import torch
import torch.nn as nn
from collections import Counter
import string

# Simple LSTM model for sentiment analysis
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)  # 2 classes: positive, negative

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Helper function to preprocess input text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Example data for training (replace with real data)
texts = ["I love this!", "This is terrible.", "I am so happy.", "I hate this!"]
labels = ['positive', 'negative', 'positive', 'negative']

# Tokenize and label encode
tokenized_texts = [preprocess(text) for text in texts]
vocab = Counter(word for sentence in tokenized_texts for word in sentence)
word2idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}  # Reserve 0 for padding

# Convert labels to numbers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Prepare dataset (You can modify this for your own dataset)
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, word2idx, max_len=10):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Convert words to indices and pad/truncate
        indices = [self.word2idx.get(word, 0) for word in text]
        indices = indices[:self.max_len] + [0] * (self.max_len - len(indices))
        return torch.tensor(indices), torch.tensor(label)

train_dataset = SentimentDataset(tokenized_texts, encoded_labels, word2idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# Instantiate and train the model
model = SentimentModel(vocab_size=len(word2idx) + 1)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item()}")

# Save the model after training
torch.save(model.state_dict(), 'sentiment_model.pth')
