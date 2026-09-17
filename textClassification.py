# ---------------------------------------------
# Text Classification without torchtext (Python 3.12 Safe)
# ---------------------------------------------
import re
import torch
import torch.nn as nn
import torch.optim as optim

# ----- 1. Dataset -----
data = [
    ("I love this movie", 1),
    ("Terrible film", 0),
    ("Great story", 1),
    ("Worst acting", 0),
]

# ----- 2. Tokenization -----
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Build vocabulary
vocab = {}
for text, _ in data:
    for word in tokenize(text):
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)

# ----- 3. Text to Tensor -----
def text_to_tensor(text):
    tokens = tokenize(text)
    return torch.tensor([vocab.get(w, 0) for w in tokens])

# ----- 4. Model -----
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.emb(x).mean(dim=0)
        return self.fc(x)

# ----- 5. Training -----
model = TextClassifier(vocab_size, 50, 2)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    for text, label in data:
        x = text_to_tensor(text)
        y = torch.tensor([label])
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out.unsqueeze(0), y)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1} done")

# ----- 6. Prediction -----
def predict(sentence):
    x = text_to_tensor(sentence)
    out = model(x)
    return "Positive" if torch.argmax(out).item() == 1 else "Negative"

print("\nPredictions:")
print("I really liked it! →", predict("I really liked it!"))
print("It was awful. →", predict("It was awful."))
