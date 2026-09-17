# ---------------------------------------------
# Character-Level Language Model (PyTorch, GPU)
# ---------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# ----- 1. Device setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ----- 2. Dataset -----
text = "hello world! this is a character level model demo."
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
block_size = 8  # sequence length
batch_size = 4

# ----- 3. Get batches -----
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ----- 4. Model -----
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        logits = self.fc(out)
        return logits, h

# ----- 5. Initialize model -----
model = CharRNN(vocab_size, embed_dim=64, hidden_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# ----- 6. Training -----
for epoch in range(200):
    x, y = get_batch()
    optimizer.zero_grad()
    logits, _ = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----- 7. Text generation -----
def generate(start_text="h", length=100):
    model.eval()
    chars = list(start_text)
    input = torch.tensor([encode(start_text)], dtype=torch.long).to(device)
    h = None
    for _ in range(length):
        logits, h = model(input, h)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        chars.append(itos[next_id])
        input = torch.tensor([[next_id]], dtype=torch.long).to(device)
    return ''.join(chars)

print("\n🧠 Generated text:\n", generate("h"))
