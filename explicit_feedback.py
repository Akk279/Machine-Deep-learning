
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. DATA (Explicit Feedback)
# -----------------------------
data = np.array([
    [0, 0, 5],
    [0, 1, 3],
    [0, 2, 1],
    [1, 0, 4],
    [1, 1, 2],
    [2, 1, 5],
    [2, 2, 4],
], dtype=np.int64)

num_users = len(set(data[:, 0]))
num_items = len(set(data[:, 1]))

user_ids = torch.LongTensor(data[:, 0])
item_ids = torch.LongTensor(data[:, 1])
ratings = torch.FloatTensor(data[:, 2])

# -----------------------------
# 2. MODEL DEFINITION
# -----------------------------
class NeuralRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=2, hidden_dim=8):
        super(NeuralRecommender, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embed(user_ids)
        item_vec = self.item_embed(item_ids)
        x = torch.cat([user_vec, item_vec], dim=1)
        out = self.fc_layers(x)
        return out.squeeze()

# -----------------------------
# 3. TRAINING
# -----------------------------
model = NeuralRecommender(num_users, num_items)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()
    preds = model(user_ids, item_ids)
    loss = criterion(preds, ratings)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -----------------------------
# 4. VISUALIZE EMBEDDINGS
# -----------------------------
user_embeds = model.user_embed.weight.detach().numpy()
item_embeds = model.item_embed.weight.detach().numpy()

plt.figure(figsize=(6, 5))
plt.scatter(user_embeds[:, 0], user_embeds[:, 1], color='blue', label='Users', s=80)
plt.scatter(item_embeds[:, 0], item_embeds[:, 1], color='red', label='Items', s=80)

# Label points
for i in range(num_users):
    plt.text(user_embeds[i, 0]+0.02, user_embeds[i, 1], f'U{i}', fontsize=9, color='blue')
for j in range(num_items):
    plt.text(item_embeds[j, 0]+0.02, item_embeds[j, 1], f'I{j}', fontsize=9, color='red')

plt.title("2D Visualization of User & Item Embeddings")
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")
plt.legend()
plt.grid(True)
plt.show()
