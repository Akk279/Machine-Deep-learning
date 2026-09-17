import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------------
# Sample implicit feedback dataset
# -------------------------------
# Each row: (user_id, positive_item_id, negative_item_id)
data = np.array([
    [0, 1, 3],
    [0, 2, 4],
    [1, 2, 0],
    [1, 3, 1],
    [2, 4, 1],
    [2, 0, 3],
])

num_users = len(set(data[:, 0]))
num_items = 5

user_ids = torch.LongTensor(data[:, 0])
pos_items = torch.LongTensor(data[:, 1])
neg_items = torch.LongTensor(data[:, 2])

# -------------------------------
# Neural Recommender Model
# -------------------------------
class NeuralImplicitRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=8):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)

    def forward(self, users, items):
        user_vec = self.user_embed(users)
        item_vec = self.item_embed(items)
        return user_vec, item_vec

# -------------------------------
# Triplet Loss (Margin-based)
# -------------------------------
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, user_vec, pos_vec, neg_vec):
        # Cosine similarity (higher = more similar)
        pos_sim = torch.sum(user_vec * pos_vec, dim=1)
        neg_sim = torch.sum(user_vec * neg_vec, dim=1)
        loss = torch.clamp(self.margin + neg_sim - pos_sim, min=0)
        return loss.mean()

# -------------------------------
# Training
# -------------------------------
model = NeuralImplicitRecommender(num_users, num_items)
criterion = TripletLoss(margin=0.3)
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()

    u_vec, pos_vec = model(user_ids, pos_items)
    _, neg_vec = model(user_ids, neg_items)

    loss = criterion(u_vec, pos_vec, neg_vec)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------------------
# Test recommendation
# -------------------------------
model.eval()
with torch.no_grad():
    test_user = torch.LongTensor([0])
    all_items = torch.arange(num_items)
    user_vec, all_vecs = model(test_user.repeat(num_items), all_items)
    scores = torch.sum(user_vec * all_vecs, dim=1)
    ranked_items = torch.argsort(scores, descending=True)
    print("\nRecommended items for user 0 (highest to lowest score):", ranked_items.tolist())
