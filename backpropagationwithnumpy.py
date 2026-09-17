import numpy as np

# -----------------------------
# 1. Activation Functions
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # derivative assuming input is sigmoid(x)
    return x * (1 - x)

# -----------------------------
# 2. Dataset (XOR problem)
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# -----------------------------
# 3. Initialize Weights
# -----------------------------
np.random.seed(42)
input_dim = 2
hidden_dim = 2
output_dim = 1

W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

# -----------------------------
# 4. Training Loop
# -----------------------------
lr = 0.1
epochs = 10000

for epoch in range(epochs):
    # ---- Forward Pass ----
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # final output (prediction)

    # ---- Compute Loss (MSE) ----
    loss = np.mean((y - a2)**2)

    # ---- Backpropagation ----
    # Output layer error
    d_a2 = a2 - y                     # derivative of loss w.r.t. output
    d_z2 = d_a2 * sigmoid_derivative(a2)

    # Gradients for W2 and b2
    dW2 = np.dot(a1.T, d_z2)
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    # Hidden layer error
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)

    # Gradients for W1 and b1
    dW1 = np.dot(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # ---- Update Weights ----
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -----------------------------
# 5. Test Predictions
# -----------------------------
print("\nPredictions after training:")
print(a2)
 