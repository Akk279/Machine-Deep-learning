import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

# -------------------------
# Build a Simple Model
# -------------------------
model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# -------------------------
# 1. Print Model Summary
# -------------------------
print("\n📌 Model Summary:\n")
model.summary()

# -------------------------
# 2. Visualize Model Architecture
# -------------------------
# This creates an image "model.png"
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
print("\n✅ Model diagram saved as 'model.png'")

# -------------------------
# 3. (Optional) Visualize Weights of First Layer
# -------------------------
weights, biases = model.layers[0].get_weights()
print("\nFirst layer weights shape:", weights.shape)  # (784, 128)
print("First layer biases shape:", biases.shape)
