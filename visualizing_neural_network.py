# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.utils import plot_model

# # -------------------------
# # Build a Simple Model
# # -------------------------
# model = models.Sequential([
#     layers.Dense(128, activation="relu", input_shape=(784,)),
#     layers.Dense(64, activation="relu"),
#     layers.Dense(10, activation="softmax")
# ])

# # Compile the model
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# # -------------------------
# # 1. Print Model Summary
# # -------------------------
# print("\n📌 Model Summary:\n")
# model.summary()

# # -------------------------
# # 2. Visualize Model Architecture
# # -------------------------
# plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
# print("\n✅ Model diagram saved as 'model.png'")

# # -------------------------
# # 3. (Optional) Use model_to_dot (requires Graphviz)
# # -------------------------
# try:
#     dot_img = tf.keras.utils.model_to_dot(model, show_shapes=True, show_layer_names=True)
#     dot_img.write_png("model_dot.png")
#     print("\n✅ Model diagram saved as 'model_dot.png'")
# except Exception as e:
#     print("\n⚠ model_to_dot failed (Graphviz not installed properly):", e)

# # -------------------------
# # 4. Visualize Weights of First Layer
# # -------------------------
# weights, biases = model.layers[0].get_weights()
# print("\nFirst layer weights shape:", weights.shape)  # (784, 128)
# print("First layer biases shape:", biases.shape)

import matplotlib.pyplot as plt

def draw_neural_network(layer_sizes):
    """
    Draws a simple feedforward neural network using matplotlib.
    layer_sizes: list of integers, e.g. [784, 128, 64, 10]
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # X positions for layers
    n_layers = len(layer_sizes)
    x_spacing = 2
    y_spacing = 1

    for i, layer_size in enumerate(layer_sizes):
        x = i * x_spacing
        y_top = (max(layer_sizes) - layer_size) / 2 * y_spacing

        # Draw neurons
        for j in range(layer_size):
            y = y_top + j * y_spacing
            circle = plt.Circle((x, y), 0.1, fill=True, color="skyblue", ec="black")
            ax.add_artist(circle)

            # Draw connections from previous layer
            if i > 0:
                prev_layer_size = layer_sizes[i - 1]
                prev_y_top = (max(layer_sizes) - prev_layer_size) / 2 * y_spacing
                for k in range(prev_layer_size):
                    prev_y = prev_y_top + k * y_spacing
                    ax.plot([x - x_spacing, x], [prev_y, y], "k-", lw=0.2)

        # Add layer label
        ax.text(x, max(layer_sizes) * y_spacing + 0.5, f"Layer {i+1}\n({layer_size} neurons)",
                ha="center", fontsize=8, color="blue")

    plt.title("Neural Network Architecture Visualization", fontsize=12)
    plt.show()

# Example: [Input, Hidden1, Hidden2, Output]
draw_neural_network([784, 128, 64, 10])


