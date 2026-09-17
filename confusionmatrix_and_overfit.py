import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# -------------------------
# Load Dataset
# -------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and flatten
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# -------------------------
# Simple Model (no regularization → will overfit)
# -------------------------
model = models.Sequential([
    layers.Dense(512, activation="relu", input_shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# -------------------------
# Train Model (many epochs → simulate overfitting)
# -------------------------
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=30,      # intentionally high → overfit
    batch_size=64,
    verbose=1
)

# -------------------------
# Confusion Matrix
# -------------------------
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - MNIST")
plt.show()

# -------------------------
# Training vs Validation Loss Curve
# -------------------------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Overfitting Simulation")
plt.show()
