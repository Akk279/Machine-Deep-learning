import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras_cv

def use_retinanet_detection():
    """Use pre-trained RetinaNet from KerasCV for object detection"""
    print("🎯 Using RetinaNet with KerasCV (pre-trained on COCO)")

    # Load pre-trained RetinaNet
    model = keras_cv.models.RetinaNet.from_preset("retinanet_resnet50_coco")

    # Create a test image with shapes
    image = create_test_image()

    # Convert image to tensor and preprocess
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0) / 255.0

    # Run inference
    predictions = model.predict(image_tensor, verbose=0)

    # Extract predictions
    boxes = predictions["boxes"][0].numpy()
    scores = predictions["confidence"][0].numpy()
    labels = predictions["classes"][0].numpy()

    # Keep only high-confidence detections
    keep = scores > 0.3
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image.astype("uint8"))
    ax.set_title("RetinaNet Object Detection")
    ax.axis("off")

    # Draw bounding boxes
    for box, score, label in zip(boxes, scores, labels):
        y1, x1, y2, x2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, color="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{int(label)}:{score:.2f}",
                color="yellow", fontsize=10, weight="bold")

    plt.show()
    print("✅ RetinaNet detection demo complete")


def create_test_image():
    """Create a test image with various shapes"""
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Red square
    image[50:150, 50:150] = [255, 0, 0]

    # Green circle
    y, x = np.ogrid[:400, :400]
    circle_mask = (x - 300) ** 2 + (y - 100) ** 2 <= 50 ** 2
    image[circle_mask] = [0, 255, 0]

    # Blue triangle
    for i in range(100):
        for j in range(i + 1):
            if 200 + i < 400 and 200 + j < 400:
                image[200 + i, 200 + j] = [0, 0, 255]

    return image


# Run RetinaNet demo
use_retinanet_detection()
