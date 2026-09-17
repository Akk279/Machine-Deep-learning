
# -----------------------------------------------
# ConvNets for Classification and Localization
# -----------------------------------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# 1. Define CNN model
# -----------------------------
class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # [3,224,224] -> [16,224,224]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> [16,112,112]

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> [32,56,56]

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> [64,28,28]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # x_min, y_min, x_max, y_max
        )

    def forward(self, x):
        feats = self.features(x)
        class_out = self.classifier(feats)
        bbox_out = self.bbox_regressor(feats)
        return class_out, bbox_out

# -----------------------------
# 2. Load or create image
# -----------------------------
# 🖼️ Replace this with your local image path:
# Use raw string (r"") to avoid \U unicode error
img_path = r"C:\Users\ak902\Downloads\sampledog.jpg"

# Load image
img = Image.open(img_path).convert("RGB")

# Transform for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img_tensor = transform(img).unsqueeze(0)

# -----------------------------
# 3. Forward pass through model
# -----------------------------
model = ConvNet(num_classes=5)
model.eval()

with torch.no_grad():
    class_pred, bbox_pred = model(img_tensor)

print(f"✅ Class prediction shape: {class_pred.shape}")
print(f"✅ BBox prediction shape: {bbox_pred.shape}")

# -----------------------------
# 4. Visualize predictions
# -----------------------------
# Fake softmax and labels (since not trained)
classes = ["Dog", "Cat", "Car", "Person", "Bird"]
pred_class = classes[class_pred.argmax(dim=1).item()]
bbox = bbox_pred[0].tolist()

# Convert normalized bbox to image scale (fake values)
x_min, y_min, x_max, y_max = [abs(int(v * 224) % 224) for v in bbox]

fig, ax = plt.subplots()
ax.imshow(img)
rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.text(x_min, y_min - 10, pred_class, color='white',
         backgroundcolor='red', fontsize=10, fontweight='bold')

plt.title("Predicted Bounding Box & Class")
plt.axis("off")
plt.show()
