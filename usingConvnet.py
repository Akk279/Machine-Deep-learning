import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# ---------------------------------------
# 1️⃣ Define Model
# ---------------------------------------
class ResNet_Classifier_Localizer(nn.Module):
    def __init__(self, num_classes=5, backbone='resnet18', pretrained=True):
        super(ResNet_Classifier_Localizer, self).__init__()
        
        # Load ResNet backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_size = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_size = 2048
        else:
            raise ValueError("Unsupported backbone")
        
        # Remove ResNet classifier
        self.backbone.fc = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, num_classes)
        )

        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 4)  # (x, y, w, h)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        class_out = F.softmax(self.classifier(features), dim=1)
        bbox_out = torch.sigmoid(self.bbox_regressor(features))
        return class_out, bbox_out


# ---------------------------------------
# 2️⃣ Load or Create Image
# ---------------------------------------
img_path = r"C:\Users\ak902\Downloads\close-up-portrait-of-dog-sticking-out-tongue-royalty-free-image-1686912464.avif"

# Try AVIF support if installed
try:
    import pillow_avif
except ImportError:
    pass

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_img = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

# ---------------------------------------
# 3️⃣ Model Inference (Random Example)
# ---------------------------------------
model = ResNet_Classifier_Localizer(num_classes=5, backbone='resnet18', pretrained=False)
model.eval()

with torch.no_grad():
    class_pred, bbox_pred = model(input_img)

# Decode predictions
class_names = ["Cat", "Dog", "Car", "Tree", "Person"]
pred_class = torch.argmax(class_pred, dim=1).item()
pred_label = class_names[pred_class]
bbox = bbox_pred[0].cpu().numpy()

# Scale to image size
x, y, w, h = bbox * np.array([224, 224, 224, 224])

# ---------------------------------------
# 4️⃣ Visualization
# ---------------------------------------
img_np = np.array(img.resize((224, 224)))

fig, ax = plt.subplots(1, figsize=(6, 6))
ax.imshow(img_np)

# Draw bounding box
rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.text(x, y - 5, f"{pred_label}", color='white', fontsize=12, backgroundcolor='red')

plt.title("Predicted Bounding Box & Class")
plt.axis("off")
plt.show()
