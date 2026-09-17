import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleFCN, self).__init__()
        
        # Encoder (feature extractor)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # (B, 3, H, W) -> (B, 64, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # Downsample -> (B, 64, H/2, W/2)
            
            nn.Conv2d(64, 128, 3, padding=1),# -> (B, 128, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> (B, 128, H/4, W/4)
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 2, stride=2) # (B, C, H, W)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # shape: (B, num_classes, H, W)

# Test
model = SimpleFCN(num_classes=3)
x = torch.randn(1, 3, 128, 128)  # Random input image
out = model(x)
print("Output shape:", out.shape)  # (1, 3, 128, 128)
