import torch
import torch.nn as nn
import torchvision.models as models

class SpatialAttention(nn.Module):
    """
    Forces the network to focus on key pain Action Units (eyes, brows, mouth)
    rather than background skin or hair.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)

class ConvNeXtPainRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super(ConvNeXtPainRegressor, self).__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = models.convnext_tiny(weights=weights)
        
        # Backbone feature extractor: outputs (B, 768, 7, 7)
        self.features = base.features
        self.attention = SpatialAttention(kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Continuous regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Outputs value in [0.0, 1.0]
        )

    def forward(self, x):
        feat = self.features(x)
        attended_feat = self.attention(feat)
        pooled = self.pool(attended_feat)
        score = self.regressor(pooled)
        return score