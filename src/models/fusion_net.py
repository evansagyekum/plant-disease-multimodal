import torch
import torch.nn as nn
import torchvision.models as models

class SpectralEncoder(nn.Module):
    """
    Encodes 1D hyperspectral data (e.g., 200 bands).
    """
    def __init__(self, num_bands=200, embedding_dim=128):
        super(SpectralEncoder, self).__init__()
        # Input: (Batch, 1, Bands)
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global pooling
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        # Expect input: (Batch, Bands) -> unsqueeze to (Batch, 1, Bands)
        x = x.unsqueeze(1)
        features = self.net(x)
        features = features.squeeze(-1) # (Batch, 128)
        return self.fc(features)

class FusionModel(nn.Module):
    def __init__(self, num_classes=5, num_bands=200):
        super(FusionModel, self).__init__()
        # 1. Image Branch (ResNet50)
        # Using default weights (IMAGENET1K_V1) if available, else standard pretrained
        try:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        except:
            resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1]) # Remove fc
        self.img_fc = nn.Linear(2048, 256) # Project to match spectral scale
        # 2. Spectral Branch
        self.spectral_encoder = SpectralEncoder(num_bands=num_bands, embedding_dim=128)
        # 3. Fusion Branch
        # Concatenating 256 (Image) + 128 (Spectral) = 384 input features
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, spectrum):
        # Image Path
        img_feat = self.image_encoder(image)
        img_feat = img_feat.view(img_feat.size(0), -1) # Flatten (Batch, 2048)
        img_feat = self.img_fc(img_feat)               # (Batch, 256)
        # Spectral Path
        spec_feat = self.spectral_encoder(spectrum)    # (Batch, 128)
        # Concatenate
        combined = torch.cat((img_feat, spec_feat), dim=1) # (Batch, 384)
        # Classify
        output = self.fusion_fc(combined)
        return output
