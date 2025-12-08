import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticMultimodalDataset(Dataset):
    """
    Generates random RGB images and random spectral curves 
    to test the training pipeline mechanics.
    """
    def __init__(self, num_samples=1000, num_bands=200, num_classes=5):
        self.num_samples = num_samples
        self.num_bands = num_bands
        self.num_classes = num_classes
        
        # Simulate labels (0: Healthy, 1: Rust, 2: Scab, etc.)
        self.labels = np.random.randint(0, num_classes, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Fake RGB Image: (Channels, Height, Width)
        # Random noise mimicking a normalized image
        image = torch.randn(3, 224, 224)
        
        # 2. Fake Spectral Data: (Num_Bands)
        # Random float values mimicking reflectance data
        spectrum = torch.randn(self.num_bands)
        
        # 3. Label
        label = self.labels[idx]
        
        return image, spectrum, label
