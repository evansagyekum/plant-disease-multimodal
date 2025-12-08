import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class RealMultimodalDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        root_dir: path to 'data/processed/tomato_small'
        """
        self.root_dir = root_dir
        
        # --- FIX: Filter out .DS_Store and other non-folders ---
        all_items = os.listdir(root_dir)
        self.classes = sorted([d for d in all_items if os.path.isdir(os.path.join(root_dir, d))])
        
        self.files = []
        
        # Collect all files
        for label_idx, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls_name)
            
            # Safety check: ensure cls_folder is actually a directory
            if not os.path.isdir(cls_folder):
                continue
                
            for img_name in os.listdir(cls_folder):
                # Only process valid image extensions
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.files.append((os.path.join(cls_folder, img_name), label_idx))
        
        # Simple Image Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet standards
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def _simulate_spectrum(self, label_idx):
        """
        Simulate realistic vegetation spectra.
        Label 0 (first in sorted list) usually Bacterial Spot
        Label 1 (second in sorted list) usually Healthy
        """
        bands = np.linspace(400, 1000, 200) # 400nm to 1000nm
        spectrum = np.zeros_like(bands)
        
        # Base Curve (Vegetation signature)
        # Visible (400-700): Green peak ~550nm
        spectrum += 0.1 * np.exp(-0.5 * ((bands - 550)/50)**2) 
        
        # NIR Plateau (700-1000): The "Red Edge"
        # sigmoid function to jump at 700nm
        nir_edge = 1 / (1 + np.exp(-(bands - 720)/20))
        
        # Logic: We assume 'Tomato___healthy' is the second folder alphabetically (index 1)
        # If your folders are different, this logic might flip, but for MVP it's fine.
        is_healthy = (label_idx == 1) 
        
        if is_healthy: # Healthy
            # Strong Red Edge, High NIR reflectance
            spectrum += 0.6 * nir_edge
            spectrum += np.random.normal(0, 0.02, 200) # Sensor noise
        else: # Diseased
            # Lower NIR (Cell structure damage), Higher Red (Chlorosis)
            spectrum += 0.3 * nir_edge 
            spectrum += 0.1 * np.exp(-0.5 * ((bands - 650)/50)**2) # Red/Yellow shift
            spectrum += np.random.normal(0, 0.02, 200)

        return torch.tensor(spectrum, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        
        # 1. Load Real Image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 2. Generate Bio-Simulated Spectrum
        spectrum = self._simulate_spectrum(label)
        
        return image, spectrum, torch.tensor(label, dtype=torch.long)