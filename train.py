import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- UPDATED IMPORTS ---
from src.models.fusion_net import FusionModel
from src.dataset_real import RealMultimodalDataset  # <--- Using the Real Dataset

# --- CONFIGURATION ---
# Detect Mac M1/M2 vs CUDA vs CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
SAVE_PATH = "experiments/fusion_model_real.pth" 

def train():
    print(f"🚀 Initializing Training on: {DEVICE}")
    print("-" * 40)

    # --- 1. Load Real Data ---
    print("dataset: Loading Tomato Image data...")
    data_path = "data/processed/tomato_small"
    
    # Safety check
    if not os.path.exists(data_path):
        raise FileNotFoundError("Run 'python download_data.py' first!")

    dataset = RealMultimodalDataset(root_dir=data_path)
    
    # Split Train/Val (80% Train, 20% Validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"dataset: Found {len(dataset)} total images.")
    print(f"dataset: Training on {len(train_dataset)}, Validation on {len(val_dataset)}")

    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- 2. Initialize Model (2 Classes now: Healthy vs Spot) ---
    print("model:   Loading ResNet50 + 1D-CNN (2 Classes)...")
    model = FusionModel(num_classes=2).to(DEVICE) # <--- Changed to 2
    
    # --- 3. Optimizer & Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision (Disable on M1 for stability, enable on NVIDIA)
    use_amp = (DEVICE.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, spectra, labels in loop:
            # Move data to GPU/CPU
            images = images.to(DEVICE)
            spectra = spectra.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images, spectra)
                loss = criterion(outputs, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=loss.item())

    # --- SAVE MODEL ---
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    torch.save(model.state_dict(), SAVE_PATH)
    print("-" * 40)
    print(f"✅ Success! Real-data model saved to {SAVE_PATH}")
    print("Now update 'explain.py' to visualize the difference!")

if __name__ == "__main__":
    train()