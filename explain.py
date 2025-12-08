import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.fusion_net import FusionModel
from src.dataset_real import RealMultimodalDataset

# --- CONFIGURATION ---
DEVICE = torch.device("cpu") # CPU is better for plotting
MODEL_PATH = "experiments/fusion_model_real.pth"
DATA_PATH = "data/processed/tomato_small"

# Class Names (Alphabetical order from the folders)
CLASS_NAMES = ["Bacterial Spot", "Healthy"]

class ExplainableAI:
    def __init__(self, model):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        
        # Hook into ResNet50's last convolutional layer
        target_layer = list(self.model.image_encoder.children())[-2][-1].conv3
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_tensor, spectral_tensor):
        # Forward Pass
        output = self.model(image_tensor.unsqueeze(0), spectral_tensor.unsqueeze(0))
        # Get the highest confidence class
        pred_idx = output.argmax(dim=1).item()
        
        # Backward Pass
        self.model.zero_grad()
        score = output[:, pred_idx]
        score.backward()
        
        # Generate CAM
        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, pred_idx

    def get_spectral_importance(self, spectral_tensor):
        spectral_tensor.requires_grad = True
        output = self.model(torch.randn(1, 3, 224, 224), spectral_tensor.unsqueeze(0))
        pred_idx = output.argmax(dim=1).item()
        output[:, pred_idx].backward()
        return spectral_tensor.grad.abs().numpy()

def denormalize(tensor):
    """Reverses ImageNet normalization so the leaf looks real."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize():
    print(f"🔍 Loading Real Model from {MODEL_PATH}...")
    
    # Load Model (2 Classes)
    model = FusionModel(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    # Load Data
    ds = RealMultimodalDataset(root_dir=DATA_PATH)
    
    # Pick a specific sample (Change index to see different leaves)
    # Index 0 is usually Bacterial Spot, Index -1 is usually Healthy
    sample_idx = 0 
    img, spec, label = ds[sample_idx]
    
    true_label = CLASS_NAMES[label.item()]
    print(f"📸 Analyzing Sample {sample_idx}: True Label = {true_label}")
    
    # Run Explanation
    explainer = ExplainableAI(model)
    heatmap, pred_idx = explainer.generate_heatmap(img, spec)
    spec_imp = explainer.get_spectral_importance(spec.clone())
    
    pred_label = CLASS_NAMES[pred_idx]
    
    # --- PLOTTING ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original Image
    real_img = denormalize(img)
    axs[0].imshow(real_img)
    axs[0].set_title(f"True: {true_label}")
    axs[0].axis('off')
    
    # 2. Heatmap Overlay
    import PIL.Image
    cam_img = PIL.Image.fromarray(np.uint8(255 * heatmap))
    cam_img = cam_img.resize((224, 224), resample=PIL.Image.BILINEAR)
    
    axs[1].imshow(real_img)
    axs[1].imshow(cam_img, cmap='jet', alpha=0.5)
    axs[1].set_title(f"Model Attention\nPred: {pred_label}")
    axs[1].axis('off')
    
    # 3. Spectral Analysis
    bands = np.linspace(400, 1000, 200)
    axs[2].plot(bands, spec.numpy(), label='Reflectance', color='green', linewidth=2)
    axs[2].fill_between(bands, 0, spec_imp * 10, color='red', alpha=0.3, label='Model Focus') # Scale imp for visibility
    axs[2].set_xlabel("Wavelength (nm)")
    axs[2].set_ylabel("Reflectance")
    axs[2].set_title("Spectral Signature Analysis")
    axs[2].legend()
    
    plt.tight_layout()
    output_file = "experiments/real_data_report.png"
    plt.savefig(output_file)
    print(f"✅ Report saved to: {output_file}")

if __name__ == "__main__":
    visualize()