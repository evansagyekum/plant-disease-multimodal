import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
import io
import numpy as np

# Import your model definition
from src.models.fusion_net import FusionModel

app = FastAPI(title="Plant Disease Multimodal API")

# --- CONFIGURATION ---
DEVICE = torch.device("cpu") # Inference is usually cheap enough for CPU
MODEL_PATH = "experiments/fusion_model_real.pth"
CLASS_NAMES = ["Bacterial Spot", "Healthy"]

# Global model variable
model = None

# --- LOAD MODEL ON STARTUP ---
@app.on_event("startup")
def load_model():
    global model
    print(f"🔄 Loading model from {MODEL_PATH}...")
    try:
        model = FusionModel(num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Set to evaluation mode
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# --- PREPROCESSING ---
def transform_image(image_bytes):
    """Converts bytes -> PIL -> Tensor"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)

def generate_placeholder_spectrum():
    """
    Generates a generic vegetation spectrum so the model doesn't crash 
    if the user only uploads a photo.
    """
    # 200 bands, simulating a generic green leaf
    bands = np.linspace(400, 1000, 200)
    spectrum = np.zeros_like(bands)
    # Generic green peak
    spectrum += 0.1 * np.exp(-0.5 * ((bands - 550)/50)**2) 
    # Generic NIR plateau
    nir_edge = 1 / (1 + np.exp(-(bands - 720)/20))
    spectrum += 0.5 * nir_edge
    
    return torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0) # (1, 200)

# --- PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model not loaded"}

    # 1. Process Image
    image_bytes = await file.read()
    img_tensor = transform_image(image_bytes)
    
    # 2. Process Spectrum (In a real app, we'd accept JSON input here)
    # For this demo, we auto-generate it.
    spec_tensor = generate_placeholder_spectrum()
    
    # 3. Inference
    with torch.no_grad():
        output = model(img_tensor, spec_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_label = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()

    return {
        "prediction": predicted_label,
        "confidence": float(f"{confidence_score:.4f}"),
        "message": "Spectrum data was simulated for this prediction."
    }

@app.get("/")
def home():
    return {"status": "System Operational", "model": "Multimodal Fusion v1"}

