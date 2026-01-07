# 🌿 Multimodal Plant Disease Detection System (RGB + Hyperspectral)

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-Container-2496ed?style=for-the-badge&logo=docker)
![Azure](https://img.shields.io/badge/Azure-Cloud-0078d4?style=for-the-badge&logo=microsoftazure)
![FastAPI](https://img.shields.io/badge/FastAPI-Live_Server-009688?style=for-the-badge&logo=fastapi)

> **🚀 Live API Demo:** [http://9.141.105.190/docs](http://9.141.105.190/docs)  
> *(Note: Hosted on Azure Linux VM. If the link is down, the VM is paused to save costs.)*

## 📋 Executive Summary
Traditional Computer Vision models often fail to detect plant stress until visible symptoms (yellowing, lesions) appear. By then, the crop yield is already compromised.

This project implements a **Multimodal Deep Learning Architecture** that fuses **Visual data (RGB)** with **Bio-Simulated Hyperspectral data**. By analyzing the "Red Edge" shift (700nm-1000nm) alongside visual texture, this model detects stress indicators with higher confidence than single-modality models.

---

graph LR
    %% Styles
    classDef azure fill:#0078d4,stroke:#fff,stroke-width:2px,color:#fff;
    classDef docker fill:#2496ed,stroke:#fff,stroke-width:2px,color:#fff;
    classDef python fill:#ffd43b,stroke:#333,stroke-width:2px,color:#000;
    classDef user fill:#333,stroke:#fff,stroke-width:2px,color:#fff;

    %% User Layer
    User((👤 Client)) -->|HTTP POST Image| Firewall[("🛡️ Azure NSG <br/> Firewall")]
    class User user

    %% Azure Layer
    subgraph Cloud ["☁️ Azure Resource Group (East US)"]
        direction LR
        Firewall -->|Allow Port 80| VM
        
        subgraph VM ["💻 Linux Virtual Machine (Ubuntu)"]
            
            subgraph Container ["🐳 Docker Container"]
                direction TB
                Port["Port Mapping <br/> (80:8000)"] --> Server["⚡ Uvicorn Server"]
                
                subgraph App ["🐍 FastAPI Application"]
                    Server --> Router["/predict Endpoint"]
                    
                    subgraph Logic ["Data Pipeline"]
                        Router --> Pre["1. Preprocessing"]
                        Pre --> Sim["2. Physics Engine"]
                        Sim --> Inference["3. Model Inference"]
                    end
                end
            end
        end
    end

    %% Return Flow
    Inference -->|JSON Result| User

    %% Apply Styles
    class Cloud,VM azure
    class Container docker
    class App,Logic python


    See Below

    graph TD
    %% Inputs
    InputImg[("📸 Input Image <br/> (RGB)")]
    
    %% Preprocessing
    subgraph Simulation ["🧪 Physics-Based Simulation"]
        Logic1["Vegetation Index Calculation"]
        Logic2["Red Edge Modeling (700nm)"]
        Logic3["Noise Injection"]
    end
    
    InputImg --> Logic1
    Logic1 --> Logic2 --> Logic3
    Logic3 --> InputSpec[("📈 Simulated Spectrum <br/> (200 Bands)")]

    %% Model
    subgraph Model ["🧠 Multimodal Fusion Network"]
        direction TB
        
        subgraph Vision ["👁️ Vision Branch"]
            InputImg --> ResNet["ResNet50 Backbone"]
            ResNet --> FeatImg["Spatial Embeddings <br/> (256 dim)"]
        end
        
        subgraph Spectral ["📊 Spectral Branch"]
            InputSpec --> CNN["1D-CNN (3-Layer)"]
            CNN --> FeatSpec["Spectral Embeddings <br/> (128 dim)"]
        end
        
        FeatImg --> Concatenate((➕ Concatenate))
        FeatSpec --> Concatenate
        Concatenate --> Dense["Fully Connected Layer"]
        Dense --> Output["✅ Prediction <br/> (Healthy vs Disease)"]
    end

Key Technical Components
Vision Branch: ResNet50 (Pretrained on ImageNet) extracts texture and leaf shape features.
Spectral Branch: A custom 3-layer 1D-CNN processes 200 spectral bands to detect biochemical changes (chlorophyll absorption).
Physics Simulation: A custom data loader (src/dataset_real.py) simulates realistic vegetation reflectance curves based on biological principles (NDVI and Red Edge shifts).
🔍 Explainability (XAI)
To ensure the model is trustworthy, I implemented Grad-CAM and Spectral Saliency Analysis:
Visual Attention (Grad-CAM)	Spectral Importance (Saliency)
The model focuses on the leaf lesion, ignoring the background.	The model prioritizes the 700nm-750nm (Red Edge) range, proving it learned physical stress signals.
(See experiments/real_data_report.png in repo)	(See experiments/real_data_report.png in repo)
🛠️ Deployment & MLOps
This project is not just a notebook; it is a full production pipeline.
1. Containerization (Docker)
The application is wrapped in a Docker container using python:3.9-slim.
Challenge: Developing on Mac M1 (ARM64) vs Deploying on Azure (AMD64).
Solution: Built the Docker image directly on the Linux host to ensure architecture compatibility.
2. Cloud Infrastructure (Azure)
Compute: Azure Virtual Machine (Ubuntu 20.04).
Networking: Configured NSG (Network Security Groups) to allow traffic on Port 80.
Server: uvicorn running behind a Docker proxy.
🚀 How to Run Locally
Prerequisites
Python 3.8+
Docker (Optional)
1. Installation
code
Bash
git clone https://github.com/mostkwadwo/plant-disease-multimodal.git
cd plant-disease-multimodal
pip install -r requirements.txt
2. Training
Download the dataset and train the fusion model:
code
Bash
# 1. Download PlantVillage subset
python download_data.py

# 2. Train the model (ResNet + 1D-CNN)
python train.py
The trained model will be saved to experiments/fusion_model_real.pth.
3. Run the API
code
Bash
uvicorn src.app:app --reload
Visit http://localhost:8000/docs to test the API.
☁️ How to Deploy (Docker)
If you want to run this in a container:
code
Bash
# 1. Build
docker build -t plant-api .

# 2. Run
docker run -p 80:8000 plant-api uvicorn src.app:app --host 0.0.0.0 --port 8000
📊 Results
Training Accuracy: ~99% (Binary Classification: Healthy vs Bacterial Spot)
Inference Latency: <200ms on CPU
Model Size: ~100MB
🔮 Future Improvements
Transformer Integration: Replace ResNet with a Vision Transformer (ViT) for global attention mechanisms.
Edge Deployment: Quantize the model (INT8) for deployment on NVIDIA Jetson devices for field usage.

Author: Evans Agyekum
Tech Stack: PyTorch, FastAPI, Azure, Docker, Git , Limux VM
