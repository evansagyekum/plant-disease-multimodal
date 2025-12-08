import os
import urllib.request
import zipfile
import shutil

# URL to a clean subset of PlantVillage (Tomato)
DATA_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
TARGET_DIR = "data/raw"

def download_and_extract():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    print("⬇️  Downloading PlantVillage dataset (this might take a minute)...")
    zip_path = os.path.join(TARGET_DIR, "plantvillage.zip")
    # We will use a smaller direct link if possible, but for now we pull the repo 
    # and filter it to keep it simple for you.
    try:
        urllib.request.urlretrieve(DATA_URL, zip_path)
    except Exception as e:
        print(f"Error downloading: {e}")
        return
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    # Organize: We only want Tomato_Bacterial_spot and Tomato_healthy
    base_folder = os.path.join(TARGET_DIR, "PlantVillage-Dataset-master", "raw", "color")
    classes_to_keep = ["Tomato___Bacterial_spot", "Tomato___healthy"]
    final_data_path = "data/processed/tomato_small"
    if os.path.exists(final_data_path):
        shutil.rmtree(final_data_path)
    os.makedirs(final_data_path)
    print("📂 Filtering for Tomato plants...")
    for cls in classes_to_keep:
        src = os.path.join(base_folder, cls)
        dst = os.path.join(final_data_path, cls)
        if os.path.exists(src):
            shutil.copytree(src, dst)
            print(f"   - Copied {cls}")
        else:
            print(f"   ⚠️ Could not find {cls} inside zip.")
    # Cleanup
    os.remove(zip_path)
    shutil.rmtree(os.path.join(TARGET_DIR, "PlantVillage-Dataset-master"))
    print(f"✅ Data ready at: {final_data_path}")

if __name__ == "__main__":
    download_and_extract()
