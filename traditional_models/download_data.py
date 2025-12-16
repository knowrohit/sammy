import os
import yaml
import requests
import zipfile
import shutil
from huggingface_hub import snapshot_download
import config

def create_yolo_config(data_path):
    """Creates the YOLOv11 yaml config file."""
    # Ensure the model directory exists
    os.makedirs(config.ROOT_DIR, exist_ok=True)
    
    yolo_settings = {
        'path': data_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': config.LABEL_MAP
    }
    
    with open(config.YOLO_YAML_PATH, 'w') as f:
        yaml.dump(yolo_settings, f, sort_keys=False)
    print(f"YOLOv11 configuration saved to {config.YOLO_YAML_PATH}")

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzip complete.")

def reorganize_extracted_files(extract_root, zip_filename):
    """
    Moves files from a single subfolder (if created by zip) to root.
    """
    # Get all items in directory excluding the zip file itself
    items = [i for i in os.listdir(extract_root) if i != zip_filename]
    
    # If there is only one item and it's a directory, chances are it's a wrapper folder
    if len(items) == 1:
        subdir_path = os.path.join(extract_root, items[0])
        if os.path.isdir(subdir_path):
            print(f"Reorganizing: Moving contents from {items[0]} to {extract_root}...")
            for sub_item in os.listdir(subdir_path):
                src = os.path.join(subdir_path, sub_item)
                dst = os.path.join(extract_root, sub_item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            os.rmdir(subdir_path)

def download_data():
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    
    print("Downloading Voxel51/BTCV-CT-as-video-MedSAM2-dataset...")
    try:
        snapshot_download(
            repo_id="Voxel51/BTCV-CT-as-video-MedSAM2-dataset",
            repo_type="dataset",
            local_dir=os.path.join(config.DATA_ROOT, "BTCV_Video"),
            allow_patterns=["*.mp4", "*.json", "*.png", "*.yml", "*.md"]
        )
        print("Video dataset downloaded.")
    except Exception as e:
        print(f"Error downloading Video dataset: {e}")

    # Image Dataset (jiayuanz3/btcv)
    print("Checking for jiayuanz3/btcv (Image Segmentation) dataset...")
    btcv_image_root = os.path.join(config.DATA_ROOT, "BTCV_Image")
    os.makedirs(btcv_image_root, exist_ok=True)
    
    zip_filename = "btcv.zip"
    zip_path = os.path.join(btcv_image_root, zip_filename)
    
    # Check if we have extracted content already
    # We scan for specific folders that indicate successful extraction (e.g., 'image' or 'Training')
    # Scanning recursively for 'img' folders is a good heuristic based on the dataset structure
    is_extracted = False
    for root, dirs, files in os.walk(btcv_image_root):
        if "image" in dirs or "Training" in dirs:
            is_extracted = True
            break
            
    if is_extracted:
        print(f"✅ Dataset appears to be already extracted in {btcv_image_root}.")
    else:
        # If not extracted, check for the zip file
        if os.path.exists(zip_path):
            print(f"Found {zip_filename}. Proceeding to unzip...")
            try:
                unzip_file(zip_path, btcv_image_root)
                reorganize_extracted_files(btcv_image_root, zip_filename)
                print("Unzip and reorganization complete.")
            except Exception as e:
                print(f"Error during unzipping: {e}")
        else:
            # Zip file missing
            print(f"Error: {zip_filename} not found in {btcv_image_root}")
            print("Please run the following command to download the dataset manually:")
            print(f"\n    cd {btcv_image_root} && wget https://huggingface.co/datasets/jiayuanz3/btcv/resolve/main/btcv.zip\n")
            # We exit here because create_yolo_config won't be useful without data
            return

    # Create YOLO Config
    create_yolo_config(os.path.join(config.DATA_ROOT, "BTCV_Image"))

if __name__ == "__main__":
    download_data()
