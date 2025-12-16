import torch
import os

# User and Path Configuration
USER_ID = "dr3432"
ROOT_DIR = f"/scratch/{USER_ID}/deepLearning/models"
DATA_ROOT = f"/scratch/{USER_ID}/deepLearning/data"
YOLO_YAML_PATH = os.path.join(ROOT_DIR, "yolo_config.yaml")

# Label Mapping
LABEL_MAP = {
    0: "background", 1: "spleen", 2: "right_kidney", 3: "left_kidney",
    4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach",
    8: "aorta", 9: "inferior_vena_cava", 10: "portal_vein_and_splenic_vein",
    11: "pancreas", 12: "right_adrenal_gland", 13: "left_adrenal_gland"
}
NUM_CLASSES = len(LABEL_MAP)

# Model Input Sizes
SAM_INPUT_SIZE = 1008
RESNET_INPUT_SIZE = 224
IMG_ORIG_SIZE = 359

# Training Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
EPOCHS = 100
TRAIN_RATIO = 0.8
BATCH_SIZE = 1
