import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import copy
import csv
import glob
import shutil
import yaml
import numpy as np
import cv2
import wandb

# Import our custom modules
from data_loader import get_dataloader, get_all_patient_ids, get_patient_split
from model_sam import build_sam3_model
from model_traditional import build_traditional_model
import config

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Logits [B, C, H, W] or [B, C]
        targets: Targets [B, H, W] or [B]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def calculate_semantic_miou(outputs, targets):
    """
    Calculates the Intersection and Union for Semantic Segmentation mIoU 
    from Mask R-CNN instance outputs.
    
    Args:
        outputs (list): List of dicts from Mask R-CNN inference (boxes, labels, masks, scores).
        targets (list): List of dicts containing ground truth (boxes, labels, masks).
        
    Returns:
        intersection (float): Sum of intersection pixels.
        union (float): Sum of union pixels.
    """
    total_intersection = 0
    total_union = 0
    
    # Iterate over the batch
    for output, target in zip(outputs, targets):
        # --- 1. Construct Ground Truth Semantic Mask ---
        # Target masks are typically [N, H, W]. We merge them into one binary mask [H, W]
        # (Assuming binary segmentation for simplicity, or multi-class flattened)
        gt_masks = target['masks']
        if gt_masks.shape[0] > 0:
            # Max across instances to get a single map where >0 is foreground
            # If you have multiple classes, you might need specific class logic here.
            # Here we assume a simple "Foreground vs Background" evaluation.
            gt_semantic = (gt_masks.sum(dim=0) > 0).float()
        else:
            # Handle empty ground truth (all background)
            _, h, w = gt_masks.shape if gt_masks.dim() == 3 else (0, *gt_masks.shape[-2:])
            if h == 0: # handling edge case if shape is weird
                 # Try to get shape from output or target context if possible, 
                 # otherwise rely on the prediction shape
                 h, w = output['masks'].shape[-2:]
            gt_semantic = torch.zeros((h, w), device=gt_masks.device)

        # --- 2. Construct Predicted Semantic Mask ---
        # Outputs are usually Softmax or Logits. Mask R-CNN gives probabilities [N, 1, H, W]
        pred_masks = output['masks']
        scores = output['scores']
        
        # Filter by confidence threshold (optional but recommended)
        score_threshold = 0.5
        mask_threshold = 0.5 # Pixel probability threshold
        
        valid_indices = scores > score_threshold
        valid_masks = pred_masks[valid_indices]
        
        if valid_masks.shape[0] > 0:
            # valid_masks is [N, 1, H, W]. Squeeze to [N, H, W]
            valid_masks = valid_masks.squeeze(1)
            # Threshold pixels
            valid_masks = (valid_masks > mask_threshold).float()
            # Combine all instances into one semantic map
            pred_semantic = (valid_masks.sum(dim=0) > 0).float()
        else:
            pred_semantic = torch.zeros_like(gt_semantic)

        # --- 3. Calculate IoU ---
        # Ensure shapes match
        if pred_semantic.shape != gt_semantic.shape:
             # Resize pred to match GT if necessary (though Mask R-CNN usually preserves size)
             pass

        intersection = (pred_semantic * gt_semantic).sum().item()
        union = (pred_semantic + gt_semantic).gt(0).sum().item() # .gt(0) handles overlaps correctly
        
        total_intersection += intersection
        total_union += union
        
    return total_intersection, total_union


def prepare_mask_rcnn_targets(masks, device):
    """
    Converts segmentation masks into Mask R-CNN compatible targets 
    (Boxes, Labels, Masks).
    """
    b, n_patches, h, w = masks.shape
    masks_flat = masks.view(-1, h, w)
    targets = []

    for i in range(masks_flat.shape[0]):
        mask_np = masks_flat[i].numpy() 
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[obj_ids > 0] # Remove background
        
        boxes = []
        labels = []
        inst_masks = []
        
        for class_id in obj_ids:
            binary_mask = (mask_np == class_id).astype(np.uint8)
            pos = np.where(binary_mask)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)
                inst_masks.append(binary_mask)
        
        if len(boxes) > 0:
            target_d = {
                'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
                'labels': torch.tensor(labels, dtype=torch.int64).to(device),
                'masks': torch.tensor(np.array(inst_masks), dtype=torch.uint8).to(device)
            }
        else:
            # Handle empty patches (background only)
            target_d = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32).to(device),
                'labels': torch.zeros((0,), dtype=torch.int64).to(device),
                'masks': torch.zeros((0, h, w), dtype=torch.uint8).to(device)
            }
        targets.append(target_d)
        
    return targets

def validate_resnet(model, val_loader, epoch, total_epochs, device):
    """
    Runs validation on the dataset. 
    Calculates both Loss (requires Train mode) and mIoU (requires Eval mode).
    """
    val_loss_accum = 0.0
    total_intersection = 0
    total_union = 0
    
    # We don't use torch.no_grad() globally here because we need to 
    # run a forward pass in train mode to get the loss dict.
    # However, we wrap the inference part in no_grad.
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val]")
    
    for patches, masks in pbar:
        b, n_patches, c, h, w = patches.shape
        images = patches.view(-1, c, h, w).to(device)
        
        # Use helper function to generate targets
        targets = prepare_mask_rcnn_targets(masks, device)
        
        # --- 1. Validation Loss ---
        # Torchvision Mask R-CNN only calculates loss when in .train() mode
        with torch.no_grad(): # Disable gradient calc, but keep mode=Train
            model.train() 
            loss_dict = model(list(images), targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss_accum += losses.item()

        # --- 2. Validation mIoU ---
        # Switch to eval mode for inference
        model.eval()
        with torch.no_grad():
            outputs = model(list(images))
            batch_inter, batch_union = calculate_semantic_miou(outputs, targets)
            total_intersection += batch_inter
            total_union += batch_union

    avg_val_loss = val_loss_accum / len(val_loader) if len(val_loader) > 0 else float('inf')
    avg_val_miou = total_intersection / total_union if total_union > 0 else 0.0
    
    return avg_val_loss, avg_val_miou

def train_resnet(train_loader, val_loader):
    """Main Training Loop."""
    print("Starting Mask R-CNN Training & Validation...")
    
    # --- Initialize WandB ---
    wandb.init(
        project="medical-resnet",
        name = "resnet_run",
        config={
            "net": "ResNet50",
            "encoder": "resnet50",
            "learning_rate": 0.005,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
        }
    )

    model = build_traditional_model("resnet").to(config.DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    best_miou = 0.0 
    csv_filename = os.path.join(config.ROOT_DIR, "resnet_progress.csv")
    csv_file = open(csv_filename, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_miou"])
    print(f"Logging progress to {csv_filename}")

    try:
        for epoch in range(config.EPOCHS):
            epoch_idx = epoch + 1
            
            # --- Training ---
            model.train()
            train_loss_accum = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx}/{config.EPOCHS} [Train]")
            
            for patches, masks in pbar:
                b, n_patches, c, h, w = patches.shape
                images = patches.view(-1, c, h, w).to(config.DEVICE)
                
                # Use helper function
                targets = prepare_mask_rcnn_targets(masks, config.DEVICE)

                loss_dict = model(list(images), targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                loss_value = losses.item()
                train_loss_accum += loss_value
                pbar.set_postfix(loss=loss_value)
                wandb.log({"train/batch_loss": loss_value})
            
            avg_train_loss = train_loss_accum / len(train_loader) if len(train_loader) > 0 else 0.0

            # --- Validation (Using split function) ---
            avg_val_loss, avg_val_miou = validate_resnet(
                model, val_loader, epoch_idx, config.EPOCHS, config.DEVICE
            )
            
            print(f"Epoch {epoch_idx} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f}")

            # --- Logging ---
            wandb.log({
                "epoch": epoch_idx,
                "train/total_loss": avg_train_loss,
                "val/total_loss": avg_val_loss,
                "val/miou": avg_val_miou
            })

            writer.writerow([epoch_idx, avg_train_loss, avg_val_loss, avg_val_miou])
            csv_file.flush()

            # --- Save Best Model ---
            if avg_val_miou > best_miou:
                best_miou = avg_val_miou
                save_path = os.path.join(config.ROOT_DIR, "resnet_best_model.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  --> New Best Model (mIoU: {best_miou:.4f}) Saved to {save_path}!")
                
    finally:
        csv_file.close()
        wandb.finish()


def convert_npy_mask_to_yolo_txt(mask_path, label_path):
    """
    Converts a single .npy mask file to YOLO .txt format.
    YOLO format: class_id x1 y1 x2 y2 ... (normalized 0-1)
    
    YOLO requires a contour shape around objects in order to train 
    this is why we need to manipulate the data beforehand.
    
    NOTE: This is a potential spot for errors between models and should be noted in the study.
    """
    try:
        mask = np.load(mask_path)
        height, width = mask.shape
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        # Collect lines first to check if file is empty
        lines = []
        classes = np.unique(mask)
        for cls_id in classes:
            if cls_id == 0: continue

            # Create binary mask for this class
            binary_mask = (mask == cls_id).astype(np.uint8)
                
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 5: # Noise filter
                    continue

                cnt = cnt.flatten()
                normalized_cnt = []
                for i in range(0, len(cnt), 2):
                    normalized_cnt.append(cnt[i] / width)
                    normalized_cnt.append(cnt[i+1] / height)
                
                # Verify validity (at least 3 points)
                if len(normalized_cnt) >= 6: 
                    line = f"{int(cls_id)} " + " ".join([f"{x:.6f}" for x in normalized_cnt])
                    lines.append(line)
        with open(label_path, 'w') as f:
            # Open and write at the end of iterating
            f.write('\n'.join(lines))

        return len(lines) > 0
            
    except Exception as e:
        print(f"Error converting {mask_path}: {e}")

def prepare_yolo_data(root_dir):
    """
    Will create a YOLO-compliatn directory structure via Symlinks.
    This will allow the YOLO model to accurately find the images and corresponding masks.
    Structure:
    config.DATA_ROOT/yolo_ready/
        images/
            train/
            val/
        labels/
            train/
            val/
    1. Converts .npy masks to YOLO .txt labels.
    2. Generates train.txt and val.txt.
    3. Updates yolo_config.yaml.
    """
    print("Preparing YOLO data splits & converting masks...")
    
    # Define new root for YOLO data
    yolo_root = os.path.join(config.DATA_ROOT, "yolo_ready")

    # Directories for images and labels
    img_train_dir = os.path.join(yolo_root, "images", "train")
    img_val_dir = os.path.join(yolo_root, "images", "val")
    lbl_train_dir = os.path.join(yolo_root, "labels", "train")
    lbl_val_dir = os.path.join(yolo_root, "labels", "val")
    
    # Create directories (and clear previous run to ensure clean state)
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    all_ids = get_all_patient_ids(root_dir, is_video=False)
    if not all_ids:
        print(f"WARNING: No patient IDs found in {root_dir}.")
        return

    train_ids, val_ids = get_patient_split(all_ids, config.TRAIN_RATIO)
    
    def process_subset(patient_ids, img_dest, lbl_dest, desc):
        count = 0
        for pid in tqdm(patient_ids, desc=desc):
            # Find all images for this patient
            # Note: We look in the ORIGINAL structure
            p_img_files = glob.glob(os.path.join(root_dir, "**", pid, "*.jpg"), recursive=True)
            
            for img_path_orig in p_img_files:
                parts = img_path_orig.split(os.sep)
                if "Test" in parts: continue # Skip Test folder if present in Training split

                # Calculate Mask Path
                mask_path_orig = img_path_orig.replace(f"{os.sep}image{os.sep}", f"{os.sep}mask{os.sep}").replace(".jpg", ".npy")
                
                if not os.path.exists(mask_path_orig):
                    continue

                # --- Create Standardized Filename (Flattened) ---
                # e.g., img0001_0.jpg
                base_name = os.path.basename(img_path_orig)
                # We prefix with patient ID to avoid collision if base names are just "0.jpg", "1.jpg"
                new_filename = f"{pid}_{base_name}" 
                
                # --- 1. Symlink Image ---
                new_img_path = os.path.join(img_dest, new_filename)
                if not os.path.exists(new_img_path):
                    os.symlink(os.path.abspath(img_path_orig), new_img_path)
                
                # --- 2. Generate Label ---
                new_txt_filename = new_filename.replace(".jpg", ".txt")
                new_lbl_path = os.path.join(lbl_dest, new_txt_filename)
                
                convert_npy_mask_to_yolo_txt(mask_path_orig, new_lbl_path)
                count += 1
        return count

    print("Processing Training Split...")
    n_train = process_subset(train_ids, img_train_dir, lbl_train_dir, "Building Train Set")
    
    print("Processing Validation Split...")
    n_val = process_subset(val_ids, img_val_dir, lbl_val_dir, "Building Val Set")
    
    print(f"YOLO Dataset Ready: {n_train} training, {n_val} validation pairs.")

    # Update YAML to point to the directory, NOT the txt file list.
    # YOLO auto-detects images in the directory.
    yolo_conf = {
        'path': yolo_root,
        'train': 'images/train',
        'val': 'images/val',
        'names': config.LABEL_MAP
    }
    
    with open(config.YOLO_YAML_PATH, 'w') as f:
        yaml.dump(yolo_conf, f, sort_keys=False)
    
    print(f"Updated {config.YOLO_YAML_PATH}")


def train_yolo():
    print("Starting YOLOv11 Training...")
    
    # Ensure splits are generated before training
    root_dir = os.path.join(config.DATA_ROOT, "BTCV_Image")
    prepare_yolo_data(root_dir)
    
    # Ultralytics automatically logs results to config.ROOT_DIR/yolo_medical_run
    model_wrapper = build_traditional_model("yolo")
    model_wrapper.train(epochs=config.EPOCHS) 
    print("YOLO Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "yolo"], help="Model to train")
    args = parser.parse_args()
    
    # Ensure Model Directory Exists
    os.makedirs(config.ROOT_DIR, exist_ok=True)
    
    if args.model == "yolo":
        train_yolo()
    else:
        # 1. Determine Root Directory
        root_dir = os.path.join(config.DATA_ROOT, "BTCV_Image")
        is_video = False
        
        # 2. Get All Patient IDs
        all_ids = get_all_patient_ids(root_dir, is_video=is_video)
        print(f"Found {len(all_ids)} patients: {all_ids[:5]}...")
        
        # 3. Split IDs
        train_ids, val_ids = get_patient_split(all_ids, config.TRAIN_RATIO)
        print(f"Split: {len(train_ids)} Train, {len(val_ids)} Val")
        
        # 4. Create DataLoaders with filtered lists
        train_loader = get_dataloader(root_dir, args.model, patient_ids=train_ids)
        val_loader = get_dataloader(root_dir, args.model, patient_ids=val_ids)
        
        train_resnet(train_loader, val_loader)
