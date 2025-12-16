import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import glob
import json
import random
import config

def get_all_patient_ids(root_dir, is_video=False):
    """
    Scans the directory to find all unique patient IDs (e.g., 'img0001').
    """
    if is_video:
        # For video: data/img0001.mp4
        files = glob.glob(os.path.join(root_dir, "data", "*.mp4"))
        ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    else:
        # For image: search recursively for folders named imgXXXX
        # This handles structures like root/Training/image/img0001 or root/image/img0001
        search_pattern = os.path.join(root_dir, "**", "img*")
        image_folders = glob.glob(search_pattern, recursive=True)
        
        # Filter to ensure we only get the folder names (IDs)
        ids = []
        for f in image_folders:
            if os.path.isdir(f) and "img" in os.path.basename(f):
                ids.append(os.path.basename(f))
    
    return sorted(list(set(ids)))

def get_patient_split(all_ids, train_ratio=config.TRAIN_RATIO):
    """
    Splits patient IDs into train and validation sets.
    """
    # Fix seed for reproducibility of the split
    random.seed(42)
    # Ensure we work on a copy to avoid in-place shuffle issues if called multiple times
    ids_copy = all_ids[:] 
    random.shuffle(ids_copy)
    
    split_idx = int(len(ids_copy) * train_ratio)
    train_ids = ids_copy[:split_idx]
    val_ids = ids_copy[split_idx:]
    
    return train_ids, val_ids

class BTCVClipDataset(Dataset):
    def __init__(self, root_dir, patient_ids=None, clip_length=4, transform=None):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform
        
        # Get list of video files
        all_files = sorted(glob.glob(os.path.join(root_dir, "data", "*.mp4")))
        if patient_ids is not None:
            self.video_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in patient_ids]
        else:
            self.video_files = all_files

        # Pre-calculate metadata for all clips
        # Format: (video_path, start_frame_index, is_start_of_video)
        self.clips = []
        
        print("Indexing video clips...")
        for vid_path in self.video_files:
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Create sequential chunks
            for start_idx in range(0, total_frames, clip_length):
                # If it's the very first clip of a video, we flag it True
                # This tells the model to reset its memory.
                is_start = (start_idx == 0)
                self.clips.append((vid_path, start_idx, is_start))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_path, start_idx, is_start = self.clips[idx]
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        frames = []
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret: break
            
            # --- Insert your Resizing/Padding Logic Here ---
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            old_size = frame.shape[:2]
            target_size = 1008 # Use 1008 per our previous fix
            ratio = float(target_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            frame_resized = cv2.resize(frame, (new_size[1], new_size[0]))
            
            delta_w = target_size - new_size[1]
            delta_h = target_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            frames.append(frame_padded)
            # -----------------------------------------------
            
        cap.release()
        
        # Handle edge case: last clip might be shorter than clip_length
        # We must pad it or the collate_fn will fail if batch_size > 1
        # (Though usually for video SAM, batch_size=1 is standard)
        while len(frames) < self.clip_length:
             frames.append(frames[-1])

        frames_np = np.array(frames)
        # [T, C, H, W]
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        
        # Return the frames and the "Reset Flag"
        return frames_tensor, is_start


class BTCVImageDataset(Dataset):
    """
    Loader for Traditional Models.
    """
    def __init__(self, root_dir, patient_ids=None, model_type="resnet", transform=None):
        self.root_dir = root_dir
        self.model_type = model_type
        self.transform = transform
        
        # Traverse subfolders recursively to find all jpgs
        # Pattern: root_dir/**/imgXXXX/*.jpg
        search_pattern = os.path.join(root_dir, "**", "*.jpg")
        all_imgs = sorted(glob.glob(search_pattern, recursive=True))
        
        if patient_ids is not None:
            # Filter: Keep images only if their parent folder name is in patient_ids
            self.img_files = [
                f for f in all_imgs 
                if os.path.basename(os.path.dirname(f)) in patient_ids
            ]
        else:
            self.img_files = all_imgs

    def __len__(self):
        return len(self.img_files)

    def _patch_image(self, img, mask):
        patches_img = []
        patches_mask = []
        
        coords = [
            (0, 0), 
            (0, config.IMG_ORIG_SIZE - config.RESNET_INPUT_SIZE), 
            (config.IMG_ORIG_SIZE - config.RESNET_INPUT_SIZE, 0), 
            (config.IMG_ORIG_SIZE - config.RESNET_INPUT_SIZE, config.IMG_ORIG_SIZE - config.RESNET_INPUT_SIZE)
        ]
        
        for r, c in coords:
            patch_i = img[r:r+config.RESNET_INPUT_SIZE, c:c+config.RESNET_INPUT_SIZE, :]
            patch_m = mask[r:r+config.RESNET_INPUT_SIZE, c:c+config.RESNET_INPUT_SIZE]
            patches_img.append(patch_i)
            patches_mask.append(patch_m)
            
        return np.array(patches_img), np.array(patches_mask)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Error handling for bad paths
            return torch.zeros(4, 3, config.RESNET_INPUT_SIZE, config.RESNET_INPUT_SIZE), torch.zeros(4, config.RESNET_INPUT_SIZE, config.RESNET_INPUT_SIZE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load Mask
        # Using the same path for the image but replacing the .jpg for .npy
        mask_path = img_path.replace(f"{os.sep}image{os.sep}", f"{os.sep}mask{os.sep}").replace(".jpg", ".npy")

        if os.path.exists(mask_path):
            mask = np.load(mask_path)
        else:
            raise FileNotFoundError(f"Mask not found at {mask_path} for image {img_path}.")

        if self.model_type == "resnet":
            p_img, p_mask = self._patch_image(img, mask)
            p_img_tensor = torch.from_numpy(p_img).permute(0, 3, 1, 2).float() / 255.0
            p_mask_tensor = torch.from_numpy(p_mask).long()
            return p_img_tensor, p_mask_tensor
            
        elif self.model_type == "yolo":
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return img_tensor, torch.from_numpy(mask).long()


def get_dataloader(root_path, model_type, patient_ids=None, batch_size=1):
    if model_type == "sam3":
        ds = BTCVClipDataset(root_path, patient_ids=patient_ids)
        # Video data needs batch_size=1 usually due to varying temporal dim if not fixed
        return DataLoader(ds, batch_size=1, shuffle=True) 
    else:
        ds = BTCVImageDataset(root_path, patient_ids=patient_ids, model_type=model_type)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)
