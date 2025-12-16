import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from ultralytics import YOLO
import config

def get_maskrcnn_model(num_classes=config.NUM_CLASSES):
    """
    Builds Mask R-CNN with ResNet50 backbone.
    Adjusted for classes in config.
    """
    print("Building Mask R-CNN (ResNet50)...")
    # Load pre-trained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

class YOLOWrapper:
    """Wrapper to interact with Ultralytics YOLOv11."""
    def __init__(self, config_path=config.YOLO_YAML_PATH):
        print("Building YOLOv11-Seg...")
        # 'yolo11l-seg.pt' is the nano segmentation model, good starting point
        self.model = YOLO("yolo11l-seg.pt") 
        self.config_path = config_path

    def train(self, epochs=50):
        # Ultralytics handles its own loop
        results = self.model.train(
            data=self.config_path, 
            epochs=epochs, 
            imgsz=config.IMG_ORIG_SIZE, 
            project=config.ROOT_DIR,
            name="medical_yolo")
        
        return results

    def predict(self, image):
        return self.model(image)

def build_traditional_model(model_name="resnet"):
    if model_name == "resnet":
        return get_maskrcnn_model()
    elif model_name == "yolo":
        return YOLOWrapper()
    else:
        raise ValueError("Unknown model name")
