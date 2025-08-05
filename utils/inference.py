import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

from models.model_defs import TumorClassifier, AttentionUNet
from utils.transforms import classification_transform, segmentation_transform
from utils.preprocess import ensure_rgb

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODELS ===
# Classification Model
clf_model = TumorClassifier(num_classes=4).to(device)
clf_model.load_state_dict(torch.load("models/tumor_classifier_resnet50.pth", map_location=device))
clf_model.eval()

# Segmentation Model
seg_model = AttentionUNet().to(device)
seg_model.load_state_dict(torch.load("models/Attention_unet_best.pth", map_location=device))
seg_model.eval()

# === CLASS LABELS ===
label_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# === CLASSIFICATION FUNCTION ===
def classify_tumor(image: Image.Image) -> str:
    image = ensure_rgb(image)
    input_tensor = classification_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = clf_model(input_tensor)
        _, pred = torch.max(outputs, 1)
    
    return label_names[pred.item()]

# === SEGMENTATION FUNCTION ===
def segment_tumor(image: Image.Image) -> np.ndarray:
    image = ensure_rgb(image)
    input_tensor = segmentation_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(input_tensor)
        output = torch.sigmoid(output)  # sigmoid for binary mask
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)  # threshold into binary

    return mask  # returns 2D numpy array (binary mask)

# === COMBINED FUNCTION FOR GRADIO ===
def predict_and_segment(image: Image.Image):
    image = ensure_rgb(image)
    resized_image = image.resize((224, 224))
    
    # === CLASSIFICATION ===
    input_tensor = classification_transform(resized_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = clf_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        label = label_names[pred_idx.item()]
        confidence = conf.item()
    
    # === SEGMENTATION or original image ===
    if label == "no_tumor":
        return label, confidence, resized_image
    else:
        input_tensor = segmentation_transform(resized_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = seg_model(input_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

        # Resize mask to match resized_image (224x224)
        mask_img = Image.fromarray(mask * 255).resize(resized_image.size, resample=Image.NEAREST)
        mask_resized = np.array(mask_img) // 255  # Convert back to binary mask

        # Red overlay with transparency
        overlay = np.array(resized_image).copy()
        alpha = 0.4
        overlay[mask_resized == 1] = (
            overlay[mask_resized == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
        ).astype(np.uint8)

        return label, confidence, Image.fromarray(overlay)
