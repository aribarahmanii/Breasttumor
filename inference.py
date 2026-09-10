"""
Inference helpers: load a checkpoint, preprocess an uploaded image, run prediction,
and build an overlay visualization. Pure functions, no Streamlit dependency here
so this can be tested standalone (e.g. in a notebook) before wiring into the app.
"""
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from models import MODEL_REGISTRY

IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str, checkpoint_path: str, device):
    """model_name must be one of MODEL_REGISTRY's keys."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}'. Options: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(pil_image: Image.Image, img_size: int = IMG_SIZE) -> torch.Tensor:
    """PIL image -> normalized tensor of shape [1, 3, img_size, img_size]."""
    transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=tuple(MEAN), std=tuple(STD)),
            ToTensorV2(),
        ]
    )
    image_np = np.array(pil_image.convert("RGB"))
    tensor = transform(image=image_np)["image"]
    return tensor.unsqueeze(0)


def predict_mask(model, image_tensor: torch.Tensor, device, threshold: float = 0.5) -> np.ndarray:
    """Returns a boolean mask of shape [img_size, img_size]."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).cpu().numpy()[0, 0]
    return mask


def predict_probability_map(model, image_tensor: torch.Tensor, device) -> np.ndarray:
    """Returns the raw sigmoid probability map, shape [img_size, img_size], values in [0, 1]."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
    return prob


def overlay_mask(pil_image: Image.Image, mask: np.ndarray, color=(255, 60, 60), alpha: float = 0.45,
                  img_size: int = IMG_SIZE) -> np.ndarray:
    """Blend a translucent color over the predicted tumor region, on top of the resized input image."""
    base = np.array(pil_image.resize((img_size, img_size)).convert("RGB")).astype(np.float32)
    overlay = base.copy()
    overlay[mask] = color
    blended = base * (1 - alpha) + overlay * alpha
    blended[~mask] = base[~mask]  # keep background pixels untouched, only blend inside the mask
    return blended.astype(np.uint8)


def mask_area_fraction(mask: np.ndarray) -> float:
    """Fraction of the image predicted as tumor — a simple, interpretable summary stat for the UI."""
    return float(mask.sum()) / mask.size
