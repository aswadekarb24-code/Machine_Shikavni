import torchvision.transforms.v2 as transforms
from src.config import IMG_SIZE
import torch
train_tform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
import cv2
import numpy as np
from PIL import Image

class RobustPreprocessShield:
    def __call__(self, img):
        # 1. Convert PIL to OpenCV format (NumPy array)
        img_np = np.array(img.convert('RGB'))
        
        # 2. Convert to Grayscale (removes color-based corruption vulnerabilities)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 3. Apply Median Blur to kill pixel noise (Shot, Impulse, Gaussian)
        # A kernel size of 3 is usually enough to kill noise without destroying shapes
        denoised = cv2.medianBlur(gray, 3)
        
        # 4. Apply CLAHE to destroy Fog, Contrast, and Brightness issues
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        equalized = clahe.apply(denoised)
        
        # 5. Convert back to PIL Image so PyTorch can use it
        # Note: Image is now 1 channel (grayscale) instead of 3
        return Image.fromarray(equalized)

# Inject the shield into your transformation pipeline
tform_robust = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    RobustPreprocessShield(), # <-- The Magic Happens Here
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    # Note: Normalize only needs 1 value now because the image is Grayscale
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

# ---------------------------------------------------------------------------
# Per-member preprocessing for the corruption-specialist ensemble.
# Each function maps one IMG_SIZE x IMG_SIZE x 3 uint8 RGB array to a
# IMG_SIZE x IMG_SIZE uint8 grayscale array. Scaling to [-1, 1] is done by
# src.preproc.cache.to_input, equivalent to ToDtype(scale) + Normalize(0.5, 0.5).
# ---------------------------------------------------------------------------
def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def _clahe(g, clip=2.0):
    # built per call: cv2 CLAHE objects are not thread-safe
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4)).apply(g)

def noise_tform(rgb):
    g = cv2.medianBlur(_gray(rgb), 3)
    return cv2.bilateralFilter(g, d=3, sigmaColor=40, sigmaSpace=40)

def blur_tform(rgb):
    g = _gray(rgb)
    soft = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.addWeighted(g, 1.8, soft, -0.8, 0)

def light_tform(rgb):
    g = _gray(rgb)
    lo, hi = np.percentile(g, (2, 98))
    g = np.clip((g.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
    return _clahe(g, 3.0)

def weather_tform(rgb):
    return _clahe(cv2.medianBlur(_gray(rgb), 3))

def distort_tform(rgb):
    return cv2.GaussianBlur(_gray(rgb), (0, 0), 0.8)

def generalist_tform(rgb):
    # same steps as RobustPreprocessShield
    return _clahe(cv2.medianBlur(_gray(rgb), 3))

def fog_tform(rgb, sigma=1.25, eps=8.0):
    """Local contrast normalisation: removes the low-frequency veil that fog adds (sigma chosen by sweep)."""
    g = _gray(rgb).astype(np.float32)
    m = cv2.GaussianBlur(g, (0, 0), sigma)
    v = cv2.GaussianBlur((g - m) ** 2, (0, 0), sigma)
    x = (g - m) / (np.sqrt(v) + eps)
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)


def _lowpass(rgb, sigma):
    return _clahe(cv2.GaussianBlur(_gray(rgb), (0, 0), sigma))

def defocus_tform(rgb):
    """Pre-smoothing so the clean-trained net relies on low-frequency structure that survives defocus blur."""
    return _lowpass(rgb, 2.0)

def heavyblur_tform(rgb):
    return _lowpass(rgb, 3.0)

def motion_tform(rgb, sigma=1.5, iters=15):
    """Richardson-Lucy deconvolution (Gaussian PSF); helped motion blur in the transform sweep."""
    g = _gray(rgb).astype(np.float32) + 1
    x = g.copy()
    for _ in range(iters):
        est = cv2.GaussianBlur(x, (0, 0), sigma)
        x = x * cv2.GaussianBlur(g / (est + 1e-3), (0, 0), sigma)
    lo, hi = np.percentile(x, (1, 99))
    return _clahe(np.clip((x - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8))
