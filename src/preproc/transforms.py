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