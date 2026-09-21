from imagecorruptions import corrupt
from PIL import Image
import numpy as np
CORRUPTIONS = {
    "gaussian_noise":"gaussian_noise",
    "shot_noise":"shot_noise",
    "impulse_noise":"impulse_noise",
    "defocus_blur":"defocus_blur",
    "glass_blur":"glass_blur",
    "motion_blur":"motion_blur",
    "zoom_blur":"zoom_blur",
    "snow":"snow",
    "fog":"fog",
    "frost":"frost",
    "brightness":"brightness",
    "contrast":"contrast",
    "elastic_transform":"elastic_transform",
    "pixelate":"pixelate",
}

def corruptimg(img, corruption, severity=1):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.width < 32 and img.height < 32:
        new_w = max(32, img.width)
        new_h = max(32, img.height)
        img = img.resize((new_w,new_h), Image.Resampling.BILINEAR)
    img = np.array(img)
    return Image.fromarray(corrupt(img, corruption_name=CORRUPTIONS.get(corruption,"gaussian_noise"), severity=severity))