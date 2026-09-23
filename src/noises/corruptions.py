import numpy as np
import cv2
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter, map_coordinates

# Severity scales (1 to 5) for noise parameters
SEVERITY_LEVELS = {
    'gaussian_noise': [0.04, 0.06, 0.08, 0.09, 0.10],
    'shot_noise': [60, 25, 12, 5, 3],
    'impulse_noise': [0.03, 0.06, 0.09, 0.17, 0.27],
    'defocus_blur': [0.5, 1.0, 1.5, 2.0, 2.5],
    'glass_blur': [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)],
    'motion_blur': [3, 5, 7, 9, 11],
    'zoom_blur': [
        np.linspace(1, 1.06, 5),
        np.linspace(1, 1.12, 7),
        np.linspace(1, 1.18, 9),
        np.linspace(1, 1.24, 11),
        np.linspace(1, 1.30, 13)
    ],
    'snow': [0.1, 0.2, 0.3, 0.4, 0.5],
    'frost': [0.1, 0.2, 0.3, 0.4, 0.5],
    'fog': [0.2, 0.35, 0.5, 0.65, 0.8],
    'brightness': [0.1, 0.2, 0.3, 0.4, 0.5],
    'contrast': [0.5, 0.4, 0.3, 0.2, 0.1],
    'elastic_transform': [(15, 3), (20, 3), (25, 4), (30, 4), (35, 5)],
    'pixelate': [0.8, 0.6, 0.4, 0.3, 0.2]
}

# 1. Noise Corruptions
def gaussian_noise(img, severity=1):
    a = np.array(img) / 255.0
    c = SEVERITY_LEVELS['gaussian_noise'][severity - 1]
    noise = np.random.normal(size=a.shape, scale=c)
    a = np.clip(a + noise, 0, 1) * 255
    return Image.fromarray(np.uint8(a))

def shot_noise(img, severity=1):
    c = SEVERITY_LEVELS['shot_noise'][severity - 1]
    a = np.array(img) / 255.0
    a = np.random.poisson(a * c) / float(c)
    a = np.clip(a, 0, 1) * 255
    return Image.fromarray(np.uint8(a))

def impulse_noise(img, severity=1):
    c = SEVERITY_LEVELS['impulse_noise'][severity - 1]
    a = np.array(img) / 255.0
    mask = np.random.uniform(0, 1, size=a.shape[:2])
    a[mask < c / 2] = 0
    a[(mask >= c / 2) & (mask < c)] = 1
    a = np.clip(a, 0, 1) * 255
    return Image.fromarray(np.uint8(a))

# 2. Blur Corruptions
def defocus_blur(img, severity=1):
    c = SEVERITY_LEVELS['defocus_blur'][severity - 1]
    a = np.array(img).astype(np.float32)
    a = gaussian_filter(a, sigma=(c, c, 0))
    a = np.clip(a, 0, 255)
    return Image.fromarray(np.uint8(a))

def glass_blur(img, severity=1):
    c = SEVERITY_LEVELS['glass_blur'][severity - 1]
    a = np.array(img)
    h, w, _ = a.shape
    for _ in range(c[1]):
        for i in range(c[0], h - c[0]):
            for j in range(c[0], w - c[0]):
                dx = np.random.randint(-c[0], c[0] + 1)
                dy = np.random.randint(-c[0], c[0] + 1)
                a[i, j], a[i + dx, j + dy] = a[i + dx, j + dy], a[i, j]
    return Image.fromarray(np.uint8(a))

def motion_blur(img, severity=1):
    c = SEVERITY_LEVELS['motion_blur'][severity - 1]
    kernel = np.zeros((c, c))
    kernel[int((c - 1) / 2), :] = np.ones(c)
    kernel /= c
    a = np.array(img)
    a = cv2.filter2D(a, -1, kernel)
    return Image.fromarray(np.uint8(a))

def zoom_blur(img, severity=1):
    c = SEVERITY_LEVELS['zoom_blur'][severity - 1]
    a = np.array(img).astype(np.float32)
    out = np.zeros_like(a)
    h, w, _ = a.shape
    for factor in c:
        nh, nw = int(h * factor), int(w * factor)
        resized = cv2.resize(a, (nw, nh))
        top, left = (nh - h) // 2, (nw - w) // 2
        out += resized[top:top + h, left:left + w]
    out /= len(c)
    a = np.clip(out, 0, 255)
    return Image.fromarray(np.uint8(a))

# 3. Weather Corruptions
def snow(img, severity=1):
    c = SEVERITY_LEVELS['snow'][severity - 1]
    a = np.array(img).astype(np.float32) / 255.0
    snow_layer = np.random.normal(loc=0.5, scale=0.3, size=a.shape[:2])
    snow_layer = gaussian_filter(snow_layer, sigma=1.0)
    snow_mask = snow_layer < c
    a[snow_mask] = np.clip(a[snow_mask] + 0.5, 0, 1)
    a = np.clip(a * 255, 0, 255)
    return Image.fromarray(np.uint8(a))

def frost(img, severity=1):
    c = SEVERITY_LEVELS['frost'][severity - 1]
    a = np.array(img).astype(np.float32) / 255.0
    frost_pattern = np.random.uniform(0, 1, size=a.shape)
    frost_pattern = gaussian_filter(frost_pattern, sigma=(2, 2, 0))
    a = (1 - c) * a + c * frost_pattern
    a = np.clip(a * 255, 0, 255)
    return Image.fromarray(np.uint8(a))

def fog(img, severity=1):
    c = SEVERITY_LEVELS['fog'][severity - 1]
    a = np.array(img).astype(np.float32) / 255.0
    h, w, _ = a.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    fog_layer = c * (1 - np.sqrt(xx**2 + yy**2) / np.sqrt(2))
    fog_layer = np.clip(fog_layer[:, :, None], 0, 1)
    a = a * (1 - fog_layer) + fog_layer
    a = np.clip(a * 255, 0, 255)
    return Image.fromarray(np.uint8(a))

# 4. Digital & Geometric Corruptions
def brightness(img, severity=1):
    c = SEVERITY_LEVELS['brightness'][severity - 1]
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1 + c)

def contrast(img, severity=1):
    c = SEVERITY_LEVELS['contrast'][severity - 1]
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(c)

def elastic_transform(img, severity=1):
    c = SEVERITY_LEVELS['elastic_transform'][severity - 1]
    a = np.array(img)
    h, w, ch = a.shape
    
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), c[1]) * c[0]
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), c[1]) * c[0]
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    channels = []
    for i in range(ch):
        channel = map_coordinates(a[:, :, i], indices, order=1, mode='reflect')
        channels.append(channel.reshape(h, w))
        
    a = np.stack(channels, axis=-1)
    return Image.fromarray(np.uint8(np.clip(a, 0, 255)))

def pixelate(img, severity=1):
    c = SEVERITY_LEVELS['pixelate'][severity - 1]
    w, h = img.size
    small_w, small_h = max(1, int(w * c)), max(1, int(h * c))
    img_small = img.resize((small_w, small_h), resample=Image.BOX)
    return img_small.resize((w, h), resample=Image.NEAREST)

# 5. Corruption Registry
CORRUPTION_REGISTRY = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur,
    'glass_blur': glass_blur,
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'snow': snow,
    'frost': frost,
    'fog': fog,
    'brightness': brightness,
    'contrast': contrast,
    'elastic_transform': elastic_transform,
    'pixelate': pixelate
}
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

SEVERITIES = [1,2,3,4,5]
from imagecorruptions import corrupt
def corruptimg(img:Image.Image, corruption, severity=1):
    og_h, og_w = img.height, img.width
    n_img = img.copy()
    if og_w < 32 or og_h < 32:
        nw, nh = max(32,og_w), max(32,og_h)
        n_img = img.resize((nw,nh), Image.Resampling.BILINEAR)
    nd_arr_img = np.asarray(n_img)
    corr_nd_arr = corrupt(nd_arr_img, corruption_name=CORRUPTIONS[corruption],severity=severity)
    corr_img = Image.fromarray(corr_nd_arr)
    retimg=corr_img
    if corr_img.width != og_w or corr_img.height != og_h:
        retimg = corr_img.resize((og_w,og_h),Image.Resampling.BILINEAR)
    return retimg
    # return (CORRUPTION_REGISTRY[CORRUPTIONS[corruption]](img,severity=severity))