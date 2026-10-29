import numpy as np
from PIL import Image
CORRUPTIONS = {
    "gaussian_noise":"gaussian_noise",
    "shot_noise":"shot_noise",
    "impulse_noise":"impulse_noise",
    "defocus_blur":"defocus_blur",
    "jpeg_compression":"jpeg_compression",
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
