from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
from collections import OrderedDict
from src.config import BATCH_SIZE
from src.config import IMG_SIZE
from src.noises.corruptions import CORRUPTIONS
from src.preproc.makedat import (
    g_traindat,c_traindat,b_traindat,
    b_testdat, c_testdat, g_testdat, tform
)

from src.preproc.transforms import tform_robust, train_tform

from src.config import EXTDIR
dsetmp = {
    'GTSRB' :{'train' : g_traindat, 'test':g_testdat},
    'BTSD' :{'train' : b_traindat, 'test':b_testdat},
    'CTSD' : {'train':c_traindat, 'test':c_testdat}
}

TRANSFORMS = {
    'robust': (tform_robust, 1),
    'train': (train_tform, 3),
}


def transform_spec(name):
    """Return the preprocessing transform and required model input channels."""
    try:
        return TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown transform '{name}'. Choose one of {list(TRANSFORMS)}") from exc


class PreCorruptData(Dataset):
    def __init__(self, dsnm, type, transform=None, useprob=False, cprob=0.5,
                 corruption=None, severity=None, cache_items=8192):
        self.basedata = dsetmp[dsnm][type]
        self.corruption = corruption or random.choice(list(CORRUPTIONS.keys()))
        self.severity = severity or random.randint(1,5)
        if self.corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption: {self.corruption}")
        if self.severity not in range(1, 6):
            raise ValueError(f"Severity must be 1-5, got {self.severity}")
        self.transform = transform
        self.dirr = os.path.join(EXTDIR,self.corruption,str(self.severity),dsnm,type)
        self.useprob = useprob
        self.cprob = cprob
        # Each persistent DataLoader worker owns this bounded cache. It stores
        # decoded, resized RGB pixels only, preserving train_tform randomness.
        self.cache_items = cache_items
        self._image_cache = OrderedDict()

    def __len__(self):
        return len(self.basedata)

    def _cached_image(self, path, from_npy=False):
        key = ('npy' if from_npy else 'image', path)
        pixels = self._image_cache.pop(key, None)
        if pixels is None:
            if from_npy:
                img = Image.fromarray(np.load(path))
            else:
                with Image.open(path) as source:
                    img = source.convert('RGB')
            pixels = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)).copy()
            if self.cache_items:
                self._image_cache[key] = pixels
                if len(self._image_cache) > self.cache_items:
                    self._image_cache.popitem(last=False)
        else:
            self._image_cache[key] = pixels
        return Image.fromarray(pixels)

    def __getitem__(self,i):
        impth, label = self.basedata.samples[i]
        if self.useprob and self.cprob > random.random():
            bpth = os.path.splitext(os.path.basename(impth))[0]
            odirr = os.path.join(self.dirr,str(label))
            npy_path = os.path.join(odirr,f'{i}_{bpth}.npy')
            img = self._cached_image(npy_path, from_npy=True)
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        img = self._cached_image(impth)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

def pre_cloader(dataset, type, useprob=False, cprob=0.5, transform_name='robust',
                corruption=None, severity=None, batch_size=BATCH_SIZE):
    if dataset not in dsetmp.keys() or type not in dsetmp[dataset].keys():
        raise ValueError("Unknown dataset/type")

    transform, _ = transform_spec(transform_name)
    cdataset = PreCorruptData(
        dsnm=dataset,
        type=type, 
        transform=transform,
        useprob=useprob,
        cprob=cprob,
        corruption=corruption,
        severity=severity,
    )
    return DataLoader(
        cdataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        # old_evalnoise creates a fresh one-pass loader for every condition.
        # Persistent workers would accumulate atexit cleanup callbacks and can
        # fail during interpreter shutdown after a full evaluation sweep.
    )

def pre_cdata(dataset, type, useprob=False, cprob=0.5, transform_name='robust'):
    if dataset not in dsetmp.keys() or type not in dsetmp[dataset].keys():
        raise ValueError("Unknown dataset/type")

    transform, _ = transform_spec(transform_name)
    return PreCorruptData(
        dsnm=dataset,
        type=type,
        transform=transform,
        useprob=useprob,
        cprob=cprob
    )
