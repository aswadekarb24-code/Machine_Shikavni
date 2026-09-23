from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
from src.config import BATCH_SIZE
from src.noises.corruptions import CORRUPTIONS, SEVERITIES
from src.preproc.makedat import (
    g_traindat,c_traindat,b_traindat,
    b_testdat, c_testdat, g_testdat, tform
)

from src.config import EXTDIR
dsetmp = {
    'GTSRB' :{'train' : g_traindat, 'test':g_testdat},
    'BTSD' :{'train' : b_traindat, 'test':b_testdat},
    'CTSD' : {'train':c_traindat, 'test':c_testdat}
}
corlen, sevlen = len(CORRUPTIONS), len(SEVERITIES)
corrlst = list(CORRUPTIONS.keys())
class FullPreCorruptData(Dataset):
    def __init__(self, dsnm, type, transform=None):
        self.dsnm=dsnm
        self.type = type
        self.basedata = dsetmp[dsnm][type]
        self.transform = transform
    def __len__(self):
        return (1+corlen*sevlen)*len(self.basedata)

    def __getitem__(self,i):
        i_mod = i % len(self.basedata)
        impth, label = self.basedata.samples[i_mod]
        bpth = os.path.splitext(os.path.basename(impth))[0]
        ident = i // len(self.basedata) - 1
        if ident < 0:
            img = Image.open(impth).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img,label

        corr_i = ident // sevlen
        sev_i = (ident % sevlen)
        odirr = os.path.join(EXTDIR,corrlst[corr_i], str(SEVERITIES[sev_i]), self.dsnm, self.type,str(label))
        
        npy_path = os.path.join(odirr,f'{i_mod}_{bpth}.npy')
        img_np = np.load(npy_path)
        img = Image.fromarray(img_np)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

from src.preproc.transforms import train_tform

def fpre_cloader(dataset,type):
    if dataset not in dsetmp.keys() or type not in dsetmp[dataset].keys():
        raise ValueError("Unknown dataset/type")

    fcdataset = FullPreCorruptData(
        dsnm=dataset,
        type=type, 
        transform=train_tform,
    )
    return DataLoader(
        fcdataset,
        batch_size=BATCH_SIZE,
        shuffle=type=='train',
        pin_memory=True
    )

def fpre_cdata(dataset,type):
    if dataset not in dsetmp.keys() or type not in dsetmp[dataset].keys():
        raise ValueError("Unknown dataset/type")

    return FullPreCorruptData(
        dsnm=dataset,
        type=type,
        transform=tform,
    )