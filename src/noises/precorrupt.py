from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
from src.config import BATCH_SIZE
from src.noises.corruptions import CORRUPTIONS, corruptimg
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
class PreCorruptData(Dataset):
    def __init__(self, dsnm, type, transform=None, useprob=False, cprob=0.5):
        self.basedata = dsetmp[dsnm][type]
        self.corruption = random.choice(list(CORRUPTIONS.keys()))
        self.severity = random.randint(1,5)
        self.transform = transform
        self.dirr = os.path.join(EXTDIR,self.corruption,str(self.severity),dsnm,type)
        self.useprob = useprob
        self.cprob = cprob

    def __len__(self):
        return len(self.basedata)

    def __getitem__(self,i):
        impth, label = self.basedata.samples[i]
        if self.useprob and self.cprob > random.random():
            bpth = os.path.splitext(os.path.basename(impth))[0]
            odirr = os.path.join(self.dirr,str(label))
            npy_path = os.path.join(odirr,f'{i}_{bpth}.npy')
            img_np = np.load(npy_path)
            img = Image.fromarray(img_np)
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        img = Image.open(impth).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,label

def pre_cloader(dataset,type, useprob=False,cprob=0.5):
    if dataset not in dsetmp.keys() or type not in dsetmp[dataset].keys():
        raise ValueError("Unknown dataset/type")

    cdataset = PreCorruptData(
        dsnm=dataset,
        type=type, 
        transform=tform,
        useprob=useprob,
        cprob=cprob
    )
    return DataLoader(
        cdataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

def pre_cdata(dataset,type,useprob=False,cprob=0.5):
    if dataset not in dsetmp.keys() or type not in dsetmp[dataset].keys():
        raise ValueError("Unknown dataset/type")

    return PreCorruptData(
        dsnm=dataset,
        type=type,
        transform=tform,
        useprob=useprob,
        cprob=cprob
    )