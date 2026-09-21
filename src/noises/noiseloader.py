from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE
from src.noises.corruptions import CORRUPTIONS, corruptimg
from src.preproc.makedat import b_testdat, c_testdat, g_testdat, tform


class CorruptData(Dataset):
    def __init__(self, basedata, corruption, severity, transform=None):
        self.basedata = basedata
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.cfunc = CORRUPTIONS.get(self.corruption,None)
        if self.cfunc is None:
            raise ValueError("Invalid corruption function")

    def __len__(self):
        return len(self.basedata)

    def __getitem__(self,i):
        img, label = self.basedata.samples[i]
        img = Image.open(img).convert('RGB')
        cimg = corruptimg(img, self.cfunc, self.severity)
        if self.transform is not None:
            cimg = self.transform(cimg)
        return cimg,label

datasets ={
    'BTSD':b_testdat,
    'GTSRB':g_testdat,
    'CTSD':c_testdat
}

def cloader(dataset,corruption,severity=1):
    dataset = datasets.get(dataset,None)
    if dataset is None:
        raise ValueError("Unknown dataset")

    cdataset = CorruptData(
        basedata=dataset,
        corruption=corruption,
        severity=severity,
        transform=tform
    )
    return DataLoader(
        cdataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
