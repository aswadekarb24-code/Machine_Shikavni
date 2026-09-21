import random

from PIL import Image
from torch.utils.data import Dataset

from src.noises.corruptions import CORRUPTIONS, corruptimg


class ProbabilisticCorruptData(Dataset):
    def __init__(self,basedata,cprob=0.5,transform=None):
        self.basedata=basedata
        self.cprob=cprob
        self.transform =transform

    def __len__(self):
        return len(self.basedata)

    def __getitem__(self,i):
        img, label = self.basedata.samples[i]
        img = Image.open(img).convert('RGB')

        if random.random() < self.cprob:
           corruption = random.choice(list(CORRUPTIONS.keys()))
           severity = random.randint(1,5)
           img = corruptimg(img,CORRUPTIONS[corruption],severity) 
        if self.transform is not None:
            img = self.transform(img)

        return img,label