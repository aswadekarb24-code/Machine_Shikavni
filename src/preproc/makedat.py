import os

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import BATCH_SIZE, IMG_SIZE, PROCESSED_DIR

tform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

gdir_train, gdir_test = os.path.join(PROCESSED_DIR,'GTSRB','train'),os.path.join(PROCESSED_DIR,'GTSRB','test')
bdir_train, bdir_test = os.path.join(PROCESSED_DIR,'BTSD','train'),os.path.join(PROCESSED_DIR,'BTSD','test')
cdir_train, cdir_test = os.path.join(PROCESSED_DIR,'CTSD','train'),os.path.join(PROCESSED_DIR,'CTSD','test')

def labelhelp(dataset):
    dataset.samples = [(path,int(os.path.basename(os.path.dirname(path)))) for path,_ in dataset.samples]
    dataset.targets = [target for _,target in dataset.samples]
    dataset.class_to_idx ={str(i):i for i in range(len(dataset.classes))}

c_traindat = datasets.ImageFolder(root=cdir_train,transform=tform)
labelhelp(c_traindat)
c_testdat = datasets.ImageFolder(root=cdir_test,transform=tform)
labelhelp(c_testdat)
g_traindat = datasets.ImageFolder(root=gdir_train,transform=tform)
labelhelp(g_traindat)
g_testdat = datasets.ImageFolder(root=gdir_test,transform=tform)
labelhelp(g_testdat)
b_traindat = datasets.ImageFolder(root=bdir_train,transform=tform)
labelhelp(b_traindat)
b_testdat = datasets.ImageFolder(root=bdir_test,transform=tform)
labelhelp(b_testdat)

c_train_loader = DataLoader(c_traindat,batch_size=BATCH_SIZE,shuffle=True, pin_memory=True)
c_test_loader = DataLoader(c_testdat,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True)

b_train_loader = DataLoader(b_traindat,batch_size=BATCH_SIZE,shuffle=True, pin_memory=True)
b_test_loader = DataLoader(b_testdat,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True)

g_train_loader = DataLoader(g_traindat,batch_size=BATCH_SIZE,shuffle=True, pin_memory=True)
g_test_loader = DataLoader(g_testdat,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True)

print(f"CTSD Train samples: {len(c_traindat)}")
print(f"CTSD Test samples: {len(c_testdat)}")

print(f"GTSRB Train samples: {len(g_traindat)}")
print(f"GTSRB Test samples: {len(g_testdat)}")

print(f"BTSD Train samples: {len(b_traindat)}")
print(f"BTSD Test samples: {len(b_testdat)}")