from torch.utils.data import DataLoader

from src.config import BATCH_SIZE
from src.noises.noisedataset import ProbabilisticCorruptData
from src.noises.precorrupt import pre_cdata
from src.noises.fullnoisedset import fpre_cdata
from src.preproc.makedat import (
    b_testdat,
    b_traindat,
    c_testdat,
    c_traindat,
    g_testdat,
    g_traindat,
    tform,
)
import os
from src.config import EXTDIR

datamp ={
    'GTSRB': {'train' : g_traindat, 'test': g_testdat,'n_classes':43},
    'BTSD': {'train' : b_traindat, 'test': b_testdat,'n_classes':62},
    'CTSD': {'train' : c_traindat, 'test': c_testdat,'n_classes':58},
}

def getloader(dataset, batch_size = BATCH_SIZE, useprob=False, cprob=0.5):
    conf = datamp.get(dataset,None)
    if conf is None:
        raise ValueError("Dataset does not have torch dataset")

    train,test = conf['train'],conf['test']
    if os.path.exists(EXTDIR):
        train, test = pre_cdata(dataset,'train', useprob,cprob), pre_cdata(dataset,'test', useprob,cprob)
    elif useprob:
        train = ProbabilisticCorruptData(train,cprob=cprob,transform=tform)

    trainld = DataLoader(
        train, batch_size=batch_size,shuffle=True,
        num_workers=4,pin_memory=True
    )
    testld = DataLoader(
        test, batch_size=batch_size,shuffle=False,
        num_workers=4,pin_memory=True
    )
    return trainld,testld,conf['n_classes']
    
def get_fulllloader(dataset, batch_size = BATCH_SIZE):
    conf = datamp.get(dataset,None)
    if conf is None:
        raise ValueError("Dataset does not have torch dataset")

    train,test = conf['train'],conf['test']
    if os.path.exists(EXTDIR):
        train, test = fpre_cdata(dataset,'train'), fpre_cdata(dataset,'test')
    else:
        raise NotADirectoryError("FULL NOISE DATA NOT FOUND")
    
    trainld = DataLoader(
        train, batch_size=batch_size,shuffle=True,
        num_workers=4,pin_memory=True
    )
    testld = DataLoader(
        test, batch_size=batch_size,shuffle=False,
        num_workers=4,pin_memory=True
    )
    return trainld,testld,conf['n_classes']