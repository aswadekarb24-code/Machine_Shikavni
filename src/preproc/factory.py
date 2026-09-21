from torch.utils.data import DataLoader

from src.config import BATCH_SIZE
from src.noises.noisedataset import ProbabilisticCorruptData
from src.preproc.makedat import (
    b_testdat,
    b_traindat,
    c_testdat,
    c_traindat,
    g_testdat,
    g_traindat,
    tform,
)

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

    if useprob:
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
    
    
    