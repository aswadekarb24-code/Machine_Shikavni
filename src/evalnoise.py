import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from src.config import CHECKPOINT
from src.models.cnn import TrialCNN
from src.models.helper import get_criterion
from src.noises.corruptions import CORRUPTIONS
from src.noises.noiseloader import cloader
from src.noises.precorrupt import pre_cloader
from src.train import evaluate
from src.config import EXTDIR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else'cpu')
corruptions = list(CORRUPTIONS.keys())
severities = [1,2,3,4,5]
dsets = ['BTSD', 'CTSD'] # ['GTSRB','CTSD','BTSD']
dsetmp = {
    'GTSRB' : os.path.join(CHECKPOINT, 'gtsrb_mixed.pth'),
    'CTSD' : os.path.join(CHECKPOINT, 'ctsd_mixed.pth'),
    'BTSD' : os.path.join(CHECKPOINT, 'btsd_fullnoise.pth'),
}

dset_n_classes = {
    'GTSRB' : 43,
    'CTSD':58,
    'BTSD':62
}

results = []

for ds in dsets:
    model = TrialCNN(in_channels = 3, n_classes =dset_n_classes[ds]).to(DEVICE)
    modelpath = dsetmp[ds]
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"Model:{modelpath} not found")
    model.load_state_dict(torch.load(modelpath))
    confmat = np.zeros((dset_n_classes[ds],dset_n_classes[ds]))
    for corr in corruptions:
        for sev in severities:
            fg = os.path.exists(EXTDIR)
            loader = fpre_cloader(ds,'test',useprob=True, cprob=1.0) if fg else cloader(ds, CORRUPTIONS[corr],sev)
            criterion = get_criterion(ds)
            tstloss, tstacc, tstf1,tstprec,tstrec, ypreds, ytrues = evaluate(model,loader,criterion)
            for ytrue, ypred in zip(ytrues, ypreds):
                confmat[int(ytrue)][int(ypred)] += 1
            results.append({
                'Dataset':ds, 'Corruption':corr, 'Severity' : sev,
                'loss' : tstloss, 'accuracy' : tstacc, 'f1 score' : tstf1,
                'precision' : tstprec, 'recall' : tstrec,
            })
            print(f"Dataset:{ds}, Corruption:{corr}, Severity : {sev}")
            print(f"loss : {tstloss}, accuracy : {tstacc}, f1 score : {tstf1}")
            print(f"precision : {tstprec}, recall : {tstrec}")
    df = pd.DataFrame(
        confmat,
        columns=[str(i) for i in range(dset_n_classes[ds])]
    )
df.insert(0, 'true', range(dset_n_classes[ds]))
df.to_csv(f'{ds}_noise_conf_mat_{datetime.now()}.csv', index=False)
pd.DataFrame(results).to_csv(f'noise_results_{datetime.now()}.csv',index=False)