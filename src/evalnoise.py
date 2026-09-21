import os
import pandas as pd
import torch

from src.config import CHECKPOINT
from src.models.cnn import TrialCNN
from src.models.helper import get_criterion
from src.noises.corruptions import CORRUPTIONS
from src.noises.noiseloader import cloader
from src.train import evaluate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else'cpu')
corruptions = list(CORRUPTIONS.keys())
severities = [1,2,3,4,5]
dsets = ['GTSRB','CTSD','BTSD']
dsetmp = {
    'GTSRB' : os.path.join(CHECKPOINT, 'gtsrb_mixed.pth'),
    'CTSD' : os.path.join(CHECKPOINT, 'ctsd_mixed.pth'),
    'BTSD' : os.path.join(CHECKPOINT, 'btsd_mixed.pth'),
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
    for corr in corruptions:
        for sev in severities:
            loader = cloader(ds, CORRUPTIONS[corr],sev)
            criterion = get_criterion(ds)
            tstloss, tstacc, tstf1,tstprec,tstrec = evaluate(model,loader,criterion)
            results.append({
                'Dataset':ds, 'Corruption':corr, 'Severity' : sev,
                'loss' : tstloss, 'accuracy' : tstacc, 'f1 score' : tstf1,
                'precision' : tstprec, 'recall' : tstrec,
            })
            print(f"Dataset:{ds}, Corruption:{corr}, Severity : {sev}")
            print(f"loss : {tstloss}, accuracy : {tstacc}, f1 score : {tstf1}")
            print(f"precision : {tstprec}, recall : {tstrec}")

pd.DataFrame(results).to_csv('noise_results.csv',index=False)