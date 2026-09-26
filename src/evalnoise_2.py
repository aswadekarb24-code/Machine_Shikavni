import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from src.config import CHECKPOINT, EXTDIR
from src.models.factory import get_model
from src.models.helper import get_criterion
from src.noises.corruptions import CORRUPTIONS
from src.noises.noiseloader import cloader
from src.noises.precorrupt import pre_cloader
from src.train import evaluate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corruptions = list(CORRUPTIONS.keys())
severities = [1, 2, 3, 4, 5]
dsets = ['BTSD', 'CTSD']
models_to_eval = ['trialcnn', 'resnet18', 'mobilenet_v2', 'efficientnet_b0']

dset_n_classes = {
    'GTSRB': 43,
    'CTSD': 58,
    'BTSD': 62
}

SUFFIX = 'fullnoise'
results = []

for ds in dsets:
    n_cls = dset_n_classes[ds]
    for model_name in models_to_eval:
        modelpath = os.path.join(CHECKPOINT, f'{model_name.lower()}_{ds.lower()}_{SUFFIX}.pth')
        
        if not os.path.exists(modelpath):
            print(f"[SKIP] Model checkpoint not found: {modelpath}")
            continue

        print(f"\n==========================================")
        print(f"Evaluating: {model_name.upper()} | Dataset: {ds}")
        print(f"==========================================")
        
        model = get_model(model_name, in_channels=3, n_classes=n_cls).to(DEVICE)
        model.load_state_dict(torch.load(modelpath, map_location=DEVICE))
        
        confmat = np.zeros((n_cls, n_cls))
        criterion = get_criterion(ds)

        for corr in corruptions:
            for sev in severities:
                fg = os.path.exists(EXTDIR)
                loader = pre_cloader(ds, 'test', useprob=True, cprob=1.0) if fg else cloader(ds, CORRUPTIONS[corr], sev)
                
                tstloss, tstacc, tstf1, tstprec, tstrec, ypreds, ytrues = evaluate(model, loader, criterion)
                
                for ytrue, ypred in zip(ytrues, ypreds):
                    confmat[int(ytrue)][int(ypred)] += 1
                    
                results.append({
                    'Model': model_name,
                    'Dataset': ds,
                    'Corruption': corr,
                    'Severity': sev,
                    'loss': tstloss,
                    'accuracy': tstacc,
                    'f1 score': tstf1,
                    'precision': tstprec,
                    'recall': tstrec,
                })
                print(f"Model: {model_name:12s} | Corr: {corr:15s} | Sev: {sev} | Acc: {tstacc:.4f} | F1: {tstf1:.4f}")

        # Save confusion matrix for this specific model and dataset
        df_cm = pd.DataFrame(confmat, columns=[str(i) for i in range(n_cls)])
        df_cm.insert(0, 'true', range(n_cls))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_filename = f'{model_name}_{ds}_noise_conf_mat_{timestamp}.csv'
        df_cm.to_csv(cm_filename, index=False)
        print(f"Saved confusion matrix to: {cm_filename}")

# Export combined corruption metrics across all evaluated models
if results:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f'noise_results_all_models_{timestamp}.csv'
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nSaved all model noise results to: {output_csv}")