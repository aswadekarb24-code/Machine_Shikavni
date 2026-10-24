import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from src.config import CHECKPOINT, EXTDIR, RESULTS
from src.models.factory import get_model
from src.models.helper import get_criterion
from src.noises.corruptions import CORRUPTIONS
from src.noises.noiseloader import cloader
from src.noises.precorrupt import pre_cloader
from src.train import evaluate
from src.ensemble import ensemble_probs, load_members, member_probs, metrics
from src.preproc.cache import labels_of, load_raw

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corruptions = list(CORRUPTIONS.keys())
severities = [1, 2, 3, 4, 5]
dsets = ['GTSRB','BTSD', 'CTSD']
# models_to_eval = ['trialcnn', 'resnet18', 'mobilenet_v2', 'efficientnet_b0']
models_to_eval = ['trialcnn']  # evaluated as the corruption-specialist ensemble

dset_n_classes = {
    'GTSRB': 43,
    'CTSD': 58,
    'BTSD': 62
}

SUFFIX = 'mixed'


def eval_ensemble(ds, n_cls):
    """Evaluate every ensemble member and their probability average on clean + all corruptions/severities."""
    members = load_members(ds)
    if not members:
        return []
    ytrue = labels_of(ds, 'test')
    yt = torch.from_numpy(ytrue)
    weight = get_criterion(ds).weight.cpu()
    confmat = np.zeros((n_cls, n_cls))
    rows = []
    for corr, sev in [('none', 0)] + [(c, s) for c in corruptions for s in severities]:
        raw = load_raw(ds, 'test', None if corr == 'none' else CORRUPTIONS[corr], sev)
        probs = member_probs(members, raw)
        probs['ensemble'] = ensemble_probs(probs)
        for name, p in probs.items():
            loss = torch.nn.functional.nll_loss(p.clamp_min(1e-8).log(), yt, weight=weight).item()
            ypred = p.argmax(1).numpy()
            acc, f1, prec, rec = metrics(ytrue, ypred)
            rows.append({'Model': f'trialcnn_{name}', 'Dataset': ds, 'Corruption': corr, 'Severity': sev,
                         'loss': loss, 'accuracy': acc, 'f1 score': f1, 'precision': prec, 'recall': rec})
            if name == 'ensemble' and corr != 'none':
                np.add.at(confmat, (ytrue, ypred), 1)
        print(f"{ds} | Corr: {corr:15s} | Sev: {sev} | " +
              " | ".join(f"{n}: {rows[-len(probs) + i]['accuracy']:.3f}" for i, n in enumerate(probs)))
    df_cm = pd.DataFrame(confmat, columns=[str(i) for i in range(n_cls)])
    df_cm.insert(0, 'true', range(n_cls))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_filename = os.path.join(RESULTS, f'trialcnn_ensemble_{ds}_noise_conf_mat_{timestamp}.csv')
    df_cm.to_csv(cm_filename, index=False)
    print(f"Saved confusion matrix to: {cm_filename}")
    return rows

results = []
print(f"Training on device: {DEVICE}")
for ds in dsets:
    n_cls = dset_n_classes[ds]
    for model_name in models_to_eval:
        if model_name == 'trialcnn':
            results.extend(eval_ensemble(ds, n_cls))
            continue
        modelpath = os.path.join(CHECKPOINT, f'{model_name.lower()}_{ds.lower()}_{SUFFIX}.pth')
        
        if not os.path.exists(modelpath):
            print(f"[SKIP] Model checkpoint not found: {modelpath}")
            continue

        print(f"\n==========================================")
        print(f"Evaluating: {model_name.upper()} | Dataset: {ds}")
        print(f"==========================================")
        
        model = get_model(model_name, in_channels=1, n_classes=n_cls).to(DEVICE)
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
        cm_filename = os.path.join(RESULTS,f'{model_name}_{ds}_noise_conf_mat_{timestamp}.csv')
        df_cm.to_csv(cm_filename, index=False)
        print(f"Saved confusion matrix to: {cm_filename}")

# Export combined corruption metrics across all evaluated models
if results:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(RESULTS,f'noise_results_all_models_{timestamp}.csv')
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nSaved all model noise results to: {output_csv}")