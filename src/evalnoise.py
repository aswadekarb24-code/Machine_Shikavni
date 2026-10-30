import os

import numpy as np
import pandas as pd
import torch

from src.config import RESULTS
from src.ensemble import ENSEMBLE_MEMBERS, ensemble_probs, load_members, metrics, specialist_probs
from src.models.helper import get_criterion
from src.noises.corruptions import CORRUPTIONS
from src.preproc.cache import labels_of, load_raw

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CORRUPTIONS_TO_EVAL = list(CORRUPTIONS)
SEVERITIES = [1, 2, 3, 4, 5]
DATASETS = ['BTSD', 'CTSD', 'GTSRB']
ARCHITECTURES = ['trialcnn', 'resnet18', 'mobilenet_v2', 'efficientnet_b0']
EVAL_SPLITS = ('train', 'test')
N_CLASSES = {'GTSRB': 43, 'CTSD': 58, 'BTSD': 62}

# Stable paths let a second invocation resume the same evaluation.
OUTPUT_CSV = os.path.join(RESULTS, 'noise_results_tsrd_research_ensemble.csv')
LOG_DIR = 'logs'
PROGRESS_CSV = os.path.join(LOG_DIR, 'eval_tsrd_ensemble_progress.csv')
CM_PARTS_DIR = os.path.join(LOG_DIR, 'eval_tsrd_ensemble_confmat_parts')
COLUMNS = ['Model', 'Dataset', 'Corruption', 'Severity',
           'loss', 'accuracy', 'f1 score', 'precision', 'recall']
GLOBAL_ENSEMBLE = 'tsrd_ensemble'


def conditions():
    return [('none', 0)] + [(corr, sev) for corr in CORRUPTIONS_TO_EVAL for sev in SEVERITIES]


def expected_models(arches = ARCHITECTURES):
    return {f'{arch}_{member}' for arch in arches for member in ENSEMBLE_MEMBERS} | {GLOBAL_ENSEMBLE}


def part_path(ds, corr, sev):
    return os.path.join(CM_PARTS_DIR, f'{ds}_{corr}_{sev}.npy')


def load_completed_rows():
    if not os.path.exists(OUTPUT_CSV):
        return set()
    df = pd.read_csv(OUTPUT_CSV, usecols=['Model', 'Dataset', 'Corruption', 'Severity'])
    return set(zip(df['Model'], df['Dataset'], df['Corruption'], df['Severity']))


def append_rows(rows, completed):
    """Append and fsync rows, so a completed model never produces a duplicate CSV row."""
    rows = [row for row in rows if (row['Model'], row['Dataset'], row['Corruption'], row['Severity']) not in completed]
    if not rows:
        return
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    new_file = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', encoding='utf-8', newline='') as handle:
        pd.DataFrame(rows, columns=COLUMNS).to_csv(handle, header=new_file, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    completed.update((row['Model'], row['Dataset'], row['Corruption'], row['Severity']) for row in rows)


def save_array_atomic(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp'
    with open(tmp, 'wb') as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_progress(completed,arches=ARCHITECTURES):
    expected = expected_models(arches=arches)
    done_by_condition = {}
    for model, ds, corr, sev in completed:
        done_by_condition.setdefault((ds, corr, sev), set()).add(model)
    rows = []
    for ds in DATASETS:
        for corr, sev in conditions():
            done = done_by_condition.get((ds, corr, sev), set())
            cm_saved = corr == 'none' or os.path.exists(part_path(ds, corr, sev))
            rows.append({
                'Dataset': ds, 'Corruption': corr, 'Severity': sev,
                'Models complete': len(done & expected),
                'Ensemble complete': GLOBAL_ENSEMBLE in done,
                'Confusion part saved': cm_saved,
                'Complete': done >= expected and cm_saved,
            })
    os.makedirs(LOG_DIR, exist_ok=True)
    tmp = f'{PROGRESS_CSV}.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, PROGRESS_CSV)


def row_for_probs(model_name, ds, corr, sev, probs, ytrue, yt, weight):
    loss = torch.nn.functional.nll_loss(probs.clamp_min(1e-8).log(), yt, weight=weight).item()
    ypred = probs.argmax(1).numpy()
    acc, f1, prec, rec = metrics(ytrue, ypred)
    return ({'Model': model_name, 'Dataset': ds, 'Corruption': corr, 'Severity': sev,
             'loss': loss, 'accuracy': acc, 'f1 score': f1, 'precision': prec, 'recall': rec}, ypred)


def load_all_members(ds,arches = ARCHITECTURES):
    """Require the intended 32 checkpoints; a partial ensemble is never evaluated silently."""
    members_by_arch = {}
    for arch in arches:
        members = load_members(ds, arch=arch)
        missing = set(ENSEMBLE_MEMBERS) - set(members)
        if missing:
            raise FileNotFoundError(f'{ds}/{arch} is missing ensemble checkpoints: {sorted(missing)}')
        members_by_arch[arch] = members
    return members_by_arch


def evaluate_dataset(ds, completed,arches=ARCHITECTURES):
    n_cls = N_CLASSES[ds]
    members_by_arch = load_all_members(ds,arches=arches)
    ytrue = np.concatenate([labels_of(ds, split) for split in EVAL_SPLITS])
    yt = torch.from_numpy(ytrue)
    weight = get_criterion(ds).weight.cpu()
    expected = expected_models(arches=arches)

    for number, (corr, sev) in enumerate(conditions(), start=1):
        done = {model for model, row_ds, row_corr, row_sev in completed
                if (row_ds, row_corr, row_sev) == (ds, corr, sev)}
        cm_file = part_path(ds, corr, sev)
        if done >= expected and (corr == 'none' or os.path.exists(cm_file)):
            print(f'[{ds} {number:02d}/71] {corr} severity {sev}: already complete')
            continue

        raw = np.concatenate([
            load_raw(ds, split, None if corr == 'none' else corr, sev)
            for split in EVAL_SPLITS
        ])
        all_probs = {}
        for member_name in ENSEMBLE_MEMBERS:
            # reuse same transform for all 4 archs
            models = {arch: members_by_arch[arch][member_name] for arch in arches}
            probs_by_arch = specialist_probs(member_name, models, raw)
            for arch, probs in probs_by_arch.items():
                model_name = f'{arch}_{member_name}'
                all_probs[model_name] = probs
                row, _ = row_for_probs(model_name, ds, corr, sev, probs, ytrue, yt, weight)
                append_rows([row], completed)
                save_progress(completed)
                print(f'[{ds} {number:02d}/71] {corr} severity {sev} | {model_name} complete')

        ensemble = ensemble_probs(all_probs)
        ensemble_row, ypred = row_for_probs(GLOBAL_ENSEMBLE, ds, corr, sev, ensemble, ytrue, yt, weight)
        append_rows([ensemble_row], completed)
        if corr != 'none':
            confmat = np.zeros((n_cls, n_cls), dtype=np.int64)
            np.add.at(confmat, (ytrue, ypred), 1)
            save_array_atomic(cm_file, confmat)
        save_progress(completed)
        print(f'[{ds} {number:02d}/71] {corr} severity {sev} | {GLOBAL_ENSEMBLE} complete')


def write_final_confusion_matrices(completed, arches = ARCHITECTURES):
    expected = expected_models(arches=arches)
    for ds in DATASETS:
        parts = []
        for corr, sev in conditions():
            if corr == 'none':
                continue
            done = {model for model, row_ds, row_corr, row_sev in completed
                    if (row_ds, row_corr, row_sev) == (ds, corr, sev)}
            path = part_path(ds, corr, sev)
            if done < expected or not os.path.exists(path):
                print(f'[PENDING] {ds}: final confusion matrix waits for {corr} severity {sev}')
                break
            parts.append(np.load(path))
        else:
            confmat = np.sum(parts, axis=0)
            df = pd.DataFrame(confmat, columns=[str(i) for i in range(N_CLASSES[ds])])
            df.insert(0, 'true', range(N_CLASSES[ds]))
            output = os.path.join(RESULTS, f'ensemble_32_{ds}_noise_conf_mat.csv')
            df.to_csv(output, index=False)
            print(f'Saved final confusion matrix to: {output}')


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    arches = ['tsrdclassifier']
    completed = load_completed_rows()
    save_progress(completed,arches=arches)
    print(f'Evaluating {8*len(arches)} members + one cross-architecture ensemble on {DEVICE}.')
    print(f'Resumable metrics CSV: {OUTPUT_CSV}')
    print(f'Progress ledger: {PROGRESS_CSV}')
    for ds in DATASETS:
        evaluate_dataset(ds, completed,arches=arches)
    write_final_confusion_matrices(completed,arches=arches)
    save_progress(completed,arches=arches)
    if len(completed) == len(DATASETS) * len(conditions()) * len(expected_models(arches=arches)):
        from src.thresholds import report
        report(OUTPUT_CSV, model=GLOBAL_ENSEMBLE)


if __name__ == '__main__':
    main()
