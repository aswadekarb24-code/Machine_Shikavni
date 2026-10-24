"""Per-severity accuracy scoreboard: python -m src.thresholds [results.csv]

Exits non-zero if any (dataset, corruption, severity) cell is below src.config.THRESHOLDS.
"""
import glob
import os
import sys

import pandas as pd

from src.config import RESULTS, THRESHOLDS

MODEL = 'trialcnn_ensemble'


def report(csv_path, model=MODEL):
    d = pd.read_csv(csv_path)
    d = d[(d['Model'] == model) & (d['Corruption'] != 'none')]
    tot = ok = 0
    print(f"Thresholds {THRESHOLDS} on {model} ({os.path.basename(csv_path)})")
    for ds, g in d.groupby('Dataset'):
        p = g.pivot(index='Corruption', columns='Severity', values='accuracy')
        passed = p.apply(lambda col: col >= THRESHOLDS[col.name])
        tot += passed.size
        ok += int(passed.values.sum())
        print(f"{ds}: {int(passed.values.sum())}/{passed.size} cells pass")
        for corr in p.index:
            if not passed.loc[corr].all():
                accs = ' '.join(f"{p.loc[corr, s]:.2f}{'' if passed.loc[corr, s] else '*'}" for s in p.columns)
                print(f"    {corr:18s} {accs}")
    print(f"TOTAL: {ok}/{tot}")
    return ok == tot


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(os.path.join(RESULTS, 'noise_results_all_models_*.csv')))
    sys.exit(0 if report(path) else 1)
